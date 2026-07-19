#include "transcriber.hpp"
#include "logger.hpp"
#include "config.hpp"
#include "whisper.h"
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cctype>

// whisper.cpp + ggml-small.en is known to confidently hallucinate canned
// phrases from silence or near-silence audio. These are the documented
// offenders — when we see one (case-insensitive, after trimming punctuation
// and whitespace), the segment is treated as if whisper returned empty.
//
// Order matters only loosely; longer / more specific phrases first so they
// match before shorter substrings like "you" or "bye".
static bool isLikelyWhisperHallucination(const std::string& raw) {
    if (raw.empty()) return true;

    // Normalize: lowercase, strip leading/trailing whitespace + punctuation.
    std::string s = raw;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    auto isJunk = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
               c == '.' || c == ',' || c == '!' || c == '?' ||
               c == ';' || c == ':' || c == '"' || c == '\'';
    };
    while (!s.empty() && isJunk(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && isJunk(static_cast<unsigned char>(s.back())))  s.pop_back();

    if (s.empty()) return true;

    // Known canned hallucinations. Sources: openai/whisper issues #1762,
    // #928, #1810, plus whisper.cpp #1287. These come from training on
    // YouTube transcripts where the same end-of-video phrases recur.
    static const char* kCanned[] = {
        "thank you", "thanks for watching", "thanks for watching!",
        "thank you for watching", "thank you so much",
        "thanks", "thank you very much",
        "bye", "bye bye", "goodbye",
        "you", "yeah", "uh", "um", "hmm",
        "[music]", "(music)", "[applause]", "(applause)",
        "[silence]", "(silence)", "[noise]", "(noise)",
        "[beep]", "(beep)", "beep",
        "[blank_audio]", "(blank_audio)",
        "[inaudible]", "(inaudible)",
        "[laughter]", "(laughter)",
        ".", "..", "...", "....",
        // Whisper sometimes outputs subtitle-style attributions:
        "subtitles by the amara.org community",
        "subtitles by amara.org community",
        "subs by www.zeoranger.co.uk",
        // The killer one for ARIA — caused random terminal opens because the
        // (now-removed) initial_prompt seeded the phrase. Keep as a final
        // safety net in case it still bleeds through.
        "and then open the terminal",
        "and then open the terminal. so, that's all. thank you",
        "and then open the terminal. so, that's it. thank you",
    };
    for (auto* p : kCanned) {
        if (s == p) return true;
    }
    return false;
}

Transcriber::Transcriber(const std::string& modelPath) {
    whisper_context_params cparams = whisper_context_default_params();
    // Phase 9: default to CPU. Wake-word now gates transcription so whisper
    // runs only on demand; keeping GPU resident would starve Ollama.
    cparams.use_gpu = Config::get().whisper_gpu;
    ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx) { Logger::error("Whisper: failed to load model: " + modelPath); return; }
    Logger::info(std::string("Whisper model loaded (") +
                 (cparams.use_gpu ? "GPU" : "CPU") + "): " + modelPath);
}

Transcriber::~Transcriber() {
    if (ctx) whisper_free(ctx);
}

static std::vector<float> loadWav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};

    char riff[4];
    f.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) return {};

    f.seekg(12); // skip "RIFF", file size, "WAVE"

    // scan chunks until "data"
    while (f) {
        char id[4];
        uint32_t chunkSize = 0;
        f.read(id, 4);
        f.read(reinterpret_cast<char*>(&chunkSize), 4);
        if (!f) break;

        if (std::strncmp(id, "data", 4) == 0) {
            std::vector<int16_t> raw(chunkSize / sizeof(int16_t));
            f.read(reinterpret_cast<char*>(raw.data()), chunkSize);
            std::vector<float> samples(raw.size());
            for (size_t i = 0; i < raw.size(); ++i)
                samples[i] = raw[i] / 32768.0f;
            return samples;
        }
        f.seekg(chunkSize, std::ios::cur); // skip unknown chunk
    }
    return {};
}

std::string Transcriber::transcribe(const std::string& wavPath) {
    if (!ctx) { Logger::error("Transcriber: no model loaded."); return ""; }

    auto samples = loadWav(wavPath);
    if (samples.empty()) {
        Logger::error("Transcriber: could not read WAV or no audio data: " + wavPath);
        return "";
    }

    // Too short to be real speech — skip whisper to save GPU cycles AND
    // avoid hallucination from noise blips. 11200 samples @ 16kHz = 0.7s;
    // anything shorter is almost certainly a fan tick, doorbell, or VAD
    // false-positive. Whisper given ~0.5s of silence-ish audio will
    // confidently output "Thank you." / "open the terminal." etc — so we
    // refuse to even ask it.
    if (samples.size() < 11200) {
        Logger::info("Transcriber: skipping short audio (" +
                     std::to_string(samples.size()) + " samples, <0.7s)");
        return "";
    }

    Logger::info("Transcribing file: " + wavPath +
                 " (" + std::to_string(samples.size()) + " samples)");

    whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.language          = "en";
    // 8 = physical core count on the Ryzen 7 7435HS. Whisper scales with
    // physical cores; 4 left half the CPU idle (Phase 0: 2.3s/utterance).
    wp.n_threads         = 8;
    wp.print_realtime    = false;
    wp.print_progress    = false;
    wp.print_timestamps  = false;
    wp.single_segment    = false;
    // 0.6 (was 0.3) — whisper-cpp's no-speech probability above which the
    // segment is dropped as non-speech. Raising this kills the well-known
    // Whisper "thank you / open the terminal / [Music]" hallucinations
    // that fire on near-silence and noise blips.
    wp.no_speech_thold   = 0.6f;
    wp.no_context        = true;  // each utterance is independent; skip KV carryover
    wp.suppress_blank    = true;
    // NO initial_prompt. The previous prompt ("open firefox, open terminal,
    // open code, ...") was actively poisoning the model — when given silence
    // it would happily output "open the terminal" because we'd primed it
    // with that exact phrase. Better to let whisper run unbiased and rely
    // on the hallucination filter + no_speech_thold to drop garbage.

    auto t0 = std::chrono::steady_clock::now();
    if (whisper_full(ctx, wp, samples.data(), (int)samples.size()) != 0) {
        ++consecutiveFailures_;
        Logger::error("Transcriber: whisper_full() failed (fail #" +
                      std::to_string(consecutiveFailures_) + ").");
        return "";
    }
    consecutiveFailures_ = 0;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    Logger::info("Transcribe finished in " + std::to_string(ms) + " ms");

    std::string result;
    int n = whisper_full_n_segments(ctx);
    for (int i = 0; i < n; ++i)
        result += whisper_full_get_segment_text(ctx, i);

    if (!result.empty() && result[0] == ' ')
        result = result.substr(1);

    if (isLikelyWhisperHallucination(result)) {
        Logger::info("Transcriber: dropped Whisper hallucination: \"" + result + "\"");
        return "";
    }

    return result;
}