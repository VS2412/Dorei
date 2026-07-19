#include "tts.hpp"
#include "logger.hpp"
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <csignal>
#include <regex>
#include <unistd.h>
#include <sys/wait.h>

static long long ttsNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Make text safe for Piper to speak. Piper voices punctuation literally —
// "tilde slash", "asterisk dot pdf", "backslash open paren" — which destroys
// the user's ear. So we run shell/JSON noise through a pipeline of
// substitutions:
//   - JSON braces/brackets/quotes around args  → drop
//   - long absolute paths                       → basename only
//   - "~/foo/bar/baz.pdf"                       → "baz.pdf"
//   - shell glob/escape jank ("*.pdf", "\(", "{}") → human words or drop
//   - long blobs of code (>200 chars, no spaces) → "[output omitted]"
// This runs at every speak entry so no caller can leak raw shell output.
static std::string sanitizeForSpeech(const std::string& in) {
    if (in.empty()) return in;
    std::string s = in;

    // Strip JSON wrappers — {"command":"..."} → ...
    s = std::regex_replace(s, std::regex(R"(\{\s*\"[a-zA-Z_]+\"\s*:\s*\")"), " ");
    s = std::regex_replace(s, std::regex(R"(\"\s*\})"), " ");
    s = std::regex_replace(s, std::regex(R"(\"\s*,\s*\"[a-zA-Z_]+\"\s*:\s*\")"), " and ");

    // Long absolute path → basename only. Walk the string and replace any
    // "/a/b/c.ext" (≥2 slashes) with just "c.ext" so Piper says "report.pdf"
    // not "slash home slash Aurelius slash Downloads slash report.pdf".
    {
        std::string out;
        out.reserve(s.size());
        size_t i = 0;
        while (i < s.size()) {
            if (s[i] == '~' || (s[i] == '/' && (i == 0 || !std::isalnum(static_cast<unsigned char>(s[i-1]))))) {
                size_t start = i;
                size_t slashes = 0;
                size_t j = i;
                while (j < s.size() &&
                       (std::isalnum(static_cast<unsigned char>(s[j])) ||
                        s[j] == '/' || s[j] == '_' || s[j] == '-' || s[j] == '.' || s[j] == '~')) {
                    if (s[j] == '/') ++slashes;
                    ++j;
                }
                if (slashes >= 2 && j - start > 6) {
                    // Take basename
                    size_t lastSlash = s.rfind('/', j - 1);
                    if (lastSlash != std::string::npos && lastSlash + 1 < j) {
                        out += s.substr(lastSlash + 1, j - lastSlash - 1);
                    }
                    i = j;
                    continue;
                }
            }
            out += s[i++];
        }
        s = out;
    }

    // Common shell tokens → drop / replace
    s = std::regex_replace(s, std::regex(R"(\\\()"), "");
    s = std::regex_replace(s, std::regex(R"(\\\))"), "");
    s = std::regex_replace(s, std::regex(R"(-(maxdepth|iname|exec|type|name)\s+)"), " ");
    s = std::regex_replace(s, std::regex(R"(\{\}\s*\+)"), "");
    s = std::regex_replace(s, std::regex(R"(\*\.([a-zA-Z0-9]+))"), "$1 files");
    s = std::regex_replace(s, std::regex(R"(\b-rf?\b)"), "");
    // Strip punctuation that Piper voices literally — but preserve apostrophes
    // and quotes that sit between two letters (contractions: "What's", "don't")
    // so they remain natural in speech.
    {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '`' || c == '\'' || c == '"') {
                bool prevAlpha = (i > 0) &&
                    std::isalpha(static_cast<unsigned char>(s[i - 1]));
                bool nextAlpha = (i + 1 < s.size()) &&
                    std::isalpha(static_cast<unsigned char>(s[i + 1]));
                if (prevAlpha && nextAlpha) { out += c; continue; }
                out += ' ';
            } else if (c == '[' || c == ']' || c == '{' || c == '}' ||
                       c == '|' || c == '<' || c == '>') {
                out += ' ';
            } else {
                out += c;
            }
        }
        s = std::move(out);
    }
    s = std::regex_replace(s, std::regex(R"(\bmv -t\b)"), "move into");
    s = std::regex_replace(s, std::regex(R"(\bmkdir -p\b)"), "create folders");
    s = std::regex_replace(s, std::regex(R"(\bfind\b)"), "find ");

    // Collapse whitespace
    s = std::regex_replace(s, std::regex(R"(\s+)"), " ");
    auto trimL = s.find_first_not_of(' ');
    if (trimL == std::string::npos) return "";
    s = s.substr(trimL);
    auto trimR = s.find_last_not_of(' ');
    s = s.substr(0, trimR + 1);

    // If after sanitizing we still have a giant code-blob with few spaces,
    // bail out — better silence than gibberish.
    if (s.size() > 200) {
        size_t spaces = std::count(s.begin(), s.end(), ' ');
        if (spaces < s.size() / 12) return "[output omitted]";
    }

    return s;
}

// Strip UTF-8 sequences of 3+ bytes (covers all emoji planes, dingbats,
// smart quotes/dashes that Piper mispronounces). Keeps ASCII + 2-byte
// Latin-1/accented letters intact.
static std::string stripHighUnicode(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ) {
        unsigned char c = static_cast<unsigned char>(in[i]);
        if (c < 0x80) { out.push_back(in[i]); i += 1; continue; }
        if ((c & 0xE0) == 0xC0) { // 2-byte UTF-8 — keep
            if (i + 1 < in.size()) { out.push_back(in[i]); out.push_back(in[i+1]); }
            i += 2; continue;
        }
        if ((c & 0xF0) == 0xE0) { i += 3; continue; } // 3-byte — drop
        if ((c & 0xF8) == 0xF0) { i += 4; continue; } // 4-byte — drop
        i += 1; // stray continuation byte — skip
    }
    return out;
}

namespace fs = std::filesystem;

TTS::TTS(const std::string& modelPath) : modelPath_(modelPath) {
    if (system("which piper > /dev/null 2>&1") != 0) {
        Logger::warn("TTS: piper not found."); return;
    }
    if (!fs::exists(modelPath)) {
        Logger::warn("TTS: model not found: " + modelPath); return;
    }
    available_ = true;
    Logger::info("TTS: ready.");
}

TTS::~TTS() {
    interrupt();
    // Make sure playback thread is joined if still running
    if (playbackThread_.joinable()) {
        stopPlayback_.store(true);
        queueCV_.notify_all();
        playbackThread_.join();
    }
}

void TTS::runAndTrack(const std::string& cmd) {
    pid_t pid = fork();
    if (pid == 0) {
        execl("/bin/sh", "sh", "-c", cmd.c_str(), nullptr);
        _exit(1);
    }
    if (pid < 0) return;
    activePid_.store(pid);
    int status;
    waitpid(pid, &status, 0);
    activePid_.store(-1);
}

// Speak a single sentence: piper WAV piped through stdout to pw-play
void TTS::speakSentence(const std::string& text) {
    if (text.empty() || !available_) return;

    // Belt-and-suspenders: strip emoji/3+ byte UTF-8 for any path that
    // bypasses feedChunk (batch speak, timer announcements, etc.), then
    // sanitize for shell/JSON noise so Piper never voices "tilde slash" or
    // raw command-line escape sequences.
    std::string clean = sanitizeForSpeech(stripHighUnicode(text));
    auto s = clean.find_first_not_of(" \t\n\r");
    auto e = clean.find_last_not_of(" \t\n\r");
    if (s == std::string::npos) return;
    clean = clean.substr(s, e - s + 1);
    if (clean.empty()) return;

    // Write text to temp file (avoids shell escaping issues with echo)
    std::ofstream f("/tmp/aria_tts_in.txt");
    if (!f) { Logger::error("TTS: cannot write temp file"); return; }
    f << clean;
    f.close();

    Logger::info("TTS: speaking → " + clean.substr(0, 80));
    lastSpeakStartMs_.store(ttsNowMs());

    // Pipe WAV from piper stdout directly into pw-play — no intermediate file
    std::string cmd =
        "piper --model " + modelPath_ +
        " --output_file /dev/stdout --quiet"
        " < /tmp/aria_tts_in.txt 2>/dev/null"
        " | pw-play - 2>/dev/null";

    runAndTrack(cmd);
    lastEndedMs_.store(ttsNowMs());
}

long long TTS::msSinceLastSpeech() const {
    if (speaking_.load()) return 0;
    long long end = lastEndedMs_.load();
    if (end == 0) return 1LL << 30;  // never spoken — return "long ago"
    return ttsNowMs() - end;
}

bool TTS::wasRecentlySpeaking(long long withinMs) const {
    if (speaking_.load()) return true;
    return msSinceLastSpeech() < withinMs;
}

// ─── Batch API ───

void TTS::speak(const std::string& text) {
    if (text.empty()) return;
    if (!available_) { Logger::info("TTS (silent): " + text); return; }

    speaking_.store(true);
    speakSentence(text);
    speaking_.store(false);
}

// ─── Streaming API ───

void TTS::startStream() {
    // Clean up any previous playback thread
    if (playbackThread_.joinable()) {
        stopPlayback_.store(true);
        queueCV_.notify_all();
        playbackThread_.join();
    }

    streamBuffer_.clear();
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        std::queue<std::string> empty;
        sentenceQueue_.swap(empty);
    }

    stopPlayback_.store(false);
    streaming_.store(true);
    speaking_.store(true);

    // Start the playback thread
    playbackThread_ = std::thread(&TTS::playbackLoop, this);
}

void TTS::feedChunk(const std::string& text) {
    if (!streaming_.load() || text.empty()) return;

    streamBuffer_ += stripHighUnicode(text);

    // Check for sentence boundaries and enqueue complete sentences
    // Sentence ends: ". " "? " "! " or newline, or buffer > 120 chars
    while (true) {
        size_t breakPos = std::string::npos;

        // Look for sentence-ending punctuation followed by space or end
        for (size_t i = 0; i < streamBuffer_.size(); i++) {
            char c = streamBuffer_[i];
            if (c == '.' || c == '?' || c == '!') {
                bool atEnd    = (i + 1 >= streamBuffer_.size());
                bool prevDig  = (i > 0 && std::isdigit(static_cast<unsigned char>(streamBuffer_[i-1])));
                bool nextDig  = !atEnd && std::isdigit(static_cast<unsigned char>(streamBuffer_[i+1]));

                // Decimal point like "8.1" — never a sentence boundary.
                if (c == '.' && prevDig && nextDig) continue;

                // Trailing "8." at buffer end — could be a decimal waiting for
                // the next streamed chunk. Defer; wait for more content.
                if (c == '.' && prevDig && atEnd) { breakPos = std::string::npos; break; }

                // Check if followed by space, newline, or end of buffer
                if (atEnd || streamBuffer_[i + 1] == ' ' ||
                    streamBuffer_[i + 1] == '\n') {
                    breakPos = i + 1;
                    break;
                }
            }
            if (c == '\n') {
                breakPos = i + 1;
                break;
            }
        }

        // Force break on long buffers
        if (breakPos == std::string::npos && streamBuffer_.size() > 120) {
            // Find last space to break on
            size_t spacePos = streamBuffer_.rfind(' ', 120);
            breakPos = (spacePos != std::string::npos && spacePos > 20)
                       ? spacePos + 1 : 120;
        }

        if (breakPos == std::string::npos) break;

        std::string sentence = streamBuffer_.substr(0, breakPos);
        streamBuffer_ = streamBuffer_.substr(breakPos);

        // Trim
        auto s = sentence.find_first_not_of(" \t\n\r");
        auto e = sentence.find_last_not_of(" \t\n\r");
        if (s == std::string::npos) continue;
        sentence = sentence.substr(s, e - s + 1);

        if (!sentence.empty()) {
            std::lock_guard<std::mutex> lock(queueMutex_);
            sentenceQueue_.push(sentence);
            queueCV_.notify_one();
        }
    }
}

void TTS::flushBuffer() {
    // Push whatever remains in the buffer
    auto s = streamBuffer_.find_first_not_of(" \t\n\r");
    auto e = streamBuffer_.find_last_not_of(" \t\n\r");
    if (s != std::string::npos) {
        std::string remaining = streamBuffer_.substr(s, e - s + 1);
        if (!remaining.empty()) {
            std::lock_guard<std::mutex> lock(queueMutex_);
            sentenceQueue_.push(remaining);
            queueCV_.notify_one();
        }
    }
    streamBuffer_.clear();
}

void TTS::endStream() {
    if (!streaming_.load()) return;

    // Flush remaining buffer as final sentence
    flushBuffer();

    // Signal end of stream
    streaming_.store(false);
    queueCV_.notify_all();

    // Wait for playback thread to finish
    if (playbackThread_.joinable())
        playbackThread_.join();

    speaking_.store(false);
}

void TTS::playbackLoop() {
    while (true) {
        std::string sentence;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCV_.wait(lock, [this] {
                return !sentenceQueue_.empty() || !streaming_.load() || stopPlayback_.load();
            });

            if (stopPlayback_.load()) {
                // Drain queue and exit
                std::queue<std::string> empty;
                sentenceQueue_.swap(empty);
                return;
            }

            if (sentenceQueue_.empty()) {
                // Stream ended and queue is drained
                if (!streaming_.load()) return;
                continue;
            }

            sentence = sentenceQueue_.front();
            sentenceQueue_.pop();
        }

        if (stopPlayback_.load()) return;

        speakSentence(sentence);

        if (stopPlayback_.load()) return;
    }
}

void TTS::interrupt() {
    streaming_.store(false);
    stopPlayback_.store(true);
    speaking_.store(false);

    // Kill active piper/pw-play process
    pid_t pid = activePid_.load();
    if (pid > 0) {
        kill(pid, SIGTERM);
        activePid_.store(-1);
    }

    // Kill any lingering processes
    system("pkill -f 'pw-play.*/dev/null' 2>/dev/null");
    system("pkill -f 'piper.*output_file' 2>/dev/null");

    // Drain the sentence queue
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        std::queue<std::string> empty;
        sentenceQueue_.swap(empty);
    }
    queueCV_.notify_all();

    // Wait for playback thread to finish
    if (playbackThread_.joinable())
        playbackThread_.join();

    lastEndedMs_.store(ttsNowMs());
}
