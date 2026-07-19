// aria-bench — Phase 0 measurement harness. Two subcommands:
//
//   aria-bench core [--cases FILE] [--no-warmup] [--limit N]
//       Text → action/TTFT through the REAL production path (classifyIntent
//       then LLM::thinkStreaming with the production system prompt + tools).
//       No microphone, no TTS, no action execution — this is the falsifiable
//       test that cognition is benchmarkable without audio hardware.
//
//   aria-bench asr [--manifest FILE]
//       WAV → transcript through the production Transcriber (same model,
//       same hallucination filter). Reports WER, latency, real-time factor.
//
// Both write a JSON report under bench/reports/ and print a table.
// Determinism notes: core uses a FIXED LLMContext (never wall-clock) so the
// prompt is byte-stable across runs; history is cleared between cases.

#include "config.hpp"
#include "intent.hpp"
#include "llm.hpp"
#include "transcriber.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static long long msSince(Clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now() - t0).count();
}

static double percentile(std::vector<long long> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1) + 0.5);
    return static_cast<double>(v[std::min(idx, v.size() - 1)]);
}

static std::string isoNowCompact() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

static void writeReport(const std::string& kind, const json& report) {
    fs::path dir = fs::path("bench") / "reports";
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path out = dir / (kind + "_" + isoNowCompact() + ".json");
    std::ofstream f(out);
    f << report.dump(2) << "\n";
    std::cout << "\nreport: " << out.string() << "\n";
}

// ─── core ───────────────────────────────────────────────────────────────────

// Fixed context: NEVER wall-clock-dependent, so the serialized system prompt
// is identical across runs and machines-states — required for comparing
// baselines across phases.
static LLMContext benchContext() {
    LLMContext ctx;
    ctx.activeApp     = "org.gnome.Console";
    ctx.activeWindow  = "";
    ctx.clipboard     = "";
    ctx.memorySummary = "Known about user: user_name=Aurelius;";
    ctx.screenText    = "";
    ctx.tone          = "";
    ctx.timeOfDay     = "afternoon";
    ctx.dateLabel     = "Wednesday, July 9";
    ctx.notifications = "";
    return ctx;
}

struct CoreCase {
    std::string id;
    std::string input;
    std::vector<std::string> expect;   // acceptable action types; empty = speech-only
    std::string argsContains;          // optional substring of args JSON
};

static int runCore(const std::vector<std::string>& args) {
    std::string casesPath = "bench/cases_core.jsonl";
    bool warmup = true;
    size_t limit = 0;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--cases" && i + 1 < args.size()) casesPath = args[++i];
        else if (args[i] == "--no-warmup") warmup = false;
        else if (args[i] == "--limit" && i + 1 < args.size()) limit = std::stoul(args[++i]);
    }

    std::ifstream f(casesPath);
    if (!f) { std::cerr << "cannot open cases file: " << casesPath << "\n"; return 1; }

    std::vector<CoreCase> cases;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto j = json::parse(line, nullptr, false);
        if (j.is_discarded()) { std::cerr << "bad JSONL line skipped: " << line << "\n"; continue; }
        CoreCase c;
        c.id    = j.value("id", "");
        c.input = j.value("input", "");
        if (j.contains("expect"))
            for (auto& e : j["expect"]) c.expect.push_back(e.get<std::string>());
        c.argsContains = j.value("expect_args_contains", "");
        if (!c.input.empty()) cases.push_back(std::move(c));
    }
    if (limit && cases.size() > limit) cases.resize(limit);

    const auto& cfg = Config::get();
    std::cout << "core bench: model=" << cfg.ollama_model
              << " cases=" << cases.size() << "\n";

    // No Memory: history seeding is disabled anyway and the bench must not
    // touch the real DB.
    LLM llm(cfg.ollama_model, nullptr);
    if (!llm.isAvailable()) {
        std::cerr << "Ollama unreachable at " << cfg.ollama_url << " — start it first.\n";
        return 1;
    }

    LLMContext ctx = benchContext();

    // Cold-start probe / warmup. With no keep_alive configured (the Phase-1
    // target), this measures the model-load penalty every user turn can hit.
    long long coldMs = -1;
    if (warmup) {
        auto t0 = Clock::now();
        llm.thinkStreaming("Reply with the single word: ok", ctx,
                           [](const std::string&) {});
        coldMs = msSince(t0);
        llm.clearHistory();
        std::cout << "warmup (incl. any model load): " << coldMs << " ms\n\n";
    }

    json results = json::array();
    int passed = 0, intentHits = 0;
    std::vector<long long> ttfts, totals;

    for (auto& c : cases) {
        std::string kind, actualTypes, speechHead;
        long long ttft = -1, total = -1;
        std::vector<std::string> actual;
        json actionArgs = json::array();

        AgentAction direct = classifyIntent(c.input);
        if (!direct.empty()) {
            kind = "intent";
            ++intentHits;
            actual.push_back(direct.type);
            actionArgs.push_back(direct.args);
            total = 0;
        } else {
            kind = "llm";
            llm.clearHistory();
            auto t0 = Clock::now();
            long long firstTok = -1;
            auto resp = llm.thinkStreaming(c.input, ctx,
                [&](const std::string&) {
                    if (firstTok < 0) firstTok = msSince(t0);
                });
            total = msSince(t0);
            ttft  = firstTok;
            for (auto& a : resp.actions) {
                actual.push_back(a.type);
                actionArgs.push_back(a.args);
            }
            speechHead = resp.speech.substr(0, 60);
            if (ttft >= 0)  ttfts.push_back(ttft);
            totals.push_back(total);
        }

        // Scoring: expected empty → no action allowed. Otherwise at least one
        // produced action type must be in the accepted set (and args must
        // contain the substring when specified).
        bool ok;
        if (c.expect.empty()) {
            ok = actual.empty();
        } else {
            ok = false;
            for (size_t i = 0; i < actual.size(); ++i) {
                bool typeOk = std::find(c.expect.begin(), c.expect.end(),
                                        actual[i]) != c.expect.end();
                bool argsOk = c.argsContains.empty() ||
                              actionArgs[i].dump().find(c.argsContains) != std::string::npos;
                if (typeOk && argsOk) { ok = true; break; }
            }
        }
        if (ok) ++passed;

        for (auto& a : actual) actualTypes += (actualTypes.empty() ? "" : ",") + a;
        std::cout << (ok ? "PASS " : "FAIL ")
                  << c.id << " [" << kind << "]"
                  << " → " << (actualTypes.empty() ? "speech" : actualTypes);
        if (ttft >= 0) std::cout << "  ttft=" << ttft << "ms total=" << total << "ms";
        if (!ok && !speechHead.empty()) std::cout << "  said=\"" << speechHead << "\"";
        std::cout << "\n";

        results.push_back({
            {"id", c.id}, {"kind", kind}, {"pass", ok},
            {"actual", actual}, {"ttft_ms", ttft}, {"total_ms", total},
        });
    }

    json report;
    report["kind"]        = "core";
    report["model"]       = cfg.ollama_model;
    report["cases"]       = cases.size();
    report["passed"]      = passed;
    report["pass_rate"]   = cases.empty() ? 0.0 : double(passed) / cases.size();
    report["intent_hits"] = intentHits;
    report["cold_ms"]     = coldMs;
    report["ttft_p50_ms"] = percentile(ttfts, 0.50);
    report["ttft_p95_ms"] = percentile(ttfts, 0.95);
    report["total_p50_ms"]= percentile(totals, 0.50);
    report["total_p95_ms"]= percentile(totals, 0.95);
    report["results"]     = results;

    std::cout << "\n== core summary ==\n"
              << "pass: " << passed << "/" << cases.size()
              << "  (intent fast-path: " << intentHits << ")\n"
              << "cold/warmup: " << coldMs << " ms\n"
              << "LLM ttft  p50=" << report["ttft_p50_ms"]  << "ms  p95=" << report["ttft_p95_ms"]  << "ms\n"
              << "LLM total p50=" << report["total_p50_ms"] << "ms  p95=" << report["total_p95_ms"] << "ms\n";

    writeReport("core", report);
    return 0;
}

// ─── asr ────────────────────────────────────────────────────────────────────

static std::vector<std::string> normWords(const std::string& s) {
    std::vector<std::string> out;
    std::string w;
    // Digit↔word equivalence: whisper freely emits "2" for "two"; ARIA treats
    // them identically, so WER must too.
    auto push = [&](std::string word) {
        static const std::pair<const char*, const char*> kNum[] = {
            {"0","zero"},{"1","one"},{"2","two"},{"3","three"},{"4","four"},
            {"5","five"},{"6","six"},{"7","seven"},{"8","eight"},{"9","nine"},
            {"10","ten"},
        };
        for (auto& [d, n] : kNum)
            if (word == d) { word = n; break; }
        out.push_back(std::move(word));
    };
    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalnum(c) || ch == '\'') w += char(std::tolower(c));
        else if (!w.empty()) { push(w); w.clear(); }
    }
    if (!w.empty()) push(w);
    return out;
}

// Word-level Levenshtein distance (substitution/insert/delete each cost 1).
static size_t editDistance(const std::vector<std::string>& a,
                           const std::vector<std::string>& b) {
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            size_t sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, sub});
        }
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

static int runAsr(const std::vector<std::string>& args) {
    std::string manifestPath = "bench/wavs/manifest.tsv";
    for (size_t i = 0; i < args.size(); ++i)
        if (args[i] == "--manifest" && i + 1 < args.size()) manifestPath = args[++i];

    std::ifstream mf(manifestPath);
    if (!mf) {
        std::cerr << "cannot open manifest: " << manifestPath
                  << "\n(generate synthetic corpus: bench/make_asr_corpus.sh, or point"
                  << "\n --manifest at ~/.local/share/aria/bench_corpus/manifest.tsv"
                  << " after correcting its transcripts by hand)\n";
        return 1;
    }
    fs::path baseDir = fs::path(manifestPath).parent_path();

    const auto& cfg = Config::get();
    std::cout << "asr bench: model=" << cfg.whisper_model
              << " gpu=" << (cfg.whisper_gpu ? "yes" : "no") << "\n";
    Transcriber tr(cfg.whisper_model);

    json results = json::array();
    std::vector<long long> lats;
    size_t totalRefWords = 0, totalEdits = 0;
    int n = 0, exact = 0;

    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        fs::path wav = baseDir / line.substr(0, tab);
        std::string ref = line.substr(tab + 1);
        if (!fs::exists(wav)) { std::cerr << "missing wav: " << wav << "\n"; continue; }

        double audioSec = 0;
        std::error_code ec;
        auto sz = fs::file_size(wav, ec);
        if (!ec && sz > 44) audioSec = double(sz - 44) / 32000.0; // 16kHz mono s16le

        auto t0 = Clock::now();
        std::string hyp = tr.transcribe(wav.string());
        long long ms = msSince(t0);
        lats.push_back(ms);

        auto refW = normWords(ref), hypW = normWords(hyp);
        size_t ed = editDistance(refW, hypW);
        totalRefWords += refW.size();
        totalEdits    += ed;
        double wer = refW.empty() ? 0.0 : double(ed) / refW.size();
        bool match = (refW == hypW);
        if (match) ++exact;
        ++n;

        std::cout << (match ? "OK   " : "MISS ") << wav.filename().string()
                  << "  " << ms << "ms";
        if (audioSec > 0) {
            char rtf[16];
            std::snprintf(rtf, sizeof(rtf), " rtf=%.2f", ms / 1000.0 / audioSec);
            std::cout << rtf;
        }
        std::cout << "  wer=" << wer;
        if (!match) std::cout << "\n      ref: " << ref << "\n      hyp: " << hyp;
        std::cout << "\n";

        results.push_back({
            {"wav", wav.filename().string()}, {"ref", ref}, {"hyp", hyp},
            {"wer", wer}, {"ms", ms}, {"audio_sec", audioSec},
        });
    }

    if (n == 0) { std::cerr << "no usable manifest entries\n"; return 1; }

    double aggWer = totalRefWords ? double(totalEdits) / totalRefWords : 0.0;
    json report;
    report["kind"]         = "asr";
    report["model"]        = cfg.whisper_model;
    report["gpu"]          = cfg.whisper_gpu;
    report["files"]        = n;
    report["exact"]        = exact;
    report["wer"]          = aggWer;
    report["lat_p50_ms"]   = percentile(lats, 0.50);
    report["lat_p95_ms"]   = percentile(lats, 0.95);
    report["results"]      = results;

    std::cout << "\n== asr summary ==\n"
              << "files: " << n << "  exact: " << exact << "/" << n
              << "  WER: " << aggWer << "\n"
              << "latency p50=" << report["lat_p50_ms"] << "ms"
              << "  p95=" << report["lat_p95_ms"] << "ms\n";

    writeReport("asr", report);
    return 0;
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: aria-bench core [--cases FILE] [--no-warmup] [--limit N]\n"
                     "       aria-bench asr  [--manifest FILE]\n";
        return 2;
    }
    std::string cmd = argv[1];
    std::vector<std::string> rest(argv + 2, argv + argc);
    if (cmd == "core") return runCore(rest);
    if (cmd == "asr")  return runAsr(rest);
    std::cerr << "unknown subcommand: " << cmd << "\n";
    return 2;
}
