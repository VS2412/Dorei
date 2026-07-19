#include "daemon.hpp"
#include "logger.hpp"
#include "config.hpp"
#include "recorder.hpp"
#include "wakeword.hpp"
#include "transcriber.hpp"
#include "llm.hpp"
#include "intent.hpp"
#include "executor.hpp"
#include "tts.hpp"
#include "context.hpp"
#include "memory.hpp"
#include "timer.hpp"
#include "vision.hpp"
#include "planner.hpp"
#include <thread>
#include <chrono>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>
#include <map>
#include <regex>
#include <sstream>
#include <ctime>
#include <cstdio>
#include <optional>

Daemon::Daemon(std::atomic<bool>& flag, std::atomic<bool>& pause)
    : shutdownRequested(flag), pauseToggle(pause) {}

// ─── Phase 0 instrumentation ───

long long Daemon::nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Emit the single per-turn budget line. Convention: a stage that didn't run
// is omitted; `first_audio_ms` is speech-end → first piper spawn (the KPI),
// `turn_total_ms` additionally includes full TTS playback + settle.
void Daemon::logTurn(const std::string& text) {
    nlohmann::json j;
    std::string t = text.substr(0, 80);
    j["text"]    = t;
    j["path"]    = turn_.path.empty() ? "unknown" : turn_.path;
    if (!turn_.outcome.empty()) j["outcome"] = turn_.outcome;
    if (turn_.asrDone)    j["asr_ms"]  = turn_.asrDone - turn_.speechEnd;
    if (turn_.ctxDone)    j["ctx_ms"]  = turn_.ctxDone - turn_.asrDone;
    if (turn_.firstToken) j["ttft_ms"] = turn_.firstToken - turn_.llmStart;
    if (turn_.llmDone)    j["llm_ms"]  = turn_.llmDone - turn_.llmStart;
    if (tts_) {
        long long ttsStart = tts_->lastSpeakStartMs();
        if (ttsStart >= turn_.speechEnd)
            j["first_audio_ms"] = ttsStart - turn_.speechEnd;
    }
    j["turn_total_ms"] = nowMs() - turn_.speechEnd;
    Logger::info("TURN " + j.dump());
}

// Copy the utterance WAV + its transcript into the bench corpus so the ASR
// bench can score against the user's real voice. 16kHz mono s16le, ~32KB/s,
// keep-last-200 → worst case ~130MB, typically far less.
void Daemon::maybeArchiveUtterance(const std::string& wavPath,
                                   const std::string& text) {
    if (!benchCapture_) return;
    namespace fs = std::filesystem;
    try {
        const char* home = std::getenv("HOME");
        if (!home) return;
        fs::path dir = fs::path(home) / ".local/share/aria/bench_corpus";
        fs::create_directories(dir);

        std::string name = "turn_" + std::to_string(nowMs()) + ".wav";
        fs::copy_file(wavPath, dir / name,
                      fs::copy_options::overwrite_existing);

        std::string clean = text;
        for (char& c : clean)
            if (c == '\t' || c == '\n' || c == '\r') c = ' ';
        std::ofstream mf(dir / "manifest.tsv", std::ios::app);
        mf << name << '\t' << clean << '\n';

        // Rotation: epoch-ms names sort lexicographically == chronologically.
        std::vector<fs::path> wavs;
        for (auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".wav") wavs.push_back(e.path());
        if (wavs.size() > 200) {
            std::sort(wavs.begin(), wavs.end());
            for (size_t i = 0; i + 200 < wavs.size(); ++i)
                fs::remove(wavs[i]);
        }
    } catch (const std::exception& e) {
        Logger::warn(std::string("Corpus capture failed: ") + e.what());
    }
}

// ─── Static helpers (unchanged from v2) ───

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

static bool isInterruptCommand(const std::string& text) {
    auto t = lower(text);
    // Anything that should kill ARIA's current speech / task immediately.
    // Kept generous on purpose — false positives just stop TTS, which the
    // user can recover from by saying their request again.
    return t.find("stop")        != std::string::npos ||
           t.find("cancel")      != std::string::npos ||
           t.find("shut up")     != std::string::npos ||
           t.find("quiet")       != std::string::npos ||
           t.find("enough")      != std::string::npos ||
           t.find("wait")        != std::string::npos ||
           t.find("hold on")     != std::string::npos ||
           t.find("never mind")  != std::string::npos ||
           t.find("nevermind")   != std::string::npos ||
           t.find("forget it")   != std::string::npos ||
           t.find("abort")       != std::string::npos ||
           t.find("listen to me") != std::string::npos;
}

// Conversational small-talk that should NEVER trigger a tool call. ARIA's
// LLM was hallucinating tool calls on phrases like "you're ready to go!"
// (calling system_info) and "running smooth" (calling sysinfo). When the
// utterance fits one of these patterns, short-circuit before the LLM and
// reply with a stock acknowledgement.
static const char* matchSmallTalk(const std::string& text) {
    auto t = lower(text);
    auto trimmed = t;
    while (!trimmed.empty() && (trimmed.back()  == '.' || trimmed.back()  == '!' ||
                                trimmed.back()  == '?' || trimmed.back()  == ' ' ||
                                trimmed.back()  == ',' || trimmed.back()  == '\n')) trimmed.pop_back();
    while (!trimmed.empty() && trimmed.front() == ' ') trimmed.erase(trimmed.begin());

    static const std::pair<std::regex, const char*> kSmallTalk[] = {
        { std::regex(R"(^(ok(ay)?|kk|cool|nice|sweet|got it|gotcha|sounds? good|alright|all right)$)", std::regex::icase), "Got it." },
        { std::regex(R"(^(thanks|thank you|thx|ty|appreciate it|much appreciated)$)", std::regex::icase), "Anytime." },
        { std::regex(R"(^(you'?re ready( to go)?|you ready|ready to go|are you ready)$)", std::regex::icase), "Ready when you are." },
        { std::regex(R"(^(running smooth|all good|no problem|np)$)", std::regex::icase), "Glad to hear it." },
        { std::regex(R"(^(good (morning|afternoon|evening|night)|morning|evening)$)", std::regex::icase), "Morning." },
        { std::regex(R"(^(bye|goodbye|see you|cya|talk later)$)", std::regex::icase), "Later." },
        { std::regex(R"(^(hello|hi|hey|hiya|yo)( there)?$)", std::regex::icase), "Hey." },
        { std::regex(R"(^(how are you|how'?s it going|how you doing|what'?s up|what'?s good|sup)( today)?$)", std::regex::icase), "All good. You?" },
        { std::regex(R"(^(love you|love ya)$)", std::regex::icase), "Likewise." },
        { std::regex(R"(^(you'?re (the )?best|nice work|good job|well done)$)", std::regex::icase), "Thanks." },
        { std::regex(R"(^(yes|yeah|yep|yup|sure|of course)$)", std::regex::icase), "Got it." },
        { std::regex(R"(^(no|nope|nah|not really)$)", std::regex::icase), "Okay." },
    };
    for (auto& [re, reply] : kSmallTalk) {
        if (std::regex_match(trimmed, re)) return reply;
    }
    return nullptr;
}

// Wake-word detection. Whisper often mis-hears "aria" as "area", "ariah",
// "arya", "ariya" — so we accept a small family. Returns the suffix AFTER
// the wake phrase (possibly empty if user just said the name).
// nullopt means no wake phrase was found.
static std::optional<std::string> stripWakeWord(const std::string& text) {
    static const std::regex wakeRe(
        R"(^\s*(?:hey[,\s]+|ok[,\s]+|okay[,\s]+)?(?:aria|arya|ariah|ariya|area)[,\s.!?:-]*)",
        std::regex::icase);
    std::smatch m;
    if (!std::regex_search(text, m, wakeRe)) return std::nullopt;
    if (m.position(0) != 0) return std::nullopt;
    std::string rest = text.substr(m.position(0) + m.length(0));
    while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.erase(rest.begin());
    while (!rest.empty() && (rest.back()  == ' ' || rest.back()  == '\t')) rest.pop_back();
    return rest;
}

static int wordCount(const std::string& text) {
    int count = 0;
    bool inWord = false;
    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            if (!inWord) { ++count; inWord = true; }
        } else {
            inWord = false;
        }
    }
    return count;
}

// Passive fact extraction: scan user speech for self-identifying phrases.
static std::map<std::string, std::string> extractFacts(const std::string& text) {
    auto t = lower(text);
    std::map<std::string, std::string> facts;

    auto tryExtract = [&](const std::string& prefix) -> std::string {
        auto pos = t.find(prefix);
        if (pos == std::string::npos) return "";
        std::string rest = text.substr(pos + prefix.size());
        auto end = rest.find_first_of(".,!?;\n");
        if (end != std::string::npos) rest = rest.substr(0, end);
        while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.erase(rest.begin());
        while (!rest.empty() && (rest.back() == ' ' || rest.back() == '\t')) rest.pop_back();
        return rest;
    };

    auto firstMatch = [&](std::initializer_list<const char*> prefixes) -> std::string {
        for (auto p : prefixes) {
            auto v = tryExtract(p);
            if (!v.empty()) return v;
        }
        return "";
    };

    auto name = firstMatch({"my name is ", "call me ", "i'm called ", "i am called "});
    if (!name.empty() && name.size() < 40) facts["user_name"] = name;

    auto loc = firstMatch({"i live in ", "i am from ", "i'm from ", "i'm in ", "i am in "});
    if (!loc.empty() && loc.size() < 50) facts["location"] = loc;

    auto job = firstMatch({"i work as ", "i work at ", "my job is "});
    if (!job.empty() && job.size() < 80) facts["job"] = job;

    auto pref = firstMatch({"i prefer ", "i like using ", "i swear by ", "i love ", "i really like "});
    if (!pref.empty() && pref.size() < 80) facts["preference"] = pref;

    auto proj = firstMatch({"i'm working on ", "i am working on ", "my project is ", "i'm building "});
    if (!proj.empty() && proj.size() < 120) facts["current_project"] = proj;

    {
        std::regex favRe(R"(my favou?rite (\w+(?:\s+\w+)?) (?:is|are) ([^.,!?;\n]+))",
                         std::regex::icase);
        std::smatch m;
        if (std::regex_search(text, m, favRe)) {
            std::string topic = m[1].str();
            std::string val   = m[2].str();
            std::transform(topic.begin(), topic.end(), topic.begin(), ::tolower);
            for (auto& c : topic) if (c == ' ') c = '_';
            while (!val.empty() && (val.back()  == ' ' || val.back()  == '\t')) val.pop_back();
            while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());
            if (!val.empty() && val.size() < 80)
                facts["favorite_" + topic] = val;
        }
    }

    return facts;
}

// Phase 8.6: Passive entity extraction — captures URLs, emails, paths, @handles, #tags
// from user utterances so the entity graph builds up across sessions.
// Cheap (regex-only, no LLM call); more nuanced extraction (people, projects) will
// layer on via LLM later when we have a reason to pay for it.
static std::vector<std::pair<std::string, std::string>> extractEntities(const std::string& text) {
    std::vector<std::pair<std::string, std::string>> out;

    auto scan = [&](const std::regex& re, const char* type) {
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string m = it->str();
            if (m.size() > 1 && m.size() < 200) out.emplace_back(m, type);
        }
    };

    static const std::regex urlRe(R"(https?://[^\s]+)");
    static const std::regex emailRe(R"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})");
    static const std::regex pathRe(R"(/[A-Za-z0-9_][A-Za-z0-9_./-]{3,})");       // absolute paths
    static const std::regex handleRe(R"(@[A-Za-z0-9_]{2,30}\b)");
    static const std::regex tagRe(R"(#[A-Za-z][A-Za-z0-9_-]{1,40}\b)");

    scan(urlRe,    "url");
    scan(emailRe,  "email");
    scan(pathRe,   "path");
    scan(handleRe, "handle");
    scan(tagRe,    "tag");
    return out;
}

static std::string detectTone(const std::string& text) {
    auto t = lower(text);
    static const std::regex frustRe(
        R"((damn|dammit|shit|crap|wtf|ugh|come on|why isn'?t|not working|broken|still not|again\?|seriously|whatever))",
        std::regex::icase);
    static const std::regex urgRe(
        R"(\b(now|asap|quickly|quick|fast|hurry|right now|immediately)\b)",
        std::regex::icase);
    static const std::regex curRe(
        R"(\b(how does|how do|what is|why does|why is|explain|tell me about|curious|wonder(?:ing)?)\b)",
        std::regex::icase);
    if (std::regex_search(t, frustRe)) return "frustrated";
    if (std::regex_search(t, urgRe))   return "urgent";
    if (std::regex_search(t, curRe))   return "curious";
    return "";
}

static std::string currentTimeOfDay() {
    std::time_t now = std::time(nullptr);
    std::tm lt{};
    localtime_r(&now, &lt);
    int h = lt.tm_hour;
    if (h < 5)  return "night";
    if (h < 12) return "morning";
    if (h < 17) return "afternoon";
    if (h < 21) return "evening";
    return "night";
}

static std::string currentDateLabel() {
    std::time_t now = std::time(nullptr);
    std::tm lt{};
    localtime_r(&now, &lt);
    char buf[64];
    if (std::strftime(buf, sizeof(buf), "%A, %B %e", &lt) == 0) return "";
    return buf;
}

static std::string arg(const nlohmann::json& a, const std::string& key) {
    if (a.contains(key) && a[key].is_string()) return a[key].get<std::string>();
    if (a.contains("param") && a["param"].is_string()) return a["param"].get<std::string>();
    return "";
}

// ─── Action dispatch ───

std::string Daemon::executeAction(const AgentAction& act) {
    // Timer actions are owned by the daemon's TimerManager.
    if (act.type == "timer_set") {
        int secs = 0;
        if (act.args.contains("duration_seconds") && act.args["duration_seconds"].is_number())
            secs = act.args["duration_seconds"].get<int>();
        std::string label = arg(act.args, "label");
        if (secs <= 0) return "Invalid duration.";
        int id = timers_->addTimer(secs, label);
        return "Timer #" + std::to_string(id) + " set for " + std::to_string(secs) + " seconds.";
    }
    if (act.type == "timer_list")   return timers_->list();
    if (act.type == "timer_cancel") {
        int id = 0;
        if (act.args.contains("id") && act.args["id"].is_number())
            id = act.args["id"].get<int>();
        return timers_->cancel(id) ? "Timer cancelled." : "Timer not found.";
    }

    // Memory actions — owned by the daemon.
    if (act.type == "recall_memory") {
        std::string query = arg(act.args, "query");
        if (query.empty()) return "No query specified.";
        auto results = memory_->hybridSearch(query, 5);
        if (results.empty()) return "Nothing found in memory about that.";
        std::ostringstream out;
        for (auto& r : results) {
            out << "[" << r.entry.timestamp << " " << r.source;
            if (r.vecSim > 0) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), " sim=%.2f", r.vecSim);
                out << buf;
            }
            out << "] " << r.entry.role << ": " << r.entry.content << "\n";
        }
        return out.str();
    }
    if (act.type == "remember") {
        std::string k = arg(act.args, "key");
        std::string v = arg(act.args, "value");
        if (k.empty() || v.empty()) return "Nothing to remember.";
        memory_->setFact(k, v);
        Logger::info("Memory: LLM-stored fact " + k + " = " + v);
        return "Got it — remembered " + k + ".";
    }

    // ask_user — just speak the question; next utterance continues via LLM history.
    if (act.type == "ask") {
        std::string q = arg(act.args, "question");
        return q.empty() ? "What should I do?" : q;
    }

    return executor_->execute(act);
}

// ─── Phase 11: planner glue ───

void Daemon::unmuteAfterSettle(std::chrono::milliseconds settle) {
    if (!recorder_) return;
    if (settle.count() > 0)
        std::this_thread::sleep_for(settle);
    recorder_->unmute();
}

// Pop the next transcribed speech segment off the shared queue, with a
// bounded wait. Used by the confirmation gate. Returns the transcribed text
// (possibly empty on VAD noise) or "" on timeout.
std::string Daemon::pollNextText(std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string wavPath;
    {
        std::unique_lock<std::mutex> lock(qMutex_);
        if (!qCV_.wait_until(lock, deadline,
                [&]{ return !speechQueue_.empty() || shutdownRequested.load(); })) {
            return "";  // timed out
        }
        if (shutdownRequested.load() || speechQueue_.empty()) return "";
        wavPath = speechQueue_.front().wavPath;
        speechQueue_.pop();
    }
    if (!transcriber_) return "";
    return transcriber_->transcribe(wavPath);
}

// Regex for the affirmative side of the confirmation gate. Kept generous —
// the cost of a false negative is just "user has to repeat 'yes'", the cost
// of a false positive is a destructive step the user didn't authorize. So
// we only match words that are unambiguously affirmative.
static bool isAffirmative(const std::string& text) {
    static const std::regex yesRe(
        R"(\b(yes|yeah|yep|yup|sure|go ahead|proceed|confirm(?:ed)?|do it|approved?|ok(?:ay)?)\b)",
        std::regex::icase);
    return std::regex_search(text, yesRe);
}

void Daemon::runPlannerGoal(const std::string& goal, const LLMContext& ctx) {
    if (!planner_) {
        tts_->speak("Planner unavailable.");
        return;
    }

    // Confirmation callback — speak the (already humanized) prompt, listen
    // for one voice YES, then re-mute.
    auto onConfirm = [this](const std::string& prompt, const PlanStep& step) -> bool {
        (void)step;
        tts_->speak(prompt);
        recorder_->unmute();
        std::string reply = pollNextText(std::chrono::milliseconds(7000));
        recorder_->mute();
        bool ok = !reply.empty() && isAffirmative(reply);
        Logger::info(std::string("Planner confirm: reply='") + reply +
                     "' → " + (ok ? "YES" : "NO"));
        return ok;
    };

    // Progress callback — simple inline narration via TTS. Sentences are
    // short so an interrupt fired between them lands quickly.
    auto onProgress = [this](const std::string& message) {
        if (abortRequested_.load()) return;
        tts_->speak(message);
    };

    // Abort-check callback — polled by the planner between steps. Drains
    // any pending speech from the queue, transcribes it, and sets
    // abortRequested_ if the user said something interrupt-shaped. Also
    // honors the SIGUSR1 hard-pause path that flips abortRequested_ directly.
    auto shouldAbort = [this]() -> bool {
        if (abortRequested_.load()) return true;

        // Don't drain the queue while TTS is actively speaking — the mic
        // picks up ARIA's own voice via speaker loopback, and ARIA might
        // say "stop" or "cancel" inside her own narration ("I won't run
        // that…"), which would self-abort the plan.
        if (tts_->isSpeaking()) return false;

        // Try non-blocking pop. The planner is on this same thread so we
        // need to do this work synchronously between steps; the recorder is
        // unmuted so it has had a chance to enqueue something while the
        // last step ran.
        std::string wavPath;
        {
            std::unique_lock<std::mutex> lock(qMutex_);
            if (speechQueue_.empty()) return false;
            wavPath = speechQueue_.front().wavPath;
            speechQueue_.pop();
        }
        if (wavPath.empty() || !transcriber_) return false;

        std::string text = transcriber_->transcribe(wavPath);
        if (text.empty()) return false;

        Logger::info("Planner mid-execution heard: \"" + text + "\"");
        if (isInterruptCommand(text)) {
            tts_->interrupt();
            abortRequested_.store(true);
            return true;
        }
        // Non-interrupt utterance during plan — drop it. (Future: queue
        // post-plan tasks for the user.)
        return false;
    };

    // Unmute the recorder so the user can say "stop" during the plan.
    // processUtterance muted it before dispatching to the LLM; we hand
    // that responsibility to the planner from here.
    recorder_->unmute();
    plannerActive_.store(true);

    PlanResult result = planner_->run(goal, ctx, onConfirm, onProgress, shouldAbort);

    plannerActive_.store(false);
    recorder_->mute();  // restore the contract processUtterance expects

    if (!result.summary.empty() && !abortRequested_.load())
        tts_->speak(result.summary);

    memory_->save("assistant",
                  std::string("plan_result:") +
                  (result.success ? "ok" : (result.aborted ? "aborted" : "partial")) +
                  " (" + std::to_string(result.stepsExecuted) + " step" +
                  (result.stepsExecuted == 1 ? "" : "s") + ") " + result.summary);

    Logger::info("Planner finished: success=" + std::to_string(result.success) +
                 " aborted="  + std::to_string(result.aborted) +
                 " steps="    + std::to_string(result.stepsExecuted));
}

// Summarize older turns into a compact blurb so LLM context never explodes.
// Detaches to avoid stalling the user; runs every ~20 fresh turns.
void Daemon::maybeSummarize() {
    int lastId   = memory_->lastConversationId();
    int lastSumm = memory_->lastSummarizedId();
    if (lastId - lastSumm < 20) return;

    LLM*    llmPtr = llm_;
    Memory* memPtr = memory_;
    std::thread([memPtr, llmPtr, lastSumm, lastId]() {
        auto turns = memPtr->getRange(lastSumm, lastId);
        if (turns.size() < 10) return;
        std::ostringstream text;
        for (auto& t : turns)
            text << t.role << ": " << t.content << "\n";
        auto summary = llmPtr->summarize(text.str());
        if (!summary.empty()) {
            memPtr->saveSummary(summary, lastSumm, lastId);
            Logger::info("Memory: summarized turns " +
                         std::to_string(lastSumm + 1) + "-" + std::to_string(lastId));
        }
    }).detach();
}

// ─── Tool-call / ReAct handling ───
//
// v3 changes vs v2:
//   - No more brittle isMultiStep heuristic — the LLM decides via tool calls.
//     run_command still routes through ReAct so we can observe output.
//   - Wall-clock deadline (30s total) on top of step cap, so a stuck LLM
//     can't loop for 5×timeout seconds.
//   - If we hit the step cap or deadline without task_done, synthesize a
//     summary from the last observation instead of leaving the user silent.
void Daemon::handleLLMResponse(LLMResponse& response, LLMContext& ctx,
                                const std::string& origText) {
    (void)origText; // reserved for future use
    if (!response.hasAction()) {
        // Pure conversation — stream already delivered; just close it out.
        tts_->endStream();
        if (!response.speech.empty())
            memory_->save("assistant", response.speech);
        return;
    }

    // Tool calls arrived — end the text stream (may have been empty).
    tts_->endStream();

    // ReAct is only needed for actions that produce an observation worth
    // feeding back to the LLM. `run_command` is the canonical case; vision
    // tools likewise return data the LLM usually wants to summarize.
    bool needsReact = false;
    for (auto& a : response.actions) {
        if (a.type == "run" ||
            a.type == "see_screen" ||
            a.type == "describe_region" ||
            a.type == "read_screen_text") {
            needsReact = true;
            break;
        }
    }

    if (needsReact) {
        const int  maxSteps = Config::get().max_react_steps;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        int step = 0;
        std::string lastObservation;

        while (response.hasAction() && !response.done && step < maxSteps) {
            if (abortRequested_.load()) {
                Logger::info("ReAct: aborted by user");
                tts_->speak("Stopped.");
                break;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                Logger::warn("ReAct: 30s wall-clock timeout");
                break;
            }

            std::string observation;
            for (auto& act : response.actions) {
                if (act.type == "done") {
                    response.done = true;
                    std::string summary = arg(act.args, "summary");
                    Logger::info("ReAct: task complete — " + summary);
                    memory_->save("assistant", "task_done:" + summary);
                    tts_->speak(summary);
                    break;
                }

                Logger::info("ReAct step " + std::to_string(step + 1) +
                             ": " + act.type + " → " + act.args.dump());

                std::string result;
                if (act.type == "run")
                    result = executor_->safeShellCapture(arg(act.args, "command"));
                else
                    result = executeAction(act);
                memory_->save("assistant", act.type + ":" + act.args.dump());

                if (!observation.empty()) observation += "\n";
                observation += result;
            }

            if (response.done) break;

            lastObservation = observation;

            // NOTE: response.speech was already delivered to TTS via feedChunk
            // during the preceding {think,react}Streaming call, and flushed by
            // the endStream() above. Do NOT tts_->speak(response.speech) here —
            // that would replay the same line a second time.

            step++;
            if (step >= maxSteps || std::chrono::steady_clock::now() > deadline) {
                Logger::warn("ReAct: hit limit at step " + std::to_string(step) +
                             " (max=" + std::to_string(maxSteps) + ")");
                // Synthesize something from the last observation so the user
                // isn't left in silence when the LLM forgets to call task_done.
                std::string summary = lastObservation;
                if (summary.size() > 200) summary = summary.substr(0, 200) + "...";
                if (summary.empty())
                    tts_->speak("Done after " + std::to_string(step) + " steps.");
                else
                    tts_->speak(summary);
                break;
            }

            auto freshCtx = Context::capture();
            ctx.activeApp     = freshCtx.activeApp;
            ctx.activeWindow  = freshCtx.activeWindow;
            ctx.clipboard     = freshCtx.clipboard;
            ctx.screenText    = freshCtx.screenText;
            ctx.notifications = freshCtx.recentNotifications;

            tts_->startStream();
            response = llm_->reactStreaming(observation, ctx,
                [this](const std::string& delta) { tts_->feedChunk(delta); });
            tts_->endStream();
        }
    } else {
        // Simple action(s) — execute directly, no observation loop needed.
        // Exception: the "plan" pseudo-action hands control to the agentic
        // planner, which has its own execute/reflect/replan loop.
        for (auto& act : response.actions) {
            if (act.type == "plan") {
                std::string goal = arg(act.args, "goal");
                if (goal.empty()) goal = origText;
                Logger::info("LLM: delegating to planner — " + goal);
                memory_->save("assistant", "plan:" + act.args.dump());
                runPlannerGoal(goal, ctx);
                continue;
            }
            Logger::info("LLM action: " + act.type + " → " + act.args.dump());
            std::string feedback = executeAction(act);
            memory_->save("assistant", act.type + ":" + act.args.dump());
            if (response.speech.empty() && !feedback.empty())
                tts_->speak(feedback);
        }
    }

    // Persist the assistant speech alongside tool calls. Do NOT re-speak —
    // streaming already delivered it to TTS before handleLLMResponse ran.
    if (!response.speech.empty() && response.hasAction())
        memory_->save("assistant", response.speech);
}

// ─── Main utterance processor ───

void Daemon::processUtterance(const std::string& rawText) {
    std::string text = rawText;

    // Only treat the utterance as an interrupt if there's actually something
    // to interrupt — otherwise "stop" while idle gets eaten and never reaches
    // the LLM (annoying when the user really did want to chat).
    bool busy = (tts_ && tts_->isSpeaking()) || plannerActive_.load();
    if (busy && isInterruptCommand(text)) {
        tts_->interrupt();
        // Also flip the planner / ReAct abort flag so any in-flight task
        // (running on a worker thread) exits at its next check.
        abortRequested_.store(true);
        Logger::info("ARIA: interrupted by voice — '" + text + "'");
        turn_.path = "interrupt";
        return;
    }

    // Fresh utterance — clear any stale abort flag from the prior task.
    abortRequested_.store(false);

    // Wake-word gate — optional, controlled by ARIA_WAKE_WORD env var.
    if (wakeRequired_) {
        auto now = std::chrono::steady_clock::now();
        bool awake = (awakeUntil_ > now);
        auto stripped = stripWakeWord(text);
        if (stripped.has_value()) {
            text = *stripped;
            awakeUntil_ = now + kFollowUpWindow;
            if (text.empty()) {
                recorder_->mute();
                tts_->speak("Yes?");
                recorder_->unmute();
                Logger::info("Wake: acknowledged, awaiting follow-up.");
                turn_.path = "wake_ack";
                return;
            }
            Logger::info("Wake: activated → " + text);
        } else if (awake) {
            awakeUntil_ = now + kFollowUpWindow;
            Logger::info("Wake: follow-up accepted.");
        } else {
            Logger::info("Wake: asleep, ignored: " + text);
            turn_.path = "asleep";
            return;
        }
    }

    // Passive fact extraction — learn name, location, job, etc.
    auto newFacts = extractFacts(text);
    for (auto& [k, v] : newFacts) {
        memory_->setFact(k, v);
        Logger::info("Memory: learned " + k + " = " + v);
        lastFactKey_ = k;
    }

    // Passive entity extraction — URLs, paths, handles, tags. Silent, best-effort.
    for (auto& [name, type] : extractEntities(text)) {
        memory_->upsertEntity(name, type);
    }

    // Correction path: "actually X" / "I meant X" rewrites the most recent fact.
    if (newFacts.empty() && !lastFactKey_.empty()) {
        auto now = std::chrono::steady_clock::now();
        auto ageSec = std::chrono::duration_cast<std::chrono::seconds>(
                          now - lastFactTime_).count();
        if (ageSec <= 120) {
            std::regex corrRe(
                R"(^\s*(?:-\s*)?(?:actually|sorry|no[, ]|i meant|i mean|it'?s|it is)\s+([A-Za-z0-9][A-Za-z0-9 .\-+_]{0,60}?)\s*[.!?]?\s*$)",
                std::regex::icase);
            std::smatch m;
            if (std::regex_search(text, m, corrRe)) {
                std::string v = m[1].str();
                while (!v.empty() && (v.back()  == ' ' || v.back()  == '\t')) v.pop_back();
                while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
                if (!v.empty() && v.size() < 80) {
                    memory_->setFact(lastFactKey_, v);
                    Logger::info("Memory: corrected " + lastFactKey_ + " → " + v);
                }
            }
        }
    }
    if (!newFacts.empty())
        lastFactTime_ = std::chrono::steady_clock::now();

    // Intent classifier runs before the word-count filter so short known
    // commands ("play", "mute") still dispatch.
    AgentAction directAction = classifyIntent(text);

    if (directAction.empty() && wordCount(text) <= 2) {
        Logger::info("Skipped short utterance: " + text);
        turn_.path = "skip";
        return;
    }

    recorder_->mute();

    if (!directAction.empty()) {
        Logger::info("Intent: " + directAction.type + " → " + directAction.args.dump());
        turn_.path    = "intent";
        turn_.outcome = directAction.type;
        std::string feedback = executeAction(directAction);
        memory_->save("user", text);
        memory_->save("assistant", directAction.type + ":" + directAction.args.dump());
        if (!feedback.empty()) tts_->speak(feedback);
        unmuteAfterSettle();
        maybeSummarize();
        return;
    }

    // Unknown command → LLM path.
    auto sysCtx = Context::capture();
    turn_.ctxDone = nowMs();
    Logger::info("Context: app=" + sysCtx.activeApp + " window=" + sysCtx.activeWindow);

    // Fast-path direct context queries — avoid the LLM round-trip.
    auto lt = lower(text);
    std::string directAnswer;
    if ((lt.find("what app") != std::string::npos ||
         lt.find("which app") != std::string::npos) &&
        !sysCtx.activeApp.empty()) {
        directAnswer = "You're using " + sysCtx.activeApp + ".";
    } else if ((lt.find("what's in my clipboard") != std::string::npos ||
                lt.find("what did i copy") != std::string::npos ||
                lt.find("read clipboard") != std::string::npos) &&
               !sysCtx.clipboard.empty()) {
        directAnswer = "Your clipboard says: " + sysCtx.clipboard;
    } else if (lt.find("what window") != std::string::npos &&
               !sysCtx.activeWindow.empty()) {
        directAnswer = "The window title is " + sysCtx.activeWindow + ".";
    }
    if (!directAnswer.empty()) {
        Logger::info("Context query → " + directAnswer);
        turn_.path    = "context";
        turn_.outcome = "speech";
        memory_->save("user", text);
        memory_->save("assistant", directAnswer);
        tts_->speak(directAnswer);
        unmuteAfterSettle();
        maybeSummarize();
        return;
    }

    // Conversational fast-path — block the LLM from hallucinating tool calls
    // on phrases like "you're ready to go" (was calling system_info).
    if (const char* reply = matchSmallTalk(text)) {
        Logger::info("Small-talk fast-path: \"" + text + "\" → " + reply);
        turn_.path    = "smalltalk";
        turn_.outcome = "speech";
        memory_->save("user", text);
        memory_->save("assistant", reply);
        tts_->speak(reply);
        unmuteAfterSettle();
        return;
    }

    // Check Ollama health before wasting effort building context.
    if (!llm_->isAvailable()) {
        Logger::warn("LLM: Ollama unreachable, retrying once...");
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!llm_->isAvailable()) {
            Logger::error("LLM: still unreachable, giving up on this utterance");
            memory_->save("user", text);
            tts_->speak("I'm offline right now, try again in a moment.");
            turn_.path = "llm_offline";
            unmuteAfterSettle();
            return;
        }
    }

    LLMContext ctx;
    ctx.activeApp     = sysCtx.activeApp;
    ctx.activeWindow  = sysCtx.activeWindow;
    ctx.clipboard     = sysCtx.clipboard;
    ctx.screenText    = sysCtx.screenText;
    ctx.notifications = sysCtx.recentNotifications;
    ctx.memorySummary = memory_->getSummary();
    ctx.tone          = detectTone(text);
    ctx.timeOfDay     = currentTimeOfDay();
    ctx.dateLabel     = currentDateLabel();
    if (!ctx.tone.empty())
        Logger::info("Tone: " + ctx.tone);

    auto userName = memory_->getFact("user_name");
    std::string prompt = text;
    if (!userName.empty() &&
        (lower(text).find("hello") != std::string::npos ||
         lower(text).find("hey")   != std::string::npos))
        prompt = text + " (user's name is " + userName + ")";

    // Streaming LLM → Streaming TTS
    turn_.path     = "llm";
    turn_.llmStart = nowMs();
    tts_->startStream();
    auto response = llm_->thinkStreaming(prompt, ctx,
        [this](const std::string& delta) {
            if (!turn_.firstToken) turn_.firstToken = nowMs();
            tts_->feedChunk(delta);
        });
    turn_.llmDone = nowMs();
    if (response.hasAction()) {
        std::string types;
        for (auto& a : response.actions)
            types += (types.empty() ? "" : ",") + a.type;
        turn_.outcome = "tool:" + types;
    } else {
        turn_.outcome = "speech";
    }

    memory_->save("user", text);

    handleLLMResponse(response, ctx, text);

    // Settle window: TTS just finished; speaker→mic echo can persist for
    // a few hundred ms (room reverb + playback buffer drain). Keep the mic
    // muted through it so we don't immediately re-trigger on our own voice.
    unmuteAfterSettle();
    maybeSummarize();
}

// ─── Run loop ───

void Daemon::run() {
    Logger::info("Initializing pipeline...");
    const auto& cfg = Config::get();

    // Phase 10: sweep any orphan /tmp/aria_vlm_*.png left behind by a prior
    // crashed run before we start generating new ones this session.
    Vision::cleanupStaleTempFiles();

    Transcriber transcriber(cfg.whisper_model);
    Memory      memory(cfg.db_path);
    LLM         llm(cfg.ollama_model, &memory);
    Executor    executor;
    TTS         tts(cfg.piper_model);
    Planner     planner(&llm, &executor, &memory);

    // Publish member pointers for the extracted methods.
    transcriber_ = &transcriber;
    memory_      = &memory;
    llm_         = &llm;
    executor_    = &executor;
    tts_         = &tts;
    planner_     = &planner;

    // Startup Ollama probe — warn but don't bail; intent-only commands
    // still work without the LLM.
    if (!llm.isAvailable())
        Logger::warn("LLM: Ollama unreachable at " + cfg.ollama_url +
                     " — LLM queries will fail until it's up.");

    Recorder recorder("/tmp/aria_speech.wav",
        [this](const std::string& wavPath) {
            std::lock_guard<std::mutex> lock(qMutex_);
            speechQueue_.push({wavPath, nowMs()});
            qCV_.notify_one();
        });
    recorder_ = &recorder;

    // Phase 0: corpus capture opt-out.
    if (const char* v = std::getenv("ARIA_BENCH_CAPTURE"))
        benchCapture_ = !(std::string(v) == "0" || std::string(v) == "false");

    TimerManager timers([&tts, &recorder](const std::string& label) {
        recorder.mute();
        tts.speak("Timer done: " + label);
        system(("notify-send 'ARIA Timer' " + std::string("'") + label + "'").c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        recorder.unmute();
    });
    timers_ = &timers;

    // ── Phase 9: audio wake word ──
    // Default: audio wake word enabled. Falls back to ALWAYS_ON recorder if
    // the model files are missing or ORT init fails.
    std::shared_ptr<WakeWord> ww;
    if (cfg.wake_enabled && !cfg.always_listen) {
        ww = std::make_shared<WakeWord>();
        WakeWord::Config wcfg;
        wcfg.melspecModelPath   = cfg.wake_model_dir + "/melspectrogram.onnx";
        wcfg.embeddingModelPath = cfg.wake_model_dir + "/embedding_model.onnx";
        wcfg.wakewordModelPath  = cfg.wake_model_dir + "/" + cfg.wake_model;
        wcfg.threshold          = cfg.wake_threshold;
        wcfg.triggerLevel       = cfg.wake_trigger;
        wcfg.refractoryFrames   = cfg.wake_refractory;
        wcfg.debug              = cfg.wake_debug;
        if (!ww->init(wcfg)) {
            Logger::warn("Wake-word: init failed — falling back to always-on mode.");
            ww.reset();
        }
    }

    // Callback: play the wake cue. Spawned as a detached thread so the
    // recorder's capture loop keeps draining the audio pipe — otherwise TTS
    // playback queues in the PulseAudio buffer and gets re-fed into the wake
    // detector after unmute.
    auto wakeCue = [&tts, &recorder, &cfg]() {
        if (cfg.wake_cue == "none") return;
        std::thread([&tts, &recorder, &cfg]() {
            recorder.mute();
            if (cfg.wake_cue == "tone") {
                system("paplay --volume=30000 /usr/share/sounds/freedesktop/stereo/message.oga >/dev/null 2>&1 &");
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            } else {
                tts.speak("Yes?");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            recorder.unmute();
        }).detach();
    };

    if (ww) {
        recorder.setWakeWord(Recorder::Mode::WAKE_WORD, ww,
                             std::chrono::seconds(cfg.wake_window_sec),
                             wakeCue);
        Logger::info("Wake-word: audio gate active (model=" + cfg.wake_model +
                     ", threshold=" + std::to_string(cfg.wake_threshold) + ").");
    } else {
        Logger::info("Wake-word: disabled (always-on listening).");
    }

    // Text-prefix wake (legacy) stays useful when audio wake is off and the
    // user still wants a command gate. With audio wake active, the recorder
    // already gates so text check is a no-op.
    wakeRequired_ = false;
    if (const char* v = std::getenv("ARIA_WAKE_TEXT"); v && v[0] == '1')
        wakeRequired_ = true;

    std::thread processor([&]() {
        while (!shutdownRequested.load()) {
            std::unique_lock<std::mutex> lock(qMutex_);
            qCV_.wait_for(lock, std::chrono::milliseconds(100),
                [&]{ return !speechQueue_.empty() || shutdownRequested.load(); });
            if (speechQueue_.empty()) continue;
            QueuedWav qw = speechQueue_.front();
            speechQueue_.pop();
            lock.unlock();
            const std::string& wavPath = qw.wavPath;

            if (!ariaActive_.load()) continue;

            turn_.reset();
            turn_.speechEnd = qw.enqueuedMs;

            // Echo guard: if TTS is still speaking, or finished <1500ms ago,
            // this WAV is almost certainly speaker→mic feedback of ARIA's
            // own voice. Drop it without burning transcribe latency. The
            // 1.5s window covers the typical capture→queue→pop lag plus
            // post-playback room reverb; cost is dropping legitimate user
            // speech in the first 1.5s after ARIA finishes, which is rare.
            if (tts.wasRecentlySpeaking(1500)) {
                Logger::info("VAD: dropping likely-echo segment (TTS just spoke "
                             + std::to_string(tts.msSinceLastSpeech()) + " ms ago).");
                // Traced: these are exactly where fast user replies get
                // eaten today. Phase 3 (AEC) must drive this count to zero.
                turn_.path = "echo_drop";
                logTurn("");
                continue;
            }

            std::string text = transcriber.transcribe(wavPath);
            turn_.asrDone = nowMs();

            // After 3 consecutive whisper failures, tell the user something's
            // wrong — before that we stay silent to avoid spamming on misreads.
            if (text.empty() && transcriber.consecutiveFailures() == 3) {
                Logger::error("Transcriber: 3 consecutive failures, announcing.");
                recorder.mute();
                tts.speak("Having trouble hearing. Check your microphone.");
                recorder.unmute();
                transcriber.resetFailures();
                continue;
            }

            if (text.empty() || text.find("[BLANK_AUDIO]") != std::string::npos) {
                Logger::info("VAD: blank, skipping.");
                continue;
            }
            Logger::info("You said: " + text);
            maybeArchiveUtterance(wavPath, text);

            processUtterance(text);
            logTurn(text);
        }
    });

    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    // Startup greeting with name if known.
    recorder.mute();
    auto userName = memory.getFact("user_name");
    tts.speak(userName.empty() ? "Online." : "Online, " + userName + ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    recorder.unmute();

    Logger::info("ARIA ready.");

    while (!shutdownRequested.load()) {
        if (pauseToggle.exchange(false)) {
            bool nowActive = !ariaActive_.load();
            ariaActive_.store(nowActive);

            // Hard-stop the world: kill any ongoing TTS, signal in-flight
            // planner / ReAct work to bail, drop pending speech segments
            // so resumed listening doesn't replay a stale queue.
            tts.interrupt();
            abortRequested_.store(true);
            {
                std::lock_guard<std::mutex> lock(qMutex_);
                std::queue<QueuedWav> empty;
                speechQueue_.swap(empty);
            }
            qCV_.notify_all();

            recorder.mute();
            tts.speak(nowActive ? "Back." : "Pausing.");
            if (nowActive) recorder.unmute();
            Logger::info(nowActive ? "ARIA: resumed."
                                    : "ARIA: paused (hard-stop, queue cleared).");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    recorder.stop();
    processor.join();
    Logger::info("Daemon shutting down cleanly.");
}
