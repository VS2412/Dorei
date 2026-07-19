#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>

#include "llm.hpp"  // for LLMResponse, LLMContext, AgentAction

class Transcriber;
class Executor;
class TTS;
class Memory;
class TimerManager;
class Recorder;
class Planner;

class Daemon {
public:
    explicit Daemon(std::atomic<bool>& shutdownFlag,
                    std::atomic<bool>& pauseToggle);
    void run();

private:
    // Extracted from the old monolithic run() lambda for readability.
    void        processUtterance(const std::string& rawText);
    void        handleLLMResponse(LLMResponse& response, LLMContext& ctx,
                                  const std::string& origText);
    std::string executeAction(const AgentAction& act);
    void        maybeSummarize();

    // Phase 11: run the agentic planner for `goal`. Handles the confirmation
    // gate (voice-prompted YES/NO) and progress speech on the daemon thread.
    void        runPlannerGoal(const std::string& goal, const LLMContext& ctx);

    // Pop the next transcribed utterance off the speech queue with timeout.
    // Used by the planner's confirmation gate to read a voice YES/NO without
    // spinning up a new recording path. Returns empty string on timeout.
    std::string pollNextText(std::chrono::milliseconds timeout);

    // Echo-guard: pause briefly so the speaker→mic reverb of ARIA's own
    // voice doesn't immediately retrigger VAD when we reopen the mic.
    // Called everywhere we follow a TTS call with `recorder_->unmute()`.
    void        unmuteAfterSettle(std::chrono::milliseconds settle =
                                      std::chrono::milliseconds(500));

    std::atomic<bool>& shutdownRequested;
    std::atomic<bool>& pauseToggle;

    // Subsystems — stack-allocated in run(); member pointers set so the
    // extracted methods can reach them without captures. Valid only while
    // run() is on the stack.
    Transcriber*  transcriber_ = nullptr;
    LLM*          llm_         = nullptr;
    Executor*     executor_    = nullptr;
    TTS*          tts_         = nullptr;
    Memory*       memory_      = nullptr;
    TimerManager* timers_      = nullptr;
    Recorder*     recorder_    = nullptr;
    Planner*      planner_     = nullptr;

    // Speech queue — filled by the Recorder callback, drained by the
    // processor thread. Hoisted to members (from locals in run()) so the
    // planner's confirmation gate can pollNextText on the same queue.
    // Each entry carries its enqueue time: that instant ≈ VAD speech-end,
    // which is t=0 of the turn latency budget.
    struct QueuedWav {
        std::string wavPath;
        long long   enqueuedMs = 0;  // steady-clock ms (Daemon::nowMs epoch)
    };
    std::mutex              qMutex_;
    std::queue<QueuedWav>   speechQueue_;
    std::condition_variable qCV_;

    // ── Phase 0 instrumentation ──
    // Per-utterance stage timestamps, all steady-clock ms (nowMs epoch).
    // Filled across the processor thread + processUtterance, emitted as ONE
    // machine-parseable `TURN {json}` log line per turn so humans and bench
    // tooling read the whole latency budget from a single place.
    struct TurnTrace {
        long long speechEnd  = 0;  // WAV enqueued by recorder (≈ VAD end)
        long long asrDone    = 0;  // transcript available
        long long ctxDone    = 0;  // Context::capture finished (LLM path)
        long long llmStart   = 0;  // request sent to Ollama
        long long firstToken = 0;  // first streamed content delta
        long long llmDone    = 0;  // thinkStreaming returned
        std::string path;          // intent|context|smalltalk|llm|echo_drop|…
        std::string outcome;       // "tool:<types>" | "speech" | note
        void reset() { *this = TurnTrace{}; }
    };
    TurnTrace turn_;
    static long long nowMs();
    void logTurn(const std::string& text);

    // Archive real utterance WAVs + transcripts to ~/.local/share/aria/
    // bench_corpus/ so `aria-bench asr` can eval against the user's actual
    // voice (including mishearings). Keep-last-200; ARIA_BENCH_CAPTURE=0
    // disables.
    void maybeArchiveUtterance(const std::string& wavPath,
                               const std::string& text);
    bool benchCapture_ = true;

    // Processing state
    std::atomic<bool> ariaActive_{true};

    // Mid-task abort. Set by:
    //   - SIGUSR1 / Super+Space (also flips ariaActive_)
    //   - voice "stop"/"cancel"/"wait"/"shut up" arriving on the queue while
    //     a planner / ReAct loop is running
    // Consumers (handleLLMResponse ReAct loop, runPlannerGoal abort callback)
    // must check it cooperatively. Reset at the start of every utterance.
    std::atomic<bool> abortRequested_{false};

    // True while a planner is executing on this thread. Used by the SIGUSR1
    // path to know whether interrupting TTS alone is enough or whether we
    // need to flip abortRequested_ as well.
    std::atomic<bool> plannerActive_{false};
    std::string       lastFactKey_;
    std::chrono::steady_clock::time_point lastFactTime_{};
    bool              wakeRequired_ = false;
    std::chrono::steady_clock::time_point awakeUntil_{};

    static constexpr auto kFollowUpWindow = std::chrono::seconds(5);
};
