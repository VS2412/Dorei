#pragma once
#include <string>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <sys/types.h>

class TTS {
public:
    explicit TTS(const std::string& modelPath);
    ~TTS();

    // Batch API (still works, uses streaming internally)
    void speak(const std::string& text);

    // Streaming API — feed text chunks as they arrive from LLM
    void startStream();
    void feedChunk(const std::string& text);
    void endStream();

    void interrupt();
    bool isSpeaking() const { return speaking_.load(); }
    bool available() const  { return available_; }

    // Echo-guard: returns ms since last sentence finished playing (or 0 if
    // still speaking). Used by the daemon's processor loop to drop captures
    // that are almost certainly speaker→mic feedback of ARIA's own voice.
    long long msSinceLastSpeech() const;
    bool wasRecentlySpeaking(long long withinMs) const;

    // Phase 0 instrumentation: steady-clock ms of the most recent sentence
    // playback start (piper spawn). The daemon's TurnTrace uses this as the
    // "first audio" proxy for the turn latency budget.
    long long lastSpeakStartMs() const { return lastSpeakStartMs_.load(); }

private:
    std::string modelPath_;
    bool available_{false};
    std::atomic<bool> speaking_{false};
    std::atomic<pid_t> activePid_{-1};
    std::atomic<long long> lastEndedMs_{0};
    std::atomic<long long> lastSpeakStartMs_{0};

    // Streaming internals
    std::string streamBuffer_;
    std::mutex queueMutex_;
    std::queue<std::string> sentenceQueue_;
    std::condition_variable queueCV_;
    std::thread playbackThread_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> stopPlayback_{false};

    void runAndTrack(const std::string& cmd);
    void speakSentence(const std::string& text);
    void playbackLoop();
    void flushBuffer();
};
