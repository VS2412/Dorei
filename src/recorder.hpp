#pragma once
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <memory>

class WakeWord;

class Recorder {
public:
    enum class Mode {
        ALWAYS_ON,   // every VAD segment transcribed (legacy behavior)
        WAKE_WORD,   // wake-word gates VAD emission; fallback to ALWAYS_ON
                     // if init fails.
    };

    using SpeechCallback = std::function<void(const std::string&)>;
    using WakeCallback   = std::function<void()>;

    explicit Recorder(const std::string& outputPath, SpeechCallback onSpeech);
    ~Recorder();

    void start();
    void stop();
    void mute();
    void unmute();
    bool isActive() const { return active_.load(); }

    // Configure wake-word mode. Must be called before start(). If `ww` is null
    // or mode is ALWAYS_ON, behaves exactly like pre-Phase-9 recorder.
    // `onWake` (optional) is invoked on the capture thread when a wake fires —
    // use it to play a cue.
    void setWakeWord(Mode mode,
                     std::shared_ptr<WakeWord> ww,
                     std::chrono::seconds awakeWindow,
                     WakeCallback onWake = {});

    // Force the recorder into the "awake" state for `awakeWindow` — useful when
    // code elsewhere (e.g. an interrupt gesture) knows the user is addressing
    // ARIA. No-op in ALWAYS_ON mode.
    void armAwake();

    bool isAwake() const;

private:
    void captureLoop();
    void writeWav(const std::vector<int16_t>& samples);

    std::string       outputPath_;
    SpeechCallback    onSpeech_;
    WakeCallback      onWake_;
    std::atomic<bool> active_{false};
    std::atomic<bool> muted_{false};
    std::thread       thread_;

    // Wake-word mode state
    Mode                          mode_{Mode::ALWAYS_ON};
    std::shared_ptr<WakeWord>     wakeWord_;
    std::chrono::seconds          awakeWindow_{10};
    std::atomic<long long>        awakeUntilMs_{0};  // steady_clock ms

    static constexpr int RATE         = 16000;
    static constexpr int FRAME_MS     = 20;
    static constexpr int FRAME_SAMPLES = RATE * FRAME_MS / 1000; // 320
    static constexpr int ONSET_FRAMES  = 10;   // 200ms of voice to start
    // 15×20ms = 300ms of trailing silence ends the utterance. Was 500ms —
    // this is pure dead air added to every single turn; 300ms still clears
    // natural mid-sentence pauses (typical inter-word gaps are <200ms).
    static constexpr int TRAIL_FRAMES  = 15;
    static constexpr int MIN_FRAMES    = 10;  // 200ms minimum utterance
    static constexpr int MAX_FRAMES    = 1000; // 20s hard cap
    static constexpr int PREROLL       = 5;   // frames before onset

    // Wake-word chunk = 80ms = 4 × 20ms recorder frames = 1280 samples.
    static constexpr int WAKE_CHUNK_SAMPLES = 1280;
};
