#include "context.hpp"
#include "screen.hpp"
#include "logger.hpp"
#include <cstdio>
#include <memory>
#include <string>
#include <future>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static std::string exec(const char* cmd) {
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) return "";
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe.get()))
        result += buf;
    if (!result.empty() && result.back() == '\n')
        result.pop_back();
    return result;
}

// Terminal-like apps where OCR is both expensive (dense text) and
// low-signal (command history + clipboard already carry the relevant text).
// Skip OCR for these, fall through for GUI apps where OCR is most useful.
static bool isTerminalApp(const std::string& app) {
    static const char* kTerms[] = {
        "org.gnome.Console", "alacritty", "Alacritty", "kitty",
        "foot", "footclient", "wezterm", "terminator", "xterm",
        "urxvt", "tilix", "konsole", "com.gexperts.Tilix"
    };
    for (auto* t : kTerms)
        if (app == t) return true;
    return false;
}

// Run niri + clipboard concurrently, then OCR keyed by whatever niri returned.
// OCR is the heaviest call, so starting it after niri lets us (a) use the
// correct cache key and (b) skip it entirely for terminals where the text is
// already available via the clipboard path.
SystemContext Context::capture() {
    SystemContext ctx;

    auto t0 = std::chrono::steady_clock::now();

    auto niriF = std::async(std::launch::async, []() -> std::pair<std::string, std::string> {
        try {
            std::string raw = exec("niri msg --json focused-window 2>/dev/null");
            if (raw.empty()) return {"", ""};
            auto j = json::parse(raw);
            return { j.value("app_id", ""), j.value("title", "") };
        } catch (...) { return {"", ""}; }
    });

    auto clipF = std::async(std::launch::async, []() {
        return exec("wl-paste --no-newline 2>/dev/null | head -c 120");
    });

    // Pull recent notifications from whichever notification daemon responds.
    // dunstctl and makoctl are mutually exclusive in practice — the first
    // non-empty result wins. Budget ~300 chars so LLM context stays tight.
    auto notifF = std::async(std::launch::async, []() -> std::string {
        std::string out = exec("dunstctl history 2>/dev/null "
                               "| head -c 300");
        if (!out.empty()) return out;
        out = exec("makoctl list 2>/dev/null | head -c 300");
        return out;
    });

    auto [app, win] = niriF.get();
    ctx.activeApp    = app;
    // For terminal apps the window title is almost always the running
    // shell command ("cd ~/foo; systemctl restart ai-agent;...") which is
    // pure noise to the LLM — and worse, the LLM tends to read it as if
    // it were user intent. The activeApp ("alacritty"/"org.gnome.Console")
    // already conveys "user is at a terminal", so drop the title in that
    // case rather than injecting shell history into every prompt.
    ctx.activeWindow = isTerminalApp(app) ? std::string{} : win;

    std::future<std::string> screenF;
    bool ocrSkipped = false;
    if (isTerminalApp(app)) {
        ocrSkipped = true;
    } else {
        std::string key = app + "|" + win;
        screenF = std::async(std::launch::async, [key]() {
            return Screen::capture(key);
        });
    }

    ctx.clipboard           = clipF.get();
    ctx.recentNotifications = notifF.get();
    ctx.screenText          = ocrSkipped ? std::string{} : screenF.get();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    Logger::info("Context: captured in " + std::to_string(ms) + "ms" +
                 (ocrSkipped ? " (OCR skipped for terminal)" : ""));
    return ctx;
}
