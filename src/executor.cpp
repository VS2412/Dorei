#include "executor.hpp"
#include "logger.hpp"
#include "config.hpp"
#include "screen.hpp"
#include "vision.hpp"
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static std::string q(const std::string& s) {
    std::string o = "'";
    for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; }
    return o + "'";
}

// Extract a string arg from JSON, with "param" as generic fallback key
static std::string arg(const nlohmann::json& a, const std::string& key) {
    if (a.contains(key) && a[key].is_string()) return a[key].get<std::string>();
    if (a.contains("param") && a["param"].is_string()) return a["param"].get<std::string>();
    return "";
}

static int argInt(const nlohmann::json& a, const std::string& key, int def) {
    if (a.contains(key) && a[key].is_number()) return a[key].get<int>();
    return def;
}

static bool argBool(const nlohmann::json& a, const std::string& key, bool def) {
    if (a.contains(key) && a[key].is_boolean()) return a[key].get<bool>();
    return def;
}

// Expand ~ to $HOME
static std::string expandHome(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

static const std::vector<std::pair<std::string,std::string>> kAppMap = {
    {"firefox",         "firefox"},
    {"browser",         "firefox"},
    {"chrome",          "chromium"},
    {"chromium",        "chromium"},
    {"terminal",        "alacritty"},
    {"alacritty",       "alacritty"},
    {"kitty",           "kitty"},
    {"code",            "code"},
    {"vscode",          "code"},
    {"vs code",         "code"},
    {"visual studio",   "code"},
    {"spotify",         "spotify"},
    {"discord",         "discord"},
    {"telegram",        "telegram-desktop"},
    {"files",           "nautilus"},
    {"file manager",    "nautilus"},
    {"nautilus",        "nautilus"},
    {"calculator",      "gnome-calculator"},
    {"settings",        "gnome-control-center"},
    {"steam",           "steam"},
    {"obs",             "obs"},
    {"gimp",            "gimp"},
    {"vlc",             "vlc"},
    {"thunar",          "thunar"},
    {"dolphin",         "dolphin"},
    {"blender",         "blender"},
    {"inkscape",        "inkscape"},
};

static std::string resolveApp(const std::string& raw) {
    std::string t = raw;
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    for (auto& [k, v] : kAppMap)
        if (t.find(k) != std::string::npos) return v;
    return raw; // pass through verbatim
}

// Phase 7: Safety guard for LLM-driven commands
enum class Safety { Allow, Caution, Deny };

static const std::vector<std::string> kDenyPatterns = {
    "rm -rf /", "rm -rf /*", "rm -rf --no-preserve-root", "rm -rf ~",
    "rm -rf $home", "rm -rf ${home}",
    "mkfs", "dd if=", "dd of=/dev/sd", "dd of=/dev/nvme", "dd of=/dev/disk",
    ":(){ :|:& };:", ":(){:|:&};:",
    "> /dev/sda", "> /dev/nvme", "> /dev/disk",
    "chmod -r 777 /", "chmod 777 /",
    "shutdown", "reboot", "halt", "poweroff", "init 0", "init 6",
    "systemctl poweroff", "systemctl reboot", "systemctl halt",
    " | sh", " | bash", " |sh", " |bash",
    "userdel ", "passwd root",
    "> /etc/passwd", "> /etc/shadow",
    "fdisk /dev/", "parted /dev/", "wipefs",
};

static const std::vector<std::string> kCautionPatterns = {
    "sudo ", "rm -r ", "rm -rf ", "rm -f ",
    "kill -9", "killall", "pkill -9",
    "chmod ", "chown ", "mount ", "umount ",
    "iptables", "nft ",
};

static Safety checkSafety(const std::string& cmd) {
    std::string lower = cmd;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& p : kDenyPatterns)
        if (lower.find(p) != std::string::npos) return Safety::Deny;
    for (const auto& p : kCautionPatterns)
        if (lower.find(p) != std::string::npos) return Safety::Caution;
    return Safety::Allow;
}

static const std::vector<std::string> kCriticalProcs = {
    "systemd", "init", "kthreadd", "kernel",
    "pipewire", "wireplumber", "wayland", "niri", "ai-agent",
    "dbus", "sshd", "Xwayland",
};

static bool isCriticalProc(const std::string& target) {
    std::string t = target;
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    for (const auto& p : kCriticalProcs)
        if (t.find(p) != std::string::npos) return true;
    return false;
}

void Executor::shell(const std::string& cmd) {
    Logger::info("Executor: " + cmd);
    system((cmd + " &").c_str());
}

std::string Executor::safeShellCapture(const std::string& cmd) {
    Safety s = checkSafety(cmd);
    if (s == Safety::Deny) {
        Logger::warn("Executor: DENIED destructive command (ReAct): " + cmd);
        return "I won't run that. The command looks destructive.";
    }
    if (s == Safety::Caution)
        Logger::warn("Executor: CAUTION elevated command (ReAct): " + cmd);
    return shellCapture(cmd);
}

std::string Executor::shellCapture(const std::string& cmd) {
    Logger::info("Executor (capture): " + cmd);
    std::string result;
    FILE* f = popen((cmd + " 2>&1").c_str(), "r");
    if (!f) return "Command failed to execute.";
    char buf[512];
    while (fgets(buf, sizeof(buf), f))
        result += buf;
    int status = pclose(f);
    if (result.empty())
        result = (status == 0) ? "Command completed successfully." : "Command failed.";
    // Truncate long output for LLM context
    if (result.size() > 2000)
        result = result.substr(0, 2000) + "... (truncated)";
    return result;
}

std::string Executor::execute(const AgentAction& action) {
    // ─── Original actions ───
    if (action.type == "open") {
        std::string binary = resolveApp(arg(action.args, "app"));
        if (binary.find(' ') != std::string::npos || binary.empty()) {
            Logger::warn("Executor: rejected unknown app: " + binary);
            return "I don't know how to open that.";
        }
        shell(binary);
        return "On it.";
    }
    if (action.type == "run") {
        std::string cmd = arg(action.args, "command");
        if (cmd.empty()) return "";
        Safety s = checkSafety(cmd);
        if (s == Safety::Deny) {
            Logger::warn("Executor: DENIED destructive command: " + cmd);
            return "I won't run that. The command looks destructive.";
        }
        if (s == Safety::Caution)
            Logger::warn("Executor: CAUTION elevated command: " + cmd);
        shell(cmd);
        return "Running that now.";
    }
    if (action.type == "type") {
        std::string text = arg(action.args, "text");
        system(("wtype " + q(text)).c_str());
        return "";
    }
    if (action.type == "workspace") {
        std::string num = arg(action.args, "number");
        system(("niri msg action focus-workspace " + q(num)).c_str());
        return "Switching.";
    }
    if (action.type == "close") {
        system("niri msg action close-window");
        return "Closed.";
    }
    if (action.type == "volume") {
        std::string p = arg(action.args, "action");
        int step = Config::get().volume_step;
        if (p == "up") {
            system(("wpctl set-volume @DEFAULT_AUDIO_SINK@ " + std::to_string(step) + "%+").c_str());
            return "Louder.";
        }
        else if (p == "down") {
            system(("wpctl set-volume @DEFAULT_AUDIO_SINK@ " + std::to_string(step) + "%-").c_str());
            return "Quieter.";
        }
        else if (p == "mute") {
            system("wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle");
            return "Muted.";
        }
    }
    if (action.type == "brightness") {
        std::string p = arg(action.args, "action");
        int step = Config::get().brightness_step;
        if (p == "up") {
            system(("brightnessctl set " + std::to_string(step) + "%+").c_str());
            return "Brighter.";
        }
        else {
            system(("brightnessctl set " + std::to_string(step) + "%-").c_str());
            return "Dimmer.";
        }
    }
    if (action.type == "media") {
        std::string p = arg(action.args, "action");
        if      (p == "play" || p == "pause") system("playerctl play-pause");
        else if (p == "next")  system("playerctl next");
        else if (p == "prev")  system("playerctl previous");
        else if (p == "stop")  system("playerctl stop");
        return "";
    }
    if (action.type == "play_music") {
        // Start music from cold. Defaults to YouTube Music — works without
        // a Spotify install, opens in the user's default browser (firefox).
        // If the user gives a query, URL-encode it via jq -sRr @uri.
        std::string query = arg(action.args, "query");
        std::string url;
        if (query.empty()) {
            url = "https://music.youtube.com/";
        } else {
            // Conservative URL encoding via printf "%s" piped through jq.
            // Falls back to raw query if jq isn't installed (most arch
            // installs have it via nlohmann-json dep chain, but be safe).
            std::string enc;
            FILE* f = popen(("printf '%s' " + q(query) +
                             " | jq -sRr @uri 2>/dev/null").c_str(), "r");
            if (f) {
                char buf[512] = {};
                fread(buf, 1, sizeof(buf) - 1, f);
                pclose(f);
                enc = buf;
                while (!enc.empty() && (enc.back() == '\n' || enc.back() == '\r'))
                    enc.pop_back();
            }
            if (enc.empty()) enc = query;  // raw fallback
            url = "https://music.youtube.com/search?q=" + enc;
        }
        shell("xdg-open " + q(url));
        return query.empty() ? "Playing music." : "Playing " + query + ".";
    }
    if (action.type == "screenshot") {
        system("grim ~/Pictures/screenshot-$(date +%Y%m%d-%H%M%S).png 2>/dev/null");
        return "Screenshot saved.";
    }
    if (action.type == "clipboard") {
        std::string p = arg(action.args, "action");
        if (p == "read") {
            FILE* f = popen("wl-paste --no-newline 2>/dev/null | head -c 200", "r");
            if (!f) return "Clipboard empty.";
            char buf[256] = {}; fread(buf, 1, 255, f); pclose(f);
            return std::string(buf).empty() ? "Clipboard is empty." : std::string(buf);
        }
        if (p == "write") {
            std::string text = arg(action.args, "text");
            system(("echo " + q(text) + " | wl-copy").c_str());
            return "Copied.";
        }
    }
    if (action.type == "notify") {
        std::string msg = arg(action.args, "message");
        system(("notify-send 'ARIA' " + q(msg)).c_str());
        return "";
    }
    if (action.type == "url") {
        std::string url = arg(action.args, "url");
        shell("xdg-open " + q(url));
        return "Opening link.";
    }
    if (action.type == "sequence") {
        std::string steps = arg(action.args, "steps");
        std::istringstream ss(steps);
        std::string step;
        while (std::getline(ss, step, '|')) {
            auto sp = step.find(' ');
            AgentAction sub;
            sub.type = (sp == std::string::npos) ? step : step.substr(0, sp);
            sub.args = {{"param", (sp == std::string::npos) ? "" : step.substr(sp + 1)}};
            execute(sub);
        }
        return "Done.";
    }

    // ─── Phase 2A: File operations ───
    if (action.type == "file_read") {
        std::string path = expandHome(arg(action.args, "path"));
        int maxLines = argInt(action.args, "max_lines", 50);
        if (path.empty()) return "No path specified.";
        return shellCapture("head -n " + std::to_string(maxLines) + " " + q(path));
    }
    if (action.type == "file_write") {
        std::string path = expandHome(arg(action.args, "path"));
        std::string content = arg(action.args, "content");
        bool append = argBool(action.args, "append", false);
        if (path.empty()) return "No path specified.";
        // Use C++ file I/O to avoid shell injection
        auto mode = append ? (std::ios::out | std::ios::app) : std::ios::out;
        std::ofstream f(path, mode);
        if (!f) return "Cannot write to " + path;
        f << content;
        f.close();
        Logger::info("Executor: wrote " + std::to_string(content.size()) + " bytes to " + path);
        return "File written.";
    }
    if (action.type == "file_search") {
        std::string dir = expandHome(arg(action.args, "directory"));
        if (dir.empty()) dir = expandHome("~");
        std::string pattern = arg(action.args, "pattern");
        std::string grep = arg(action.args, "content_grep");
        if (pattern.empty()) return "No search pattern specified.";
        std::string cmd = "find " + q(dir) + " -maxdepth 5 -name " + q(pattern) + " 2>/dev/null | head -20";
        if (!grep.empty())
            cmd = "find " + q(dir) + " -maxdepth 5 -name " + q(pattern) +
                  " -exec grep -l " + q(grep) + " {} + 2>/dev/null | head -20";
        return shellCapture(cmd);
    }
    if (action.type == "file_list") {
        std::string path = expandHome(arg(action.args, "path"));
        if (path.empty()) path = expandHome("~");
        return shellCapture("ls -lah " + q(path) + " | head -30");
    }

    // ─── Phase 2B: Window management ───
    if (action.type == "window") {
        std::string act = arg(action.args, "action");
        if (act.empty()) return "No window action specified.";
        // Map friendly names to niri action names
        std::string niriAction = act;
        if (act == "maximize") niriAction = "maximize-column";
        else if (act == "fullscreen") niriAction = "fullscreen-window";
        else if (act == "center") niriAction = "center-column";
        system(("niri msg action " + q(niriAction) + " 2>/dev/null").c_str());
        return "Done.";
    }
    if (action.type == "list_windows") {
        std::string raw = shellCapture("niri msg --json windows 2>/dev/null");
        try {
            auto windows = json::parse(raw);
            std::ostringstream out;
            for (auto& w : windows) {
                std::string app = w.value("app_id", "unknown");
                std::string title = w.value("title", "");
                if (title.size() > 60) title = title.substr(0, 60) + "...";
                out << app << ": " << title << "\n";
            }
            std::string result = out.str();
            return result.empty() ? "No windows open." : result;
        } catch (...) {
            return raw; // return raw output if JSON parse fails
        }
    }
    if (action.type == "focus_window") {
        std::string app = arg(action.args, "app");
        if (app.empty()) return "No app specified.";
        std::string appLower = app;
        std::transform(appLower.begin(), appLower.end(), appLower.begin(), ::tolower);
        // Find window ID by app name
        std::string raw = shellCapture("niri msg --json windows 2>/dev/null");
        try {
            auto windows = json::parse(raw);
            for (auto& w : windows) {
                std::string wApp = w.value("app_id", "");
                std::string wAppLower = wApp;
                std::transform(wAppLower.begin(), wAppLower.end(), wAppLower.begin(), ::tolower);
                if (wAppLower.find(appLower) != std::string::npos) {
                    int id = w.value("id", 0);
                    if (id > 0) {
                        system(("niri msg action focus-window --id " + std::to_string(id) + " 2>/dev/null").c_str());
                        return "Focused " + wApp + ".";
                    }
                }
            }
        } catch (...) {}
        return "Couldn't find a window for " + app + ".";
    }

    // ─── Phase 2C: Process management ───
    if (action.type == "proc_list") {
        std::string filter = arg(action.args, "filter");
        if (filter.empty())
            return shellCapture("ps aux --sort=-%mem | head -15");
        return shellCapture("ps aux | grep -i " + q(filter) + " | grep -v grep | head -15");
    }
    if (action.type == "proc_kill") {
        std::string target = arg(action.args, "target");
        if (target.empty()) return "No process specified.";
        bool isPid = std::all_of(target.begin(), target.end(), ::isdigit);
        if (isPid) {
            int pid = std::atoi(target.c_str());
            if (pid <= 1) {
                Logger::warn("Executor: DENIED kill of pid " + target);
                return "I won't kill that process.";
            }
            return shellCapture("kill " + target);
        }
        if (isCriticalProc(target)) {
            Logger::warn("Executor: DENIED kill of critical process: " + target);
            return "I won't kill " + target + " — it's a critical process.";
        }
        return shellCapture("pkill -f " + q(target));
    }

    // ─── Phase 2D: Timers (delegated to daemon's TimerManager) ───
    if (action.type == "timer_set" || action.type == "timer_list" || action.type == "timer_cancel") {
        // These are handled by the daemon, not the executor
        // Return a marker so the daemon knows to handle it
        return "__TIMER__";
    }

    // ─── Notifications ───
    if (action.type == "read_notifications") {
        std::string out = shellCapture("dunstctl history 2>/dev/null | head -c 600");
        if (out.empty() || out == "Command completed successfully." || out == "Command failed.")
            out = shellCapture("makoctl list 2>/dev/null | head -c 600");
        if (out.empty() || out == "Command completed successfully." || out == "Command failed.")
            return "No notifications found.";
        return out;
    }

    // ─── Music search (Spotify/Youtube) ───
    if (action.type == "music_search") {
        std::string query = arg(action.args, "query");
        if (query.empty()) return "No song specified.";

        // URL-encode query for the browser fallback.
        std::string encoded;
        for (char c : query) {
            if (c == ' ')                            encoded += "%20";
            else if (std::isalnum(static_cast<unsigned char>(c))) encoded += c;
            else {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%%%02X", static_cast<unsigned char>(c));
                encoded += buf;
            }
        }

        // If Spotify is running, use its MPRIS OpenUri to trigger a search.
        std::string check = shellCapture("playerctl -l 2>/dev/null | grep -i spotify");
        if (!check.empty() && check.find("spotify") != std::string::npos) {
            std::string uri = "spotify:search:" + query;
            std::string cmd =
                "dbus-send --print-reply --dest=org.mpris.MediaPlayer2.spotify "
                "/org/mpris/MediaPlayer2 org.mpris.MediaPlayer2.Player.OpenUri "
                "string:" + q(uri) + " 2>/dev/null";
            system(cmd.c_str());
            return "Searching Spotify for " + query + ".";
        }

        // Fallback: open Spotify web search in the default browser.
        shell("xdg-open " + q("https://open.spotify.com/search/" + encoded));
        return "Opening Spotify search for " + query + ".";
    }

    // ─── Fact storage — handled by daemon which owns Memory ───
    if (action.type == "remember") {
        return "__REMEMBER__";
    }

    // ─── Phase 2E: Web search ───
    if (action.type == "web_search") {
        std::string query = arg(action.args, "query");
        if (query.empty()) return "No query specified.";
        // Use DuckDuckGo lite HTML and extract results
        std::string cmd = "curl -s 'https://lite.duckduckgo.com/lite/?q=" + q(query).substr(1, q(query).size()-2) + "'"
                          " | sed -n 's/.*<a[^>]*href=\"\\([^\"]*\\)\"[^>]*class=\"result-link\"[^>]*>\\(.*\\)<\\/a>.*/\\2: \\1/p'"
                          " | head -5 2>/dev/null";
        std::string result = shellCapture(cmd);
        if (result.empty() || result == "Command completed successfully.") {
            // Fallback: use curl to DuckDuckGo API
            std::string escaped;
            for (char c : query) {
                if (c == ' ') escaped += '+';
                else escaped += c;
            }
            result = shellCapture("curl -s 'https://api.duckduckgo.com/?q=" + escaped + "&format=json&no_html=1' "
                                  "| python3 -c \"import sys,json; d=json.load(sys.stdin); "
                                  "[print(r.get('Text','')[:200]) for r in (d.get('RelatedTopics',[]))[:5] if r.get('Text')]\" 2>/dev/null");
        }
        if (result.empty() || result == "Command completed successfully.")
            return "No results found. Try opening a browser search.";
        return result;
    }

    // ─── Phase 10: Vision (VLM + OCR) ───
    if (action.type == "see_screen") {
        std::string question = arg(action.args, "question");
        if (question.empty()) question = "Describe what is visible on the screen.";
        auto r = Vision::describeScreen(question);
        if (r.ok) return r.text;
        // VLM failed (timeout, Ollama down, model swap stall) — degrade to OCR
        // so the ReAct loop still has *something* textual to reason over
        // instead of a dead-end error. Screen::capture uses grim+tesseract.
        Screen::invalidateCache();
        std::string ocr = Screen::capture("");
        if (!ocr.empty())
            return "Vision model unavailable; OCR fallback reads: " + ocr;
        return r.text;
    }
    if (action.type == "describe_region") {
        int x = argInt(action.args, "x", -1);
        int y = argInt(action.args, "y", -1);
        int w = argInt(action.args, "width", -1);
        int h = argInt(action.args, "height", -1);
        std::string question = arg(action.args, "question");
        if (x < 0 || y < 0 || w <= 0 || h <= 0)
            return "Need x, y, width, and height to look at a region.";
        if (question.empty()) question = "Describe what is in this region.";
        auto r = Vision::describeRegion(x, y, w, h, question);
        if (!r.ok) return r.text;
        return r.text;
    }
    if (action.type == "read_screen_text") {
        // Bypass the Screen class cache so we always return *current* OCR to
        // the LLM — the cache is keyed for Context::capture's short-lived
        // snapshot, not for an explicit "read the screen right now" tool call.
        Screen::invalidateCache();
        std::string text = Screen::capture("");
        if (text.empty()) return "Nothing readable on screen.";
        return text;
    }

    // ─── Phase 2F: System diagnostics ───
    // Every category returns a one-line spoken sentence so Piper doesn't have
    // to read raw `df -h` / `free -h` columns out loud.
    if (action.type == "sysinfo") {
        std::string cat = arg(action.args, "category");
        if (cat.empty()) cat = "all";

        auto rtrim = [](std::string& s) {
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                                  s.back() == ' '  || s.back() == '\t'))
                s.pop_back();
        };

        // Pull a compact "23G" → "23 gigs" / "512M" → "512 megs" reading.
        auto humanSize = [](const std::string& tok) -> std::string {
            if (tok.empty()) return tok;
            char unit = tok.back();
            std::string num = tok.substr(0, tok.size() - 1);
            switch (unit) {
                case 'T': case 't': return num + " terabytes";
                case 'G': case 'g': return num + " gigs";
                case 'M': case 'm': return num + " megs";
                case 'K': case 'k': return num + " kay";
                default: return tok;
            }
        };

        auto sentenceDisk = [&]() -> std::string {
            std::string raw = shellCapture(
                "df -h --output=target,size,avail / /home 2>/dev/null | tail -n +2");
            rtrim(raw);
            if (raw.empty()) return "";
            std::vector<std::string> parts;
            std::istringstream iss(raw);
            std::string line;
            while (std::getline(iss, line)) {
                std::istringstream ls(line);
                std::string mount, size, avail;
                if (!(ls >> mount >> size >> avail)) continue;
                std::string label = (mount == "/") ? "Root" :
                                    (mount == "/home") ? "Home" : mount;
                parts.push_back(label + " has " + humanSize(avail) +
                                " free of " + humanSize(size));
            }
            if (parts.empty()) return "";
            std::string out;
            for (size_t i = 0; i < parts.size(); ++i) {
                if (i) out += (i + 1 == parts.size() ? ", and " : ", ");
                out += parts[i];
            }
            return out + ".";
        };

        auto sentenceMemory = [&]() -> std::string {
            std::string raw = shellCapture("free -h | awk 'NR==2{print $2,$3,$7}'");
            rtrim(raw);
            if (raw.empty()) return "";
            std::istringstream iss(raw);
            std::string total, used, avail;
            if (!(iss >> total >> used >> avail)) return "";
            return "Using " + humanSize(used) + " of " + humanSize(total) +
                   ", " + humanSize(avail) + " available.";
        };

        auto sentenceCpu = [&]() -> std::string {
            // `top -bn1` line 3 has "%Cpu(s):  X.X us, ..."; capture us+sy.
            std::string raw = shellCapture(
                "top -bn1 | awk '/^%Cpu/ {gsub(\",\",\"\"); print $2+$4; exit}'");
            rtrim(raw);
            if (raw.empty()) return "";
            // Round to nearest int for cleaner speech.
            try {
                double pct = std::stod(raw);
                int rounded = static_cast<int>(pct + 0.5);
                return "CPU load is around " + std::to_string(rounded) + " percent.";
            } catch (...) { return "CPU load is " + raw + " percent."; }
        };

        auto sentenceBattery = [&]() -> std::string {
            std::string bat = shellCapture(
                "cat /sys/class/power_supply/BAT*/capacity 2>/dev/null");
            rtrim(bat);
            if (bat.empty() || bat.find("No such file") != std::string::npos ||
                bat.find("Command failed") != std::string::npos)
                return "";
            std::string status = shellCapture(
                "cat /sys/class/power_supply/BAT*/status 2>/dev/null");
            rtrim(status);
            std::string suffix = ".";
            if (status == "Charging")    suffix = " and charging.";
            else if (status == "Discharging") suffix = " on battery.";
            return "Battery is at " + bat + " percent" + suffix;
        };

        auto sentenceNetwork = [&]() -> std::string {
            // pick the first UP iface and its IPv4
            std::string raw = shellCapture(
                "ip -br -4 addr 2>/dev/null | awk '$2==\"UP\"{print $1,$3; exit}'");
            rtrim(raw);
            if (raw.empty()) return "Not connected.";
            std::istringstream iss(raw);
            std::string iface, cidr;
            if (!(iss >> iface >> cidr)) return "Not connected.";
            // strip /XX prefix
            auto slash = cidr.find('/');
            std::string ip = (slash == std::string::npos) ? cidr : cidr.substr(0, slash);
            return "Connected on " + iface + " with IP " + ip + ".";
        };

        std::vector<std::string> sentences;
        if (cat == "disk"    || cat == "all") { auto s = sentenceDisk();    if (!s.empty()) sentences.push_back(s); }
        if (cat == "memory"  || cat == "all") { auto s = sentenceMemory();  if (!s.empty()) sentences.push_back(s); }
        if (cat == "cpu"     || cat == "all") { auto s = sentenceCpu();     if (!s.empty()) sentences.push_back(s); }
        if (cat == "battery" || cat == "all") { auto s = sentenceBattery(); if (!s.empty()) sentences.push_back(s); }
        if (cat == "network" || cat == "all") { auto s = sentenceNetwork(); if (!s.empty()) sentences.push_back(s); }

        std::string out;
        for (size_t i = 0; i < sentences.size(); ++i) {
            if (i) out += " ";
            out += sentences[i];
        }
        return out;
    }

    Logger::warn("Executor: unknown action: " + action.type);
    return "";
}
