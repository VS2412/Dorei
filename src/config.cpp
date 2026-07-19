#include "config.hpp"
#include "logger.hpp"
#include <fstream>
#include <cstdlib>
#include <string>

AriaConfig Config::cfg_;
bool       Config::loaded_ = false;

static std::string expandHome(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Strip surrounding quotes if present ("foo" / 'foo' → foo)
static std::string unquote(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"'  && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\'')))
        return s.substr(1, s.size() - 2);
    return s;
}

static std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && v[0] != '\0') ? std::string(v) : fallback;
}

static int envOrInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v && v[0] != '\0') {
        try { return std::stoi(v); } catch (...) {}
    }
    return fallback;
}

static void parseInt(const std::string& val, int& out) {
    try { out = std::stoi(val); } catch (...) {}
}

void Config::load() {
    if (loaded_) return;
    loaded_ = true;

    // ─── Defaults ───
    cfg_.whisper_model    = "~/.local/share/aria/ggml-small.en.bin";
    cfg_.piper_model      = "~/.local/share/piper/en_US-lessac-medium.onnx";
    cfg_.ollama_model     = "qwen3:8b";
    cfg_.ollama_url       = "http://localhost:11434";
    cfg_.db_path          = "~/.aria_memory.db";
    cfg_.log_path         = "~/.ai-agent.log";
    cfg_.sqlite_vec_path  = "~/.local/lib/vec0.so";
    cfg_.embedding_model  = "nomic-embed-text";

    cfg_.wake_model_dir   = "~/.local/share/aria/wakeword";
    cfg_.wake_model       = "hey_jarvis_v0.1.onnx";
    cfg_.wake_cue         = "voice";
    cfg_.wake_threshold   = 0.5f;
    cfg_.wake_trigger     = 4;
    cfg_.wake_refractory  = 20;
    cfg_.wake_window_sec  = 10;
    cfg_.wake_enabled     = true;
    cfg_.wake_debug       = false;
    cfg_.whisper_gpu      = false;
    cfg_.always_listen    = false;

    // 2 (was 3) — fvad aggressiveness 0-3. Mode 3 fires on fan hum / clock
    // ticks, generating sub-second WAVs that Whisper then hallucinates
    // sentences from. Mode 2 still catches normal speech but is less
    // trigger-happy on background noise.
    cfg_.vad_mode         = 2;
    cfg_.max_react_steps  = 5;
    cfg_.llm_timeout_sec  = 60;
    cfg_.volume_step      = 5;
    cfg_.brightness_step  = 10;
    cfg_.embedding_dim    = 768;
    cfg_.vec_backfill     = true;

    cfg_.vlm_model            = "moondream";
    cfg_.vlm_timeout_sec      = 20;
    cfg_.vision_cache_ttl_sec = 60;

    // ─── Config file overrides ───
    std::string configPath = expandHome("~/.config/aria/config.toml");
    std::ifstream f(configPath);
    if (f) {
        std::string line;
        while (std::getline(f, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = trim(line.substr(0, eq));
            std::string val = unquote(trim(line.substr(eq + 1)));

            if      (key == "whisper_model")    cfg_.whisper_model   = val;
            else if (key == "piper_model")      cfg_.piper_model     = val;
            else if (key == "ollama_model")     cfg_.ollama_model    = val;
            else if (key == "ollama_url")       cfg_.ollama_url      = val;
            else if (key == "db_path")          cfg_.db_path         = val;
            else if (key == "log_path")         cfg_.log_path        = val;
            else if (key == "sqlite_vec_path")  cfg_.sqlite_vec_path = val;
            else if (key == "embedding_model")  cfg_.embedding_model = val;
            else if (key == "vad_mode")         parseInt(val, cfg_.vad_mode);
            else if (key == "max_react_steps")  parseInt(val, cfg_.max_react_steps);
            else if (key == "llm_timeout_sec")  parseInt(val, cfg_.llm_timeout_sec);
            else if (key == "volume_step")      parseInt(val, cfg_.volume_step);
            else if (key == "brightness_step")  parseInt(val, cfg_.brightness_step);
            else if (key == "embedding_dim")    parseInt(val, cfg_.embedding_dim);
            else if (key == "vec_backfill")     cfg_.vec_backfill    = (val == "1" || val == "true" || val == "yes");
            else if (key == "wake_model_dir")   cfg_.wake_model_dir  = val;
            else if (key == "wake_model")       cfg_.wake_model      = val;
            else if (key == "wake_cue")         cfg_.wake_cue        = val;
            else if (key == "wake_threshold")   { try { cfg_.wake_threshold = std::stof(val); } catch(...) {} }
            else if (key == "wake_trigger")     parseInt(val, cfg_.wake_trigger);
            else if (key == "wake_refractory")  parseInt(val, cfg_.wake_refractory);
            else if (key == "wake_window_sec")  parseInt(val, cfg_.wake_window_sec);
            else if (key == "wake_enabled")     cfg_.wake_enabled    = (val == "1" || val == "true" || val == "yes");
            else if (key == "wake_debug")       cfg_.wake_debug      = (val == "1" || val == "true" || val == "yes");
            else if (key == "whisper_gpu")      cfg_.whisper_gpu     = (val == "1" || val == "true" || val == "yes");
            else if (key == "always_listen")    cfg_.always_listen   = (val == "1" || val == "true" || val == "yes");
            else if (key == "vlm_model")        cfg_.vlm_model       = val;
            else if (key == "vlm_timeout_sec")  parseInt(val, cfg_.vlm_timeout_sec);
            else if (key == "vision_cache_ttl_sec") parseInt(val, cfg_.vision_cache_ttl_sec);
        }
    }

    // ─── Env var overrides (highest priority) ───
    cfg_.whisper_model    = envOr("ARIA_WHISPER_MODEL",    cfg_.whisper_model);
    cfg_.piper_model      = envOr("ARIA_PIPER_MODEL",      cfg_.piper_model);
    cfg_.ollama_model     = envOr("ARIA_MODEL",            cfg_.ollama_model);
    cfg_.ollama_url       = envOr("ARIA_OLLAMA_URL",       cfg_.ollama_url);
    cfg_.db_path          = envOr("ARIA_DB_PATH",          cfg_.db_path);
    cfg_.log_path         = envOr("ARIA_LOG_PATH",         cfg_.log_path);
    cfg_.sqlite_vec_path  = envOr("ARIA_SQLITE_VEC_PATH",  cfg_.sqlite_vec_path);
    cfg_.embedding_model  = envOr("ARIA_EMBEDDING_MODEL",  cfg_.embedding_model);
    cfg_.vad_mode         = envOrInt("ARIA_VAD_MODE",        cfg_.vad_mode);
    cfg_.max_react_steps  = envOrInt("ARIA_MAX_REACT_STEPS", cfg_.max_react_steps);
    cfg_.llm_timeout_sec  = envOrInt("ARIA_LLM_TIMEOUT",     cfg_.llm_timeout_sec);
    cfg_.volume_step      = envOrInt("ARIA_VOLUME_STEP",     cfg_.volume_step);
    cfg_.brightness_step  = envOrInt("ARIA_BRIGHTNESS_STEP", cfg_.brightness_step);
    cfg_.embedding_dim    = envOrInt("ARIA_EMBEDDING_DIM",   cfg_.embedding_dim);
    if (const char* v = std::getenv("ARIA_VEC_BACKFILL"))
        cfg_.vec_backfill = (std::string(v) == "1" || std::string(v) == "true");

    // Phase 9 wake-word env overrides
    cfg_.wake_model_dir = envOr("ARIA_WAKE_MODEL_DIR", cfg_.wake_model_dir);
    cfg_.wake_model     = envOr("ARIA_WAKE_MODEL",     cfg_.wake_model);
    cfg_.wake_cue       = envOr("ARIA_WAKE_CUE",       cfg_.wake_cue);
    if (const char* v = std::getenv("ARIA_WAKE_THRESHOLD"))
        { try { cfg_.wake_threshold = std::stof(v); } catch(...) {} }
    cfg_.wake_trigger    = envOrInt("ARIA_WAKE_TRIGGER",     cfg_.wake_trigger);
    cfg_.wake_refractory = envOrInt("ARIA_WAKE_REFRACTORY",  cfg_.wake_refractory);
    cfg_.wake_window_sec = envOrInt("ARIA_WAKE_WINDOW_SEC",  cfg_.wake_window_sec);
    if (const char* v = std::getenv("ARIA_WAKE_ENABLED"))
        cfg_.wake_enabled = (std::string(v) == "1" || std::string(v) == "true");
    if (const char* v = std::getenv("ARIA_WAKE_DEBUG"))
        cfg_.wake_debug = (std::string(v) == "1" || std::string(v) == "true");
    if (const char* v = std::getenv("ARIA_WHISPER_GPU"))
        cfg_.whisper_gpu = (std::string(v) == "1" || std::string(v) == "true");
    if (const char* v = std::getenv("ARIA_ALWAYS_LISTEN"))
        cfg_.always_listen = (std::string(v) == "1" || std::string(v) == "true");

    // Phase 10 VLM env overrides
    cfg_.vlm_model            = envOr("ARIA_VLM_MODEL",          cfg_.vlm_model);
    cfg_.vlm_timeout_sec      = envOrInt("ARIA_VLM_TIMEOUT",     cfg_.vlm_timeout_sec);
    cfg_.vision_cache_ttl_sec = envOrInt("ARIA_VISION_CACHE_TTL",cfg_.vision_cache_ttl_sec);

    // Legacy: ARIA_WAKE_WORD=1 (old text-prefix flag) implies audio wake enabled.
    // Users who set =0 explicitly want wake off.
    if (const char* v = std::getenv("ARIA_WAKE_WORD")) {
        std::string s(v);
        if (s == "0" || s == "false") cfg_.wake_enabled = false;
        else if (!s.empty())          cfg_.wake_enabled = true;
    }

    // Expand ~ in every path after all overrides have landed.
    cfg_.whisper_model   = expandHome(cfg_.whisper_model);
    cfg_.piper_model     = expandHome(cfg_.piper_model);
    cfg_.db_path         = expandHome(cfg_.db_path);
    cfg_.log_path        = expandHome(cfg_.log_path);
    cfg_.sqlite_vec_path = expandHome(cfg_.sqlite_vec_path);
    cfg_.wake_model_dir  = expandHome(cfg_.wake_model_dir);
}

const AriaConfig& Config::get() {
    if (!loaded_) load();
    return cfg_;
}
