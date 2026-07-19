#include "vision.hpp"
#include "logger.hpp"
#include "config.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;
namespace clk = std::chrono;

namespace {

// ─── base64 (PEM-style, no line wrap) ───
const char* kB64Alpha =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64Encode(const unsigned char* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= len) {
        uint32_t n = (data[i] << 16) | (data[i+1] << 8) | data[i+2];
        out.push_back(kB64Alpha[(n >> 18) & 0x3F]);
        out.push_back(kB64Alpha[(n >> 12) & 0x3F]);
        out.push_back(kB64Alpha[(n >>  6) & 0x3F]);
        out.push_back(kB64Alpha[ n        & 0x3F]);
        i += 3;
    }
    if (i < len) {
        uint32_t n = data[i] << 16;
        if (i + 1 < len) n |= data[i+1] << 8;
        out.push_back(kB64Alpha[(n >> 18) & 0x3F]);
        out.push_back(kB64Alpha[(n >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? kB64Alpha[(n >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

// FNV-1a — cheap, stable hash for caching. Not cryptographic; we just want
// "same screen bytes" → same key.
std::string fnv1a(const unsigned char* data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016lx", static_cast<unsigned long>(h));
    return buf;
}

// Slurp a file into a byte buffer. Empty on failure.
std::vector<unsigned char> readBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto end = f.tellg();
    if (end <= 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(static_cast<size_t>(end));
    if (!f.read(reinterpret_cast<char*>(buf.data()), end)) return {};
    return buf;
}

size_t curlWrite(void* ptr, size_t sz, size_t nmemb, std::string* out) {
    out->append(static_cast<char*>(ptr), sz * nmemb);
    return sz * nmemb;
}

// ─── (hash|question) → answer cache with TTL + LRU bound ───
struct CacheEntry {
    std::string answer;
    clk::steady_clock::time_point at;
};
constexpr size_t kCacheMaxEntries = 64;

std::mutex                                    g_cacheMu;
std::unordered_map<std::string, CacheEntry>   g_cache;
std::list<std::string>                        g_cacheOrder;  // LRU: front = oldest

void cachePut(const std::string& key, const std::string& answer) {
    std::lock_guard<std::mutex> lk(g_cacheMu);
    g_cache[key] = {answer, clk::steady_clock::now()};
    g_cacheOrder.remove(key);
    g_cacheOrder.push_back(key);
    while (g_cacheOrder.size() > kCacheMaxEntries) {
        g_cache.erase(g_cacheOrder.front());
        g_cacheOrder.pop_front();
    }
}

bool cacheGet(const std::string& key, int ttlSec, std::string& answerOut) {
    std::lock_guard<std::mutex> lk(g_cacheMu);
    auto it = g_cache.find(key);
    if (it == g_cache.end()) return false;
    auto age = clk::duration_cast<clk::seconds>(
                   clk::steady_clock::now() - it->second.at).count();
    if (age > ttlSec) {
        g_cache.erase(it);
        g_cacheOrder.remove(key);
        return false;
    }
    answerOut = it->second.answer;
    // Touch for LRU.
    g_cacheOrder.remove(key);
    g_cacheOrder.push_back(key);
    return true;
}

// Per-call unique temp path to avoid concurrent callers trampling each other.
std::string uniqueTempPngPath(const char* tag) {
    static std::atomic<uint64_t> seq{0};
    char buf[128];
    std::snprintf(buf, sizeof(buf), "/tmp/aria_vlm_%s_%d_%llu.png",
                  tag, getpid(),
                  static_cast<unsigned long long>(seq.fetch_add(1)));
    return buf;
}

void unlinkQuiet(const std::string& path) {
    if (path.empty()) return;
    std::string cmd = "rm -f '" + path + "' 2>/dev/null";
    std::system(cmd.c_str());
}

}  // namespace

void Vision::clearCache() {
    std::lock_guard<std::mutex> lk(g_cacheMu);
    g_cache.clear();
    g_cacheOrder.clear();
}

void Vision::cleanupStaleTempFiles() {
    // Wildcard rm of our temp namespace. Only touches files we own.
    std::system("rm -f /tmp/aria_vlm_*.png 2>/dev/null");
}

// NOTE (Phase 1): prewarm() is gone on purpose. On a 6GB card, pushing
// moondream into Ollama at startup EVICTED the primary brain and caused the
// 16-second first turns measured in Phase 0. Vision now runs CPU-pinned
// (num_gpu:0 below) — first call pays a RAM load, the brain never moves.

Vision::Result Vision::describePng(const std::string& pngPath,
                                   const std::string& question,
                                   const std::string& cacheKeyTag) {
    Result r;
    auto t0 = clk::steady_clock::now();

    auto bytes = readBytes(pngPath);
    if (bytes.empty()) {
        r.text = "Screen capture produced no image.";
        return r;
    }

    // Cache key = content hash + question (+ optional tag to distinguish full
    // vs region calls on the same region bytes — unlikely to collide, but cheap
    // insurance).
    std::string key = fnv1a(bytes.data(), bytes.size()) + "|" + cacheKeyTag
                    + "|" + question;

    const auto& cfg = Config::get();
    std::string cached;
    if (cacheGet(key, cfg.vision_cache_ttl_sec, cached)) {
        r.ok        = true;
        r.cached    = true;
        r.text      = cached;
        r.latencyMs = 0;
        Logger::info("Vision: cache hit (" + cacheKeyTag + ")");
        return r;
    }

    std::string b64 = base64Encode(bytes.data(), bytes.size());

    // Moondream tends to over-explain. Nudge for a tight, spoken-style answer.
    std::string framedQuestion =
        question + "\n\nAnswer in one or two short sentences, plain spoken English.";

    // Ollama /api/chat with multimodal messages. `images` must be an array of
    // base64 strings — NO `data:image/png;base64,` prefix.
    json body;
    body["model"]      = cfg.vlm_model;
    body["stream"]     = false;
    body["think"]      = false;
    body["keep_alive"] = "10m";  // keep model resident between nearby calls
    // CPU-pinned: the VLM must never compete with the primary brain for the
    // 6GB card (model-class separation — see foundation.md resource governor).
    body["options"]    = {{"num_gpu", 0}};
    body["messages"]   = json::array({
        {
            {"role",    "user"},
            {"content", framedQuestion},
            {"images",  json::array({b64})},
        }
    });

    CURL* curl = curl_easy_init();
    if (!curl) { r.text = "VLM client init failed."; return r; }

    std::string response;
    auto* headers = curl_slist_append(nullptr, "Content-Type: application/json");
    std::string url = cfg.ollama_url + "/api/chat";
    std::string bodyStr = body.dump();

    curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    bodyStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)bodyStr.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,       (long)cfg.vlm_timeout_sec);

    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    r.latencyMs = static_cast<int>(
        clk::duration_cast<clk::milliseconds>(
            clk::steady_clock::now() - t0).count());

    if (rc != CURLE_OK) {
        r.text = std::string("VLM request failed: ") + curl_easy_strerror(rc);
        Logger::error("Vision: " + r.text);
        return r;
    }

    try {
        auto parsed = json::parse(response);
        if (parsed.contains("error")) {
            r.text = "VLM error: " + parsed["error"].get<std::string>();
            Logger::error("Vision: " + r.text);
            return r;
        }
        if (!parsed.contains("message")) {
            r.text = "VLM returned no message";
            Logger::error("Vision: " + r.text + " | raw: " + response.substr(0, 200));
            return r;
        }
        auto& msg = parsed["message"];
        if (!msg.contains("content") || msg["content"].is_null()) {
            r.text = "VLM returned empty content";
            return r;
        }
        r.text = msg["content"].get<std::string>();

        // Trim whitespace for tidy TTS.
        auto s = r.text.find_first_not_of(" \t\r\n");
        auto e = r.text.find_last_not_of(" \t\r\n");
        if (s == std::string::npos) {
            r.text = "The model returned nothing.";
            return r;
        }
        r.text = r.text.substr(s, e - s + 1);
        r.ok = true;

        cachePut(key, r.text);
        Logger::info("Vision: " + cacheKeyTag + " answered in " +
                     std::to_string(r.latencyMs) + "ms");
    } catch (const std::exception& ex) {
        r.text = std::string("VLM parse error: ") + ex.what();
        Logger::error("Vision: " + r.text + " | raw: " + response.substr(0, 300));
    }
    return r;
}

Vision::Result Vision::describeScreen(const std::string& question) {
    std::string path = uniqueTempPngPath("screen");
    std::string cmd  = "grim '" + path + "' 2>/dev/null";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        unlinkQuiet(path);
        Result r;
        r.text = "Couldn't capture the screen. Is grim installed and a Wayland session active?";
        return r;
    }
    auto res = describePng(path, question, "full");
    unlinkQuiet(path);
    return res;
}

Vision::Result Vision::describeRegion(int x, int y, int w, int h,
                                      const std::string& question) {
    if (w <= 0 || h <= 0) {
        Result r; r.text = "Invalid region dimensions."; return r;
    }
    std::string path = uniqueTempPngPath("region");
    char geom[128];
    std::snprintf(geom, sizeof(geom), "%d,%d %dx%d", x, y, w, h);
    std::string cmd = "grim -g '" + std::string(geom) + "' '" + path + "' 2>/dev/null";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        unlinkQuiet(path);
        Result r; r.text = "Couldn't capture that region."; return r;
    }
    std::ostringstream tag;
    tag << "region:" << x << "," << y << "," << w << "x" << h;
    auto res = describePng(path, question, tag.str());
    unlinkQuiet(path);
    return res;
}
