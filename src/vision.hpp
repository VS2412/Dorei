#pragma once
#include <string>

// Vision-Language Model client. Captures screen (or region) via grim and sends
// the PNG bytes + a natural-language question to Ollama's multimodal chat API
// using the model configured by Config::vlm_model (default: moondream).
//
// Calls are cache-gated: (screenHash, question) → answer with a short TTL so
// "what's on screen?" asked twice in a row doesn't re-invoke the VLM.
//
// On timeout / error, ok=false and `text` carries a user-facing reason string.
class Vision {
public:
    struct Result {
        std::string text;          // VLM answer OR error message when !ok
        bool        ok        = false;
        bool        cached    = false;
        int         latencyMs = 0;
    };

    // Capture entire focused output and ask the VLM `question`.
    static Result describeScreen(const std::string& question);

    // Capture a rectangular region (wayland pixels) and ask the VLM `question`.
    static Result describeRegion(int x, int y, int w, int h,
                                 const std::string& question);

    // Force-drop the (screenHash, question) → answer cache.
    static void clearCache();

    // Delete any stale /tmp/aria_vlm_*.png left behind by a crashed prior run.
    // Call once at daemon startup.
    static void cleanupStaleTempFiles();

private:
    // Shared between describeScreen / describeRegion.
    static Result describePng(const std::string& pngPath,
                              const std::string& question,
                              const std::string& cacheKeyTag);
};
