// `arise` CLI — Phase 1.
//
// Drives the MemoryCortex and IdentityStore directly. No daemon yet; later
// phases turn these into IPC calls to a long-running process. Subcommands
// are deliberately handrolled (no new deps) — the surface is small.

#include "blackboard/blackboard.hpp"
#include "cortex/adapter_registry.hpp"
#include "cortex/coder.hpp"
#include "cortex/critic.hpp"
#include "cortex/curator.hpp"
#include "cortex/device_store.hpp"
#include "cortex/federation_router.hpp"
#include "cortex/feedback_db.hpp"
#include "cortex/forge_tool.hpp"
#include "cortex/goal_scheduler.hpp"
#include "cortex/goals.hpp"
#include "cortex/identity.hpp"
#include "cortex/memory_cortex.hpp"
#include "cortex/orchestrator.hpp"
#include "cortex/persona_prompt.hpp"
#include "cortex/piper_engine.hpp"
#include "cortex/privacy_filter.hpp"
#include "cortex/proactive.hpp"
#include "cortex/researcher.hpp"
#include "cortex/salience.hpp"
#include "cortex/sandbox.hpp"
#include "cortex/spawn_handle.hpp"
#include "cortex/speech.hpp"
#include "cortex/sub_agent.hpp"
#include "cortex/suggestion.hpp"
#include "cortex/tool_registry.hpp"
#include "cortex/training_curator.hpp"
#include "cortex/watcher.hpp"
#include "perception/audio_scene.hpp"
#include "perception/mic_capture.hpp"
#include "perception/perception.hpp"
#include "perception/privacy_gate.hpp"
#include "perception/system_snapshot.hpp"
#include "util/log.hpp"
#include "util/paths.hpp"
#include "util/vision_client.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <sqlite3.h>

using nlohmann::json;
using namespace std::chrono;

namespace {

// ─── small arg helpers ──────────────────────────────────────────────────

struct Args {
    std::vector<std::string> pos;                  // positional
    std::unordered_map<std::string, std::string> opt;  // --key val
    std::unordered_map<std::string, bool>        flag; // --key

    bool has(const std::string& k) const {
        return opt.count(k) > 0 || flag.count(k) > 0;
    }
    std::string get(const std::string& k, std::string def = "") const {
        auto it = opt.find(k);
        return it == opt.end() ? std::move(def) : it->second;
    }
};

Args parseArgs(int argc, char** argv, int start) {
    Args a;
    for (int i = start; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            std::string k = s.substr(2);
            if (i + 1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0) {
                a.opt[k] = argv[++i];
            } else {
                a.flag[k] = true;
            }
        } else {
            a.pos.push_back(std::move(s));
        }
    }
    return a;
}

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

// ─── default cortex config ───────────────────────────────────────────────

arise::MemoryCortex::Config defaultCortexConfig() {
    arise::paths::ensureLayout();
    arise::MemoryCortex::Config c;
    c.root              = arise::paths::memoryDir();
    c.embed_cache_path  = arise::paths::cacheDir() + "/embeddings.sqlite";
    c.decay_thread      = false; // CLI runs are short-lived

    auto vec_so = arise::paths::expandHome("~/.local/lib/vec0.so");
    if (arise::paths::fileExists(vec_so)) c.sqlite_vec_path = vec_so;

    if (const char* u = std::getenv("ARIA_OLLAMA_URL")) c.ollama_url = u;
    if (const char* m = std::getenv("ARIA_EMBEDDING_MODEL")) c.embed_model = m;
    if (const char* d = std::getenv("ARIA_EMBEDDING_DIM"))   c.embed_dim = std::atoi(d);
    return c;
}

// ─── help ────────────────────────────────────────────────────────────────

int cmdHelp(int = 0) {
    std::cout <<
"arise — cognitive OS CLI (phase 1)\n"
"\n"
"USAGE\n"
"  arise <command> [args...]\n"
"\n"
"COMMANDS\n"
"  init                              create ~/.arise/ layout + default identity\n"
"\n"
"  mem record --kind K --summary S [--mood M] [--salience F]\n"
"             [--payload PATH] [--decay-days N]\n"
"  mem recall <text> [--limit N] [--types e,s,p]\n"
"  mem dump   [--recent N]\n"
"  mem mood   [show | nudge V A [LABEL] | baseline LABEL | tick]\n"
"  mem purge                          drop decayed events now\n"
"  mem fact   --subject S --predicate P --object O [--confidence F]\n"
"  mem facts  --subject S [--predicate P]\n"
"\n"
"  identity show\n"
"  identity set [--name N] [--pronouns P] [--persona TEXT]\n"
"               [--add-do TEXT] [--add-dont TEXT]\n"
"\n"
"  goal propose --summary S [--priority N] [--deadline ISO|+Nd|+Nh|EPOCH]\n"
"               [--parent ID] [--tags t1,t2] [--plan FILE]\n"
"  goal accept|start|complete|cancel|reject ID [--note TEXT]\n"
"  goal block ID --reason TEXT\n"
"  goal unblock ID [--note TEXT]\n"
"  goal show ID\n"
"  goal list [--status S] [--parent ID] [--tag T] [--limit N] [--by-priority]\n"
"  goal search <text> [--limit N]\n"
"  goal tree ID                              walk subtree rooted at ID\n"
"  goal due [--horizon-sec N]                deadlines within horizon\n"
"  goal stale [--days N]                     in-progress goals untouched\n"
"  goal set-plan ID FILE                     replace plan_json from file\n"
"  goal set-priority ID N\n"
"  goal set-deadline ID ISO|+Nd|+Nh|EPOCH|none\n"
"  goal touch ID                             bump last_progress_at to now\n"
"  goal scheduler [--tick-sec N] [--due-horizon-sec N] [--escalate-sec N]\n"
"                 [--stale-days N] [--seconds N]\n"
"                 [--no-resume-boot] [--no-resume-idle]\n"
"                 run goal scheduler, tail goal.* events to stdout\n"
"\n"
"  agents start [--no-watcher] [--no-curator] [--seconds N]\n"
"               [--curator-model NAME] [--ollama-url URL]\n"
"               run Watcher + Curator, tail agent.* events to stdout\n"
"  watcher fire {battery|caption|audio} VALUE\n"
"               inject a synthetic signal and print the resulting decision\n"
"  curator absorb --transcript FILE [--model NAME] [--no-upsert]\n"
"               extract facts from a transcript file via qwen3:0.6b\n"
"\n"
"  agents spawn {researcher|coder} --task TEXT [--sandbox DIR]\n"
"               [--max-seconds N] [--max-tool-calls N] [--model NAME]\n"
"               run a one-shot ReAct researcher or sandboxed coder\n"
"  critic review {--file FILE | --content TEXT} [--llm] [--require-llm]\n"
"               regex denylist + optional llm judge over arbitrary content\n"
"\n"
"  tools list   [--builtins] [--json]\n"
"  tools show   ID\n"
"  tools run    ID --args '{...}'  [--timeout-sec N] [--allow-network]\n"
"  tools approve ID [--by NAME]\n"
"  tools archive ID\n"
"  tools remove  ID\n"
"  tools sandbox-exec --                   raw bwrap probe (read-only /, no net)\n"
"      [--allow-network] [--timeout-sec N] [--writable PATH]... -- ARGV\n"
"  tools register --id NAME --description T --interpreter NAME\n"
"      --script FILE [--args-schema JSON] [--allow-network] [--writable PATH]...\n"
"      register a learned tool from a script on disk\n"
"  tools forge   --description T [--id NAME] [--args-schema JSON]\n"
"                [--example-args JSON] [--auto-approve] [--dry-run-only]\n"
"                [--model NAME] [--ollama-url URL]\n"
"                drive Coder→Critic→sandbox dry-run, then stage if ok\n"
"  tools sweep   [--days N] [--dry-run]\n"
"                archive learned tools unused for N days (default 90)\n"
"\n"
"  proactive start [--seconds N] [--quiet-hours] [--quiet-start H] [--quiet-end H]\n"
"      [--ambient-min-sec N] [--active-min-sec N] [--mute-after N] [--mute-hours N]\n"
"      [--publish-dropped] [--with-watcher]\n"
"      run engine, tail proactive.suggestion + .dropped events\n"
"  proactive list   [--limit N] [--decision X] [--tier X] [--category X]\n"
"  proactive decide ID --accept|--reject|--ignore\n"
"  proactive mute   CATEGORY [--hours N]    (use --hours 0 to clear)\n"
"  proactive timeout [--older-than-sec N]   sweep stale Pending suggestions\n"
"\n"
"  federation pair --name TEXT [--kind phone|tablet|desktop|mqtt|overlay|other]\n"
"                  [--no-utterance] [--no-decision] [--no-goal-query]\n"
"                  [--allow-notification] [--allow-screen-share]\n"
"                  pair a shard, prints {id, token} once — token isn't stored\n"
"  federation list                       list paired devices (no tokens shown)\n"
"  federation revoke ID                  remove a device from the registry\n"
"  federation ingest --token T --json EVENT_JSON\n"
"                  process one event end-to-end (same path as the WS layer)\n"
"\n"
"  speak \"TEXT\" [--mood X] [--no-tts] [--voice MODEL_PATH]\n"
"        [--length-scale N] [--noise-scale N] [--sentence-silence-sec N]\n"
"        speak `TEXT` via Piper, mood adjusts pacing + expressiveness\n"
"  persona prompt [--mood X] [--user-name N] [--no-mood-line] [--no-do-dont]\n"
"        print the system-prompt prefix every sub-agent will see\n"
"\n"
"  self curate   [--lookback-hours N] [--top-n N] [--no-llm] [--out FILE]\n"
"                replay last N hours of episodic, score, redact, write JSONL\n"
"  self privacy-check {--text T | --file F}     run the privacy scrubber\n"
"  self adapter list\n"
"  self adapter register --id N --path P --base-model M [--score F] [--note T]\n"
"  self adapter promote ID         mark ID as deployed (clears others)\n"
"  self adapter rollback ID\n"
"  self adapter rotate [--keep N]\n"
"\n"
"  perceive [--vision-fps F] [--no-vision] [--no-system] [--no-idle]\n"
"           [--threshold N] [--seconds N] [--episodic] [--verbose]\n"
"           [--private app1,app2] [--strict-privacy]\n"
"           [--captioner] [--no-salience] [--caption-cooldown-ms N]\n"
"           [--vision-model NAME] [--salience-model NAME]\n"
"           [--audio-scene] [--audio-device DEV]\n"
"           [--yamnet-model PATH] [--yamnet-labels PATH]\n"
"           [--yamnet-min-score F]\n"
"           run perception loop, print live blackboard events to stdout\n"
"\n"
"  system snapshot                   one-shot system state JSON\n"
"  privacy check [--private apps]    test the privacy gate against current focus\n"
"  blackboard tail                   (placeholder; in-process bus, see perceive)\n"
"\n"
"  import-aria [--source PATH] [--dry-run]\n"
"\n"
"ENV\n"
"  ARISE_ROOT          override ~/.arise\n"
"  ARIA_OLLAMA_URL     default http://127.0.0.1:11434\n"
"  ARIA_EMBEDDING_MODEL/DIM\n";
    return 0;
}

// ─── init ────────────────────────────────────────────────────────────────

int cmdInit() {
    auto root = arise::paths::ensureLayout();
    arise::log::init(arise::paths::logsDir() + "/arise.log");
    arise::log::info("arise init: layout at " + root);

    arise::IdentityStore ids(arise::paths::identityDir());
    auto rec = ids.get();
    if (rec.do_list.empty() && rec.dont_list.empty()) {
        rec.do_list  = {"address the user by their preferred name",
                        "be specific and brief",
                        "remember and reuse what worked before"};
        rec.dont_list = {"pad replies with hedging",
                         "claim memory I don't actually have"};
        ids.set(std::move(rec), "initial identity");
    }

    // Touch the cortex once to materialise empty schemas.
    {
        arise::MemoryCortex cortex(defaultCortexConfig());
        (void)cortex.totalSizeBytes();
    }

    std::cout << "ARISE initialised at " << root << "\n";
    return 0;
}

// ─── mem ─────────────────────────────────────────────────────────────────

int cmdMemRecord(const Args& a) {
    arise::EpisodicEvent ev;
    ev.kind    = a.get("kind", "note");
    ev.summary = a.get("summary");
    ev.mood_at = a.get("mood");
    if (a.has("salience")) ev.salience = std::atof(a.get("salience").c_str());
    if (a.has("payload"))  ev.payload_json = slurp(a.get("payload"));
    if (a.has("decay-days")) {
        int d = std::atoi(a.get("decay-days").c_str());
        if (d > 0) ev.decay_at = system_clock::now() + std::chrono::hours(24 * d);
    }
    if (ev.summary.empty()) {
        std::cerr << "mem record: --summary required\n";
        return 2;
    }

    arise::MemoryCortex cortex(defaultCortexConfig());
    auto id = cortex.recordEvent(std::move(ev));
    if (id <= 0) { std::cerr << "mem record: failed\n"; return 1; }
    std::cout << "id=" << id << "\n";
    return 0;
}

std::vector<arise::MemoryType> parseTypes(const std::string& csv) {
    std::vector<arise::MemoryType> out;
    if (csv.empty()) {
        return {arise::MemoryType::Episodic, arise::MemoryType::Semantic};
    }
    std::istringstream iss(csv);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
        if      (tok == "e" || tok == "episodic")   out.push_back(arise::MemoryType::Episodic);
        else if (tok == "s" || tok == "semantic")   out.push_back(arise::MemoryType::Semantic);
        else if (tok == "p" || tok == "procedural") out.push_back(arise::MemoryType::Procedural);
    }
    return out;
}

int cmdMemRecall(const Args& a) {
    if (a.pos.empty()) { std::cerr << "mem recall: <text> required\n"; return 2; }
    std::string text = a.pos[0];
    for (size_t i = 1; i < a.pos.size(); ++i) text += " " + a.pos[i];

    arise::RecallQuery q;
    q.text  = text;
    q.limit = a.has("limit") ? std::atoi(a.get("limit").c_str()) : 8;
    q.types = parseTypes(a.get("types"));

    arise::MemoryCortex cortex(defaultCortexConfig());
    auto hits = cortex.recall(q);
    if (hits.empty()) { std::cout << "(no hits)\n"; return 0; }

    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        std::printf("%2zu  [%-10s] score=%.4f  %s\n",
                    i + 1, arise::toString(h.type), h.score, h.rendered.c_str());
    }
    return 0;
}

int cmdMemDump(const Args& a) {
    int n = a.has("recent") ? std::atoi(a.get("recent").c_str()) : 20;
    arise::MemoryCortex cortex(defaultCortexConfig());
    auto evs = cortex.recentEvents(n);
    if (evs.empty()) { std::cout << "(empty)\n"; return 0; }
    for (const auto& ev : evs) {
        std::time_t t = system_clock::to_time_t(ev.ts);
        char ts[24];
        std::tm tm_buf{}; localtime_r(&t, &tm_buf);
        std::strftime(ts, sizeof ts, "%F %T", &tm_buf);
        std::printf("%5lld  %s  %-18s sal=%.2f  %s%s\n",
                    static_cast<long long>(ev.id), ts, ev.kind.c_str(),
                    ev.salience,
                    ev.mood_at.empty() ? "" : ("[" + ev.mood_at + "] ").c_str(),
                    ev.summary.c_str());
    }
    return 0;
}

int cmdMemMood(const Args& a) {
    arise::MemoryCortex cortex(defaultCortexConfig());

    std::string sub = a.pos.empty() ? "show" : a.pos[0];

    if (sub == "show") {
        auto m = cortex.mood();
        std::printf("baseline=%s current=%s valence=%.2f arousal=%.2f\n",
                    m.baseline.c_str(), m.current.c_str(), m.valence, m.arousal);
        return 0;
    }
    if (sub == "tick") {
        cortex.tickMoodDecay();
        return cmdMemMood({{}, {}, {{"_pos","show"}}});  // unused; just show
    }
    if (sub == "baseline" && a.pos.size() >= 2) {
        cortex.setMoodBaseline(a.pos[1]);
        std::cout << "baseline=" << a.pos[1] << "\n";
        return 0;
    }
    if (sub == "nudge" && a.pos.size() >= 3) {
        double v = std::atof(a.pos[1].c_str());
        double ar = std::atof(a.pos[2].c_str());
        std::string label = a.pos.size() >= 4 ? a.pos[3] : "";
        cortex.nudgeMood(v, ar, label);
        auto m = cortex.mood();
        std::printf("nudged → current=%s valence=%.2f arousal=%.2f\n",
                    m.current.c_str(), m.valence, m.arousal);
        return 0;
    }
    std::cerr << "mem mood: unknown subcommand\n";
    return 2;
}

int cmdMemPurge() {
    arise::MemoryCortex cortex(defaultCortexConfig());
    int n = cortex.purgeDecayed();
    std::cout << "purged " << n << " events\n";
    return 0;
}

int cmdMemFact(const Args& a) {
    arise::SemanticFact f;
    f.subject   = a.get("subject");
    f.predicate = a.get("predicate");
    f.object    = a.get("object");
    f.confidence = a.has("confidence")
        ? std::atof(a.get("confidence").c_str()) : 1.0;
    if (f.subject.empty() || f.predicate.empty() || f.object.empty()) {
        std::cerr << "mem fact: --subject, --predicate, --object required\n";
        return 2;
    }
    arise::MemoryCortex cortex(defaultCortexConfig());
    auto id = cortex.upsertFact(std::move(f));
    if (id <= 0) {
        std::cerr << "mem fact: rejected (lower confidence than existing)\n";
        return 1;
    }
    std::cout << "fact id=" << id << "\n";
    return 0;
}

int cmdMemFacts(const Args& a) {
    if (!a.has("subject")) {
        std::cerr << "mem facts: --subject required\n";
        return 2;
    }
    arise::MemoryCortex cortex(defaultCortexConfig());
    auto facts = cortex.queryFacts(a.get("subject"), a.get("predicate"));
    if (facts.empty()) { std::cout << "(none)\n"; return 0; }
    for (const auto& f : facts) {
        std::printf("%4lld  %s %s %s  conf=%.2f\n",
                    static_cast<long long>(f.id),
                    f.subject.c_str(), f.predicate.c_str(),
                    f.object.c_str(), f.confidence);
    }
    return 0;
}

// Forward decls — definitions live further down the file, but earlier
// command handlers (federation, scheduler, etc.) lean on them so they can
// tail blackboard events with the same formatting + signal handling, or
// reuse the cortex / feedback / goals defaults.
std::vector<std::string> splitCsv(const std::string& s);
std::string              truncatePayload(const std::string& s, std::size_t cap);
std::string              fmtClock(arise::BBTimestamp ts);
arise::FeedbackDb::Config defaultFeedbackCfg();
extern std::atomic<bool> g_stop;
extern "C" void onSig(int);

// ─── goals ───────────────────────────────────────────────────────────────

// Parse a deadline spec: "+Nd", "+Nh", "+Nm", a 10-digit-ish epoch second,
// or an ISO-ish date / datetime ("2026-05-15", "2026-05-15 14:30",
// "2026-05-15T14:30:00"). Returns nullopt on parse failure or "none".
std::optional<std::chrono::system_clock::time_point>
parseDeadlineSpec(const std::string& s) {
    using namespace std::chrono;
    if (s.empty() || s == "none" || s == "null" || s == "off") return std::nullopt;

    // Relative: +12d / +3h / +30m
    if (s.size() >= 2 && s[0] == '+') {
        char unit = s.back();
        std::string num = s.substr(1, s.size() - 2);
        long long n = 0;
        try { n = std::stoll(num); } catch (...) { return std::nullopt; }
        seconds delta{0};
        if      (unit == 'd') delta = duration_cast<seconds>(hours(24 * n));
        else if (unit == 'h') delta = duration_cast<seconds>(hours(n));
        else if (unit == 'm') delta = duration_cast<seconds>(minutes(n));
        else if (unit == 's') delta = seconds(n);
        else return std::nullopt;
        return system_clock::now() + delta;
    }

    // Bare epoch seconds (digits only).
    bool all_digits = !s.empty() && std::all_of(s.begin(), s.end(),
                          [](unsigned char c) { return std::isdigit(c); });
    if (all_digits) {
        try {
            long long e = std::stoll(s);
            return system_clock::time_point(seconds(e));
        } catch (...) { return std::nullopt; }
    }

    // ISO-ish parse via strptime, several patterns.
    static const char* kFmts[] = {
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    };
    for (const char* fmt : kFmts) {
        std::tm tm{};
        if (strptime(s.c_str(), fmt, &tm)) {
            tm.tm_isdst = -1;
            std::time_t t = std::mktime(&tm);
            if (t != (time_t)-1) return system_clock::from_time_t(t);
        }
    }
    return std::nullopt;
}

std::string fmtIso(arise::GoalTimestamp t) {
    std::time_t tt = std::chrono::system_clock::to_time_t(t);
    std::tm tm_buf{}; localtime_r(&tt, &tm_buf);
    char buf[24];
    std::strftime(buf, sizeof buf, "%F %T", &tm_buf);
    return buf;
}

arise::GoalStore::Config defaultGoalsConfig(arise::MemoryCortex* sink = nullptr) {
    arise::paths::ensureLayout();
    arise::GoalStore::Config c;
    c.db_path = arise::paths::goalsDbPath();
    c.episodic_sink = sink;
    return c;
}

void printGoalRow(const arise::Goal& g, bool compact = true) {
    if (compact) {
        std::printf("%5lld  [%-11s] p=%3d  %s%s%s\n",
                    static_cast<long long>(g.id),
                    arise::toString(g.status),
                    g.priority,
                    g.summary.c_str(),
                    g.parent_id ? ("  ↑#" + std::to_string(*g.parent_id)).c_str() : "",
                    g.deadline_at
                        ? ("  due " + fmtIso(*g.deadline_at)).c_str()
                        : "");
    } else {
        json j;
        j["id"]               = g.id;
        j["summary"]          = g.summary;
        j["status"]           = arise::toString(g.status);
        j["priority"]         = g.priority;
        j["created_at"]       = fmtIso(g.created_at);
        j["last_progress_at"] = fmtIso(g.last_progress_at);
        if (g.deadline_at)    j["deadline_at"]     = fmtIso(*g.deadline_at);
        else                  j["deadline_at"]     = nullptr;
        if (g.parent_id)      j["parent_id"]       = *g.parent_id;
        if (!g.blocked_reason.empty()) j["blocked_reason"] = g.blocked_reason;
        if (!g.tags.empty())  j["tags"]            = g.tags;
        if (!g.plan_json.empty()) j["plan_json"]   = g.plan_json;
        std::cout << j.dump(2) << "\n";
    }
}

std::optional<arise::GoalStatus> parseStatusFlag(const Args& a) {
    if (!a.has("status")) return std::nullopt;
    auto v = a.get("status");
    auto s = arise::goalStatusFromString(v);
    if (!s) std::cerr << "warn: unknown status '" << v << "', ignored\n";
    return s;
}

int cmdGoalPropose(const Args& a) {
    if (!a.has("summary")) {
        std::cerr << "goal propose: --summary required\n"; return 2;
    }
    arise::Goal g;
    g.summary  = a.get("summary");
    g.priority = a.has("priority") ? std::atoi(a.get("priority").c_str()) : 50;
    if (a.has("parent")) g.parent_id = std::atoll(a.get("parent").c_str());
    if (a.has("deadline")) {
        auto d = parseDeadlineSpec(a.get("deadline"));
        if (!d) { std::cerr << "goal propose: bad --deadline\n"; return 2; }
        g.deadline_at = d;
    }
    if (a.has("tags")) g.tags = splitCsv(a.get("tags"));
    if (a.has("plan")) g.plan_json = slurp(a.get("plan"));

    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::GoalStore store(defaultGoalsConfig(&cortex));
    auto id = store.propose(std::move(g));
    if (id <= 0) { std::cerr << "goal propose: failed\n"; return 1; }
    std::cout << "id=" << id << "\n";
    return 0;
}

// Generic "transition" handler: accept|start|complete|cancel|reject ID
int cmdGoalTransition(const Args& a, arise::GoalStatus to) {
    if (a.pos.empty()) {
        std::cerr << "goal: ID required\n"; return 2;
    }
    auto id = std::atoll(a.pos[0].c_str());
    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::GoalStore store(defaultGoalsConfig(&cortex));
    if (!store.setStatus(id, to, a.get("note"))) {
        std::cerr << "goal: refused (not found / invalid transition)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}

int cmdGoalBlock(const Args& a) {
    if (a.pos.empty() || !a.has("reason")) {
        std::cerr << "goal block: ID and --reason required\n"; return 2;
    }
    auto id = std::atoll(a.pos[0].c_str());
    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::GoalStore store(defaultGoalsConfig(&cortex));
    if (!store.block(id, a.get("reason"))) {
        std::cerr << "goal block: refused\n"; return 1;
    }
    std::cout << "ok\n"; return 0;
}

int cmdGoalUnblock(const Args& a) {
    if (a.pos.empty()) {
        std::cerr << "goal unblock: ID required\n"; return 2;
    }
    auto id = std::atoll(a.pos[0].c_str());
    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::GoalStore store(defaultGoalsConfig(&cortex));
    if (!store.unblock(id, a.get("note"))) {
        std::cerr << "goal unblock: refused (not currently blocked?)\n";
        return 1;
    }
    std::cout << "ok\n"; return 0;
}

int cmdGoalShow(const Args& a) {
    if (a.pos.empty()) { std::cerr << "goal show: ID required\n"; return 2; }
    auto id = std::atoll(a.pos[0].c_str());
    arise::GoalStore store(defaultGoalsConfig());
    auto g = store.get(id);
    if (!g) { std::cerr << "(not found)\n"; return 1; }
    printGoalRow(*g, /*compact=*/false);
    return 0;
}

int cmdGoalList(const Args& a) {
    arise::GoalQuery q;
    if (auto s = parseStatusFlag(a))    q.status = s;
    if (a.has("parent"))                q.parent_id = std::atoll(a.get("parent").c_str());
    if (a.has("tag"))                   q.tag       = a.get("tag");
    if (a.has("limit"))                 q.limit     = std::atoi(a.get("limit").c_str());
    q.order_by_priority = a.has("by-priority");

    arise::GoalStore store(defaultGoalsConfig());
    auto rows = store.list(q);
    if (rows.empty()) { std::cout << "(none)\n"; return 0; }
    for (auto& g : rows) printGoalRow(g);
    return 0;
}

int cmdGoalSearch(const Args& a) {
    if (a.pos.empty()) { std::cerr << "goal search: <text> required\n"; return 2; }
    std::string text = a.pos[0];
    for (size_t i = 1; i < a.pos.size(); ++i) text += " " + a.pos[i];
    arise::GoalQuery q;
    q.text  = text;
    q.limit = a.has("limit") ? std::atoi(a.get("limit").c_str()) : 20;
    arise::GoalStore store(defaultGoalsConfig());
    auto rows = store.list(q);
    if (rows.empty()) { std::cout << "(no hits)\n"; return 0; }
    for (auto& g : rows) printGoalRow(g);
    return 0;
}

int cmdGoalTree(const Args& a) {
    if (a.pos.empty()) { std::cerr << "goal tree: ROOT_ID required\n"; return 2; }
    auto id = std::atoll(a.pos[0].c_str());
    arise::GoalStore store(defaultGoalsConfig());
    auto rows = store.subtree(id);
    if (rows.empty()) { std::cerr << "(not found)\n"; return 1; }
    // Render with 2-space indent per depth derived from parent walk.
    std::unordered_map<std::int64_t, int> depth;
    depth[id] = 0;
    for (auto& g : rows) {
        if (g.parent_id && depth.count(*g.parent_id))
            depth[g.id] = depth[*g.parent_id] + 1;
        std::printf("%*s", depth[g.id] * 2, "");
        printGoalRow(g);
    }
    return 0;
}

int cmdGoalDue(const Args& a) {
    auto secs = a.has("horizon-sec") ? std::atoll(a.get("horizon-sec").c_str())
                                     : 24 * 3600;
    arise::GoalStore store(defaultGoalsConfig());
    auto rows = store.dueSoon(std::chrono::seconds(secs));
    if (rows.empty()) { std::cout << "(no goals due in horizon)\n"; return 0; }
    for (auto& g : rows) printGoalRow(g);
    return 0;
}

int cmdGoalStale(const Args& a) {
    auto days = a.has("days") ? std::atoi(a.get("days").c_str()) : 7;
    arise::GoalStore store(defaultGoalsConfig());
    auto rows = store.staleInProgress(std::chrono::seconds(days * 24 * 3600));
    if (rows.empty()) { std::cout << "(none stale)\n"; return 0; }
    for (auto& g : rows) printGoalRow(g);
    return 0;
}

int cmdGoalSetPlan(const Args& a) {
    if (a.pos.size() < 2) { std::cerr << "goal set-plan: ID FILE required\n"; return 2; }
    auto id = std::atoll(a.pos[0].c_str());
    auto blob = slurp(a.pos[1]);
    if (blob.empty()) { std::cerr << "goal set-plan: empty plan file\n"; return 2; }
    arise::GoalStore store(defaultGoalsConfig());
    if (!store.setPlanJson(id, blob)) { std::cerr << "goal set-plan: failed\n"; return 1; }
    std::cout << "ok\n"; return 0;
}

int cmdGoalSetPriority(const Args& a) {
    if (a.pos.size() < 2) { std::cerr << "goal set-priority: ID N required\n"; return 2; }
    auto id = std::atoll(a.pos[0].c_str());
    int  n  = std::atoi(a.pos[1].c_str());
    arise::GoalStore store(defaultGoalsConfig());
    if (!store.setPriority(id, n)) { std::cerr << "goal set-priority: failed\n"; return 1; }
    std::cout << "ok\n"; return 0;
}

int cmdGoalSetDeadline(const Args& a) {
    if (a.pos.size() < 2) { std::cerr << "goal set-deadline: ID SPEC required\n"; return 2; }
    auto id = std::atoll(a.pos[0].c_str());
    arise::GoalStore store(defaultGoalsConfig());
    auto d = parseDeadlineSpec(a.pos[1]);   // empty / "none" → nullopt → clear
    if (!store.setDeadline(id, d)) { std::cerr << "goal set-deadline: failed\n"; return 1; }
    std::cout << "ok\n"; return 0;
}

int cmdGoalTouch(const Args& a) {
    if (a.pos.empty()) { std::cerr << "goal touch: ID required\n"; return 2; }
    auto id = std::atoll(a.pos[0].c_str());
    arise::GoalStore store(defaultGoalsConfig());
    if (!store.bumpProgress(id)) { std::cerr << "goal touch: failed\n"; return 1; }
    std::cout << "ok\n"; return 0;
}

int cmdGoalScheduler(const Args& a) {
    arise::GoalScheduler::Config sc;
    if (a.has("tick-sec"))         sc.tick_interval     = std::chrono::seconds(std::atoll(a.get("tick-sec").c_str()));
    if (a.has("due-horizon-sec"))  sc.due_horizon       = std::chrono::seconds(std::atoll(a.get("due-horizon-sec").c_str()));
    if (a.has("escalate-sec"))     sc.escalate_horizon  = std::chrono::seconds(std::atoll(a.get("escalate-sec").c_str()));
    if (a.has("stale-days")) {
        long long d = std::atoll(a.get("stale-days").c_str());
        sc.stale_threshold = std::chrono::seconds(d * 24 * 3600);
    }
    if (a.has("due-renotify-sec"))      sc.due_renotify      = std::chrono::seconds(std::atoll(a.get("due-renotify-sec").c_str()));
    if (a.has("escalate-renotify-sec")) sc.escalate_renotify = std::chrono::seconds(std::atoll(a.get("escalate-renotify-sec").c_str()));
    if (a.has("stale-renotify-sec"))    sc.stale_renotify    = std::chrono::seconds(std::atoll(a.get("stale-renotify-sec").c_str()));
    if (a.has("resume-renotify-sec"))   sc.resume_renotify   = std::chrono::seconds(std::atoll(a.get("resume-renotify-sec").c_str()));
    if (a.has("no-resume-boot")) sc.resume_on_boot      = false;
    if (a.has("no-resume-idle")) sc.resume_on_idle_left = false;

    arise::Blackboard   bb;
    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::GoalStore    store(defaultGoalsConfig(&cortex));
    sc.bb    = &bb;
    sc.store = &store;

    arise::GoalScheduler sched(sc);
    auto sub = bb.subscribe("");          // wildcard tail
    sched.start();

    std::cout << "scheduler: tick=" << sc.tick_interval.count() << "s"
              << " due_horizon=" << sc.due_horizon.count() << "s"
              << " escalate=" << sc.escalate_horizon.count() << "s"
              << " stale=" << sc.stale_threshold.count() << "s"
              << "  (Ctrl-C to stop)\n";

    std::signal(SIGINT,  onSig);
    std::signal(SIGTERM, onSig);

    int seconds = a.has("seconds") ? std::atoi(a.get("seconds").c_str()) : -1;
    auto deadline = system_clock::now() + std::chrono::seconds(seconds);

    while (!g_stop.load()) {
        if (seconds > 0 && system_clock::now() >= deadline) break;
        auto ev = sub.next(std::chrono::milliseconds(250));
        if (!ev) continue;
        std::cout << "[" << fmtClock(ev->ts) << "] " << ev->topic
                  << "  " << truncatePayload(ev->payload.dump(), 200) << "\n";
        std::cout.flush();
    }

    sub.stop();
    sched.stop();

    auto st = sched.stats();
    std::cout << "\n-- stats --\n"
              << "scans            = " << st.scans            << "\n"
              << "due_fires        = " << st.due_fires        << "\n"
              << "escalated_fires  = " << st.escalated_fires  << "\n"
              << "stale_fires      = " << st.stale_fires      << "\n"
              << "resumed_fires    = " << st.resumed_fires    << "\n"
              << "idle_lefts_seen  = " << st.idle_lefts_seen  << "\n"
              << "events_total     = " << bb.totalPublished() << "\n";
    return 0;
}

int cmdGoalDispatch(const Args& a) {
    if (a.pos.empty()) { std::cerr << "goal: subcommand required\n"; return 2; }
    std::string sub = a.pos[0];
    Args sa = a; sa.pos.erase(sa.pos.begin());
    if (sub == "propose")        return cmdGoalPropose(sa);
    if (sub == "accept")         return cmdGoalTransition(sa, arise::GoalStatus::Accepted);
    if (sub == "start")          return cmdGoalTransition(sa, arise::GoalStatus::InProgress);
    if (sub == "complete")       return cmdGoalTransition(sa, arise::GoalStatus::Done);
    if (sub == "cancel")         return cmdGoalTransition(sa, arise::GoalStatus::Cancelled);
    if (sub == "reject")         return cmdGoalTransition(sa, arise::GoalStatus::Rejected);
    if (sub == "block")          return cmdGoalBlock(sa);
    if (sub == "unblock")        return cmdGoalUnblock(sa);
    if (sub == "show")           return cmdGoalShow(sa);
    if (sub == "list")           return cmdGoalList(sa);
    if (sub == "search")         return cmdGoalSearch(sa);
    if (sub == "tree")           return cmdGoalTree(sa);
    if (sub == "due")            return cmdGoalDue(sa);
    if (sub == "stale")          return cmdGoalStale(sa);
    if (sub == "set-plan")       return cmdGoalSetPlan(sa);
    if (sub == "set-priority")   return cmdGoalSetPriority(sa);
    if (sub == "set-deadline")   return cmdGoalSetDeadline(sa);
    if (sub == "touch")          return cmdGoalTouch(sa);
    if (sub == "scheduler")      return cmdGoalScheduler(sa);
    std::cerr << "goal: unknown subcommand '" << sub << "'\n";
    return 2;
}

// ─── agents (Phase 4 commit 1) ───────────────────────────────────────────

int cmdAgentsStart(const Args& a) {
    arise::Blackboard   bb;
    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::GoalStore    goals(defaultGoalsConfig(&cortex));

    arise::Orchestrator::Config oc;
    oc.bb     = &bb;
    oc.cortex = &cortex;
    oc.goals  = &goals;
    oc.enable_watcher = !a.has("no-watcher");
    oc.enable_curator = !a.has("no-curator");
    if (a.has("curator-model")) oc.curator_llm.model = a.get("curator-model");
    if (a.has("ollama-url"))    oc.curator_llm.ollama_url = a.get("ollama-url");

    arise::Orchestrator orch(oc);
    auto sub = bb.subscribe("");          // wildcard tail
    orch.start();

    std::cout << "agents: watcher=" << (oc.enable_watcher ? "on" : "off")
              << " curator="        << (oc.enable_curator ? "on" : "off")
              << "  (Ctrl-C to stop)\n";

    std::signal(SIGINT,  onSig);
    std::signal(SIGTERM, onSig);

    int seconds = a.has("seconds") ? std::atoi(a.get("seconds").c_str()) : -1;
    auto deadline = system_clock::now() + std::chrono::seconds(seconds);

    while (!g_stop.load()) {
        if (seconds > 0 && system_clock::now() >= deadline) break;
        auto ev = sub.next(std::chrono::milliseconds(250));
        if (!ev) continue;
        std::cout << "[" << fmtClock(ev->ts) << "] " << ev->topic
                  << "  " << truncatePayload(ev->payload.dump(), 200) << "\n";
        std::cout.flush();
    }

    sub.stop();
    orch.stop();

    if (auto* w = orch.watcher()) {
        auto s = w->stats();
        std::cout << "\n-- watcher --\n"
                  << "events_seen      = " << s.events_seen      << "\n"
                  << "notices_emitted  = " << s.notices_emitted  << "\n"
                  << "goals_proposed   = " << s.goals_proposed   << "\n"
                  << "llm_consultations= " << s.llm_consultations << "\n";
    }
    if (auto* c = orch.curator()) {
        auto s = c->stats();
        std::cout << "\n-- curator --\n"
                  << "conversations    = " << s.conversations_seen << "\n"
                  << "facts_extracted  = " << s.facts_extracted    << "\n"
                  << "facts_upserted   = " << s.facts_upserted     << "\n"
                  << "facts_rejected   = " << s.facts_rejected     << "\n"
                  << "llm_calls        = " << s.llm_calls          << "\n"
                  << "llm_failures     = " << s.llm_failures       << "\n";
    }
    return 0;
}

int cmdWatcherFire(const Args& a) {
    if (a.pos.size() < 2) {
        std::cerr << "watcher fire: KIND VALUE required (battery N | caption TEXT | audio SCENE)\n";
        return 2;
    }
    std::string kind  = a.pos[0];
    std::string value = a.pos[1];
    for (size_t i = 2; i < a.pos.size(); ++i) value += " " + a.pos[i];

    arise::Blackboard bb;
    arise::Watcher::Config wc;
    wc.bb = &bb;
    arise::Watcher w(wc);

    arise::Watcher::Decision d;
    if (kind == "battery") {
        d = w.evaluateBatteryPct(std::atoi(value.c_str()));
    } else if (kind == "caption") {
        d = w.evaluateCaption(value);
    } else if (kind == "audio") {
        d = w.evaluateAudioScene(value);
    } else {
        std::cerr << "watcher fire: unknown kind '" << kind << "'\n";
        return 2;
    }

    json out;
    out["severity"]      = arise::Watcher::severityToString(d.severity);
    out["kind"]          = d.kind;
    out["summary"]       = d.summary;
    out["propose_goal"]  = d.propose_goal;
    if (!d.goal_summary.empty()) out["goal_summary"] = d.goal_summary;
    std::cout << out.dump(2) << "\n";
    return 0;
}

int cmdAgentsSpawn(const Args& a) {
    if (a.pos.empty()) {
        std::cerr << "agents spawn: ROLE required (researcher | coder)\n"; return 2;
    }
    std::string role = a.pos[0];
    if (!a.has("task")) { std::cerr << "agents spawn: --task required\n"; return 2; }
    std::string task = a.get("task");

    std::string sandbox = a.has("sandbox")
        ? a.get("sandbox")
        : (arise::paths::ariseRoot() + "/sandbox");

    int max_seconds = a.has("max-seconds") ? std::atoi(a.get("max-seconds").c_str()) : 60;
    int max_calls   = a.has("max-tool-calls") ? std::atoi(a.get("max-tool-calls").c_str()) : 8;
    std::string model = a.has("model") ? a.get("model") : "qwen3:0.6b";
    std::string ollama_url = a.has("ollama-url") ? a.get("ollama-url") : "";

    arise::SubAgent::Config llm_cfg;
    llm_cfg.role        = role;
    llm_cfg.model       = model;
    llm_cfg.format_json = true;
    llm_cfg.timeout_sec = std::max(10, max_seconds);
    llm_cfg.max_predict = 1024;
    if (!ollama_url.empty()) llm_cfg.ollama_url = ollama_url;
    arise::SubAgent llm(llm_cfg);

    auto handle = arise::makeSpawnHandle(arise::newSpawnId(role), role);
    auto state  = handle.state_ptr();
    state->state.store(arise::SpawnHandle::State::Running);
    state->started_at = std::chrono::steady_clock::now();

    json result_extra;
    int  rc = 0;

    if (role == "researcher") {
        arise::Researcher::Config rc_cfg;
        rc_cfg.llm           = &llm;
        rc_cfg.sandbox_root  = sandbox;
        rc_cfg.max_tool_calls = max_calls;
        rc_cfg.max_wall_time = std::chrono::seconds(max_seconds);
        arise::Researcher r(rc_cfg);

        arise::paths::ensureDir(sandbox);
        auto rr = r.run(task, &state->kill_requested);
        result_extra["answer"]     = rr.answer;
        result_extra["budget_hit"] = rr.budget_hit;
        result_extra["killed"]     = rr.killed;
        result_extra["steps"]      = json::array();
        for (auto& s : rr.steps) {
            result_extra["steps"].push_back({
                {"thought", s.thought}, {"action", s.action},
                {"args",    s.args},    {"observation", s.observation},
            });
        }
        result_extra["error"] = rr.error;
        result_extra["ok"]    = rr.ok;
        rc = rr.ok ? 0 : 1;
    } else if (role == "coder") {
        arise::Critic::Config cc_cfg;
        cc_cfg.require_llm_for_approval = false;
        arise::Critic critic(cc_cfg);

        arise::Coder::Config co_cfg;
        co_cfg.llm           = &llm;
        co_cfg.critic        = &critic;
        co_cfg.sandbox_root  = sandbox;
        co_cfg.max_wall_time = std::chrono::seconds(max_seconds);
        arise::Coder coder(co_cfg);

        arise::paths::ensureDir(sandbox);
        auto cr = coder.run(task, &state->kill_requested);
        result_extra["summary"]      = cr.summary;
        result_extra["sandbox_path"] = cr.sandbox_path;
        result_extra["files"]        = json::array();
        for (auto& f : cr.files) {
            result_extra["files"].push_back({
                {"path", f.path}, {"rel_path", f.rel_path}, {"bytes", f.bytes},
            });
        }
        result_extra["review"]   = {
            {"approved", cr.review.approved},
            {"verdict",  cr.review.verdict},
            {"matches",  cr.review.matches},
            {"from_llm", cr.review.from_llm},
        };
        result_extra["budget_hit"] = cr.budget_hit;
        result_extra["error"]      = cr.error;
        result_extra["ok"]         = cr.ok;
        rc = cr.ok ? 0 : 1;
    } else {
        std::cerr << "agents spawn: unknown role '" << role << "'\n";
        return 2;
    }

    state->finished_at = std::chrono::steady_clock::now();
    state->state.store(arise::SpawnHandle::State::Done);

    json out;
    out["id"]          = handle.id();
    out["role"]        = handle.role();
    out["state"]       = arise::SpawnHandle::stateToString(handle.state());
    out["duration_ms"] = handle.durationMs();
    out["result"]      = std::move(result_extra);
    std::cout << out.dump(2) << "\n";
    return rc;
}

// ─── tools (Phase 5 commit 1) ────────────────────────────────────────────

arise::ToolRegistry::Config defaultRegistryConfig() {
    arise::paths::ensureLayout();
    arise::ToolRegistry::Config c;
    c.root = arise::paths::toolsDir();
    return c;
}

json toolToJson(const arise::ToolInfo& t) {
    json j;
    j["id"]              = t.id;
    j["version"]         = t.version;
    j["description"]     = t.description;
    j["interpreter"]     = t.interpreter;
    j["script_path"]     = t.script_path;
    j["args_schema"]     = t.args_schema;
    j["allow_network"]   = t.allow_network;
    j["writable_paths"]  = t.writable_paths;
    j["approved"]        = t.approved;
    j["approved_at"]     = t.approved_at;
    j["approved_by"]     = t.approved_by;
    j["usage_count"]     = t.usage_count;
    j["last_used"]       = t.last_used;
    j["created_at"]      = t.created_at;
    j["is_builtin"]      = t.is_builtin;
    return j;
}

int cmdToolsList(const Args& a) {
    arise::ToolRegistry reg(defaultRegistryConfig());
    bool include_builtins = !a.has("no-builtins") && !a.has("learned-only");
    if (a.has("builtins") && !a.has("learned-only")) include_builtins = true;
    auto rows = reg.listAll(include_builtins);
    if (a.has("json")) {
        json arr = json::array();
        for (auto& t : rows) arr.push_back(toolToJson(t));
        std::cout << arr.dump(2) << "\n";
        return 0;
    }
    if (rows.empty()) { std::cout << "(no tools registered)\n"; return 0; }
    for (auto& t : rows) {
        std::printf("%-32s [%-10s] %s%s\n",
                    t.id.c_str(),
                    t.is_builtin ? "builtin" :
                       (t.approved ? "approved" : "unapproved"),
                    t.description.c_str(),
                    t.usage_count > 0
                        ? ("  used " + std::to_string(t.usage_count) + "x").c_str()
                        : "");
    }
    return 0;
}

int cmdToolsShow(const Args& a) {
    if (a.pos.empty()) { std::cerr << "tools show: ID required\n"; return 2; }
    arise::ToolRegistry reg(defaultRegistryConfig());
    auto t = reg.get(a.pos[0]);
    if (!t) { std::cerr << "(not found)\n"; return 1; }
    std::cout << toolToJson(*t).dump(2) << "\n";
    return 0;
}

int cmdToolsRun(const Args& a) {
    if (a.pos.empty()) { std::cerr << "tools run: ID required\n"; return 2; }
    auto id = a.pos[0];

    arise::ToolRegistry reg(defaultRegistryConfig());
    auto t = reg.get(id);
    if (!t)  { std::cerr << "(not found)\n"; return 1; }
    if (t->is_builtin) {
        std::cerr << "tools run: '" << id << "' is a builtin — drive it via "
                     "the agents API (mem/goal/etc CLI directly)\n";
        return 2;
    }
    if (!t->approved) {
        std::cerr << "tools run: '" << id << "' is not approved; "
                     "use `arise tools approve "<<id<<"` first\n";
        return 1;
    }
    if (t->script_path.empty() || !arise::paths::fileExists(t->script_path)) {
        std::cerr << "tools run: script_path missing on disk: "
                  << t->script_path << "\n";
        return 1;
    }

    json args = json::object();
    if (a.has("args")) {
        try { args = json::parse(a.get("args")); }
        catch (const std::exception& e) {
            std::cerr << "tools run: --args parse: " << e.what() << "\n"; return 2;
        }
    }
    auto err = reg.validateArgs(id, args);
    if (!err.empty()) { std::cerr << "tools run: " << err << "\n"; return 2; }

    arise::Sandbox::Config sc;
    sc.allow_network  = a.has("allow-network") || t->allow_network;
    sc.writable_paths = t->writable_paths;
    // Bind the script's parent directory read-only so the script is visible
    // even if it lives under /tmp (which the sandbox shadows with a fresh
    // tmpfs). Belt-and-braces: works for any host path not already on `/`.
    {
        auto parent = std::filesystem::path(t->script_path)
                          .parent_path().string();
        if (!parent.empty()) sc.readonly_paths.push_back(parent);
    }
    if (a.has("timeout-sec"))
        sc.timeout = std::chrono::milliseconds(
            std::atoll(a.get("timeout-sec").c_str()) * 1000);
    arise::Sandbox sandbox(sc);

    std::vector<std::string> argv;
    if (!t->interpreter.empty()) argv.push_back(t->interpreter);
    argv.push_back(t->script_path);
    // Pass args as a JSON blob on stdin; tools that want positional args can
    // read it from there. Future: per-schema argv generation.
    auto rr = sandbox.run(argv, args.dump());
    reg.recordUse(id);

    json out;
    out["ok"]                = rr.ok;
    out["exit_code"]         = rr.exit_code;
    out["timed_out"]         = rr.timed_out;
    out["duration_ms"]       = rr.duration_ms;
    out["stdout"]            = rr.stdout_text;
    out["stderr"]            = rr.stderr_text;
    out["stdout_truncated"]  = rr.stdout_truncated;
    out["stderr_truncated"]  = rr.stderr_truncated;
    if (!rr.error.empty())   out["error"] = rr.error;
    std::cout << out.dump(2) << "\n";
    return rr.ok ? 0 : 1;
}

int cmdToolsApprove(const Args& a) {
    if (a.pos.empty()) { std::cerr << "tools approve: ID required\n"; return 2; }
    arise::ToolRegistry reg(defaultRegistryConfig());
    if (!reg.approve(a.pos[0], a.has("by") ? a.get("by") : "user")) {
        std::cerr << "tools approve: not found / failed\n"; return 1;
    }
    std::cout << "ok\n"; return 0;
}

int cmdToolsArchive(const Args& a) {
    if (a.pos.empty()) { std::cerr << "tools archive: ID required\n"; return 2; }
    arise::ToolRegistry reg(defaultRegistryConfig());
    if (!reg.archive(a.pos[0])) { std::cerr << "tools archive: not found\n"; return 1; }
    std::cout << "ok\n"; return 0;
}

int cmdToolsRemove(const Args& a) {
    if (a.pos.empty()) { std::cerr << "tools remove: ID required\n"; return 2; }
    arise::ToolRegistry reg(defaultRegistryConfig());
    if (!reg.remove(a.pos[0])) { std::cerr << "tools remove: not found\n"; return 1; }
    std::cout << "ok\n"; return 0;
}

int cmdToolsSandboxExec(const Args& a) {
    if (a.pos.empty()) {
        std::cerr << "tools sandbox-exec: ARGV required after positional args\n";
        return 2;
    }
    arise::Sandbox::Config sc;
    sc.allow_network = a.has("allow-network");
    if (a.has("timeout-sec"))
        sc.timeout = std::chrono::milliseconds(
            std::atoll(a.get("timeout-sec").c_str()) * 1000);
    if (a.has("writable")) sc.writable_paths.push_back(a.get("writable"));

    arise::Sandbox sandbox(sc);
    auto rr = sandbox.run(a.pos);
    json out;
    out["ok"]          = rr.ok;
    out["exit_code"]   = rr.exit_code;
    out["timed_out"]   = rr.timed_out;
    out["duration_ms"] = rr.duration_ms;
    out["stdout"]      = rr.stdout_text;
    out["stderr"]      = rr.stderr_text;
    if (!rr.error.empty()) out["error"] = rr.error;
    std::cout << out.dump(2) << "\n";
    return rr.ok ? 0 : 1;
}

// ─── speech + persona (Phase 8 commit 1) ─────────────────────────────────

std::string defaultPiperModelPath() {
    return arise::paths::expandHome("~/.local/share/piper/en_US-lessac-medium.onnx");
}

int cmdSpeak(const Args& a) {
    if (a.pos.empty()) {
        std::cerr << "speak: TEXT required\n"; return 2;
    }
    std::string text = a.pos[0];
    for (size_t i = 1; i < a.pos.size(); ++i) text += " " + a.pos[i];

    std::string mood = a.has("mood") ? a.get("mood") : std::string{"neutral"};
    arise::TtsParams params = arise::moodToParams(mood);
    if (a.has("voice"))                params.voice                = a.get("voice");
    if (a.has("length-scale"))         params.length_scale         = std::atof(a.get("length-scale").c_str());
    if (a.has("noise-scale"))          params.noise_scale          = std::atof(a.get("noise-scale").c_str());
    if (a.has("sentence-silence-sec")) params.sentence_silence_sec = std::atof(a.get("sentence-silence-sec").c_str());

    arise::PiperEngine::Config pc;
    pc.default_model_path = a.has("voice") ? a.get("voice") : defaultPiperModelPath();
    pc.play_audio         = !a.has("no-tts");
    arise::PiperEngine engine(pc);

    if (!engine.isAvailable()) {
        std::cerr << "speak: piper engine unavailable (binary, aplay, or model missing)\n";
        return 1;
    }

    arise::Speech::Config sc;
    sc.primary = &engine;
    arise::Speech speech(sc);

    auto stats = speech.say(text, mood, &params);
    json out;
    out["mood"]               = params.mood_label;
    out["sentences"]          = stats.sentences;
    out["sentences_failed"]   = stats.sentences_failed;
    out["bytes_total"]        = stats.bytes_total;
    out["duration_ms"]        = stats.duration_ms;
    out["engine_used"]        = stats.engine_used;
    out["params"]             = {
        {"length_scale",         params.length_scale},
        {"noise_scale",          params.noise_scale},
        {"sentence_silence_sec", params.sentence_silence_sec},
    };
    std::cout << out.dump(2) << "\n";
    return stats.sentences_failed == 0 ? 0 : 1;
}

int cmdPersonaPrompt(const Args& a) {
    arise::IdentityStore ids(arise::paths::identityDir());
    arise::MemoryCortex cortex(defaultCortexConfig());

    arise::PersonaPromptInput in;
    in.identity = ids.get();
    in.mood     = cortex.mood();
    if (a.has("user-name"))   in.user_name = a.get("user-name");
    if (a.has("no-mood-line")) in.include_mood_line = false;
    if (a.has("no-do-dont"))   in.include_do_dont   = false;
    if (a.has("mood")) {
        // Override the live mood for this CLI run only — useful for previewing
        // how a frustrated user's prompt would look without nudging the cortex.
        auto label = a.get("mood");
        in.mood.current = label;
        if      (label == "frustrated") { in.mood.valence = -0.6; in.mood.arousal =  0.5; }
        else if (label == "down")       { in.mood.valence = -0.6; in.mood.arousal =  0.0; }
        else if (label == "excited")    { in.mood.valence =  0.6; in.mood.arousal =  0.5; }
        else if (label == "warm")       { in.mood.valence =  0.6; in.mood.arousal =  0.0; }
        else if (label == "alert")      { in.mood.valence =  0.0; in.mood.arousal =  0.5; }
        else if (label == "tired")      { in.mood.valence =  0.0; in.mood.arousal = -0.5; }
        else                            { in.mood.valence =  0.0; in.mood.arousal =  0.0; }
    }

    auto out = arise::buildPersonaPrompt(in);
    std::cout << out;
    if (out.empty() || out.back() != '\n') std::cout << "\n";
    return 0;
}

// ─── federation (Phase 7 commit 1) ───────────────────────────────────────

arise::DeviceStore::Config defaultDeviceStoreCfg() {
    arise::paths::ensureLayout();
    arise::DeviceStore::Config c;
    c.path = arise::paths::devicesJsonPath();
    return c;
}

json deviceToCliJson(const arise::DeviceInfo& d) {
    auto isoOf = [](std::chrono::system_clock::time_point t) -> std::string {
        if (t.time_since_epoch().count() == 0) return std::string{};
        std::time_t tt = std::chrono::system_clock::to_time_t(t);
        std::tm tm_buf{}; localtime_r(&tt, &tm_buf);
        char buf[24]; std::strftime(buf, sizeof buf, "%F %T", &tm_buf);
        return buf;
    };
    json j;
    j["id"]                = d.id;
    j["name"]              = d.name;
    j["kind"]              = arise::deviceKindToString(d.kind);
    j["paired_at"]         = isoOf(d.paired_at);
    j["last_seen"]         = isoOf(d.last_seen);
    j["event_count"]       = d.event_count;
    json p;
    p["utterance"]    = d.perms.can_utterance;
    p["decision"]     = d.perms.can_decision;
    p["goal_query"]   = d.perms.can_goal_query;
    p["notification"] = d.perms.can_notification;
    p["screen_share"] = d.perms.can_screen_share;
    j["permissions"]       = std::move(p);
    j["token_sha256_hex"]  = d.token_sha256_hex;
    return j;
}

int cmdFederationPair(const Args& a) {
    if (!a.has("name")) {
        std::cerr << "federation pair: --name required\n"; return 2;
    }
    auto kind = arise::DeviceKind::Phone;
    if (a.has("kind")) {
        if (auto k = arise::deviceKindFromString(a.get("kind"))) kind = *k;
    }
    arise::DevicePermissions perms;
    if (a.has("no-utterance"))      perms.can_utterance    = false;
    if (a.has("no-decision"))       perms.can_decision     = false;
    if (a.has("no-goal-query"))     perms.can_goal_query   = false;
    if (a.has("allow-notification")) perms.can_notification = true;
    if (a.has("allow-screen-share")) perms.can_screen_share = true;

    arise::DeviceStore store(defaultDeviceStoreCfg());
    auto added = store.addDevice(a.get("name"), kind, perms);
    if (!added) { std::cerr << "federation pair: failed\n"; return 1; }

    json out;
    out["id"]        = added->device.id;
    out["name"]      = added->device.name;
    out["kind"]      = arise::deviceKindToString(added->device.kind);
    out["token"]     = added->plaintext_token;
    out["note"]      = "save this token now — it is not stored on disk and "
                       "cannot be recovered. only its SHA-256 hash is kept.";
    std::cout << out.dump(2) << "\n";
    return 0;
}

int cmdFederationList(const Args& /*a*/) {
    arise::DeviceStore store(defaultDeviceStoreCfg());
    auto devs = store.list();
    if (devs.empty()) { std::cout << "(no devices paired)\n"; return 0; }
    for (auto& d : devs) {
        std::cout << deviceToCliJson(d).dump() << "\n";
    }
    return 0;
}

int cmdFederationRevoke(const Args& a) {
    if (a.pos.empty()) { std::cerr << "federation revoke: ID required\n"; return 2; }
    arise::DeviceStore store(defaultDeviceStoreCfg());
    if (!store.revokeById(a.pos[0])) {
        std::cerr << "federation revoke: id not found\n"; return 1;
    }
    std::cout << "ok\n"; return 0;
}

int cmdFederationIngest(const Args& a) {
    if (!a.has("token")) {
        std::cerr << "federation ingest: --token required\n"; return 2;
    }
    if (!a.has("json")) {
        std::cerr << "federation ingest: --json EVENT_JSON required\n"; return 2;
    }

    arise::Blackboard   bb;     // local, ephemeral; mostly drives episodic
    arise::DeviceStore  devices(defaultDeviceStoreCfg());
    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::FeedbackDb   feedback(defaultFeedbackCfg());
    arise::GoalStore    goals(defaultGoalsConfig(&cortex));

    arise::FederationRouter::Config rc;
    rc.devices  = &devices;
    rc.bb       = &bb;
    rc.cortex   = &cortex;
    rc.feedback = &feedback;
    rc.goals    = &goals;
    arise::FederationRouter router(rc);

    auto resp = router.ingestRaw(a.get("json"), a.get("token"));
    json out;
    out["ok"]          = resp.ok;
    out["code"]        = arise::FederationRouter::codeToString(resp.code);
    out["error"]       = resp.error;
    out["payload"]     = resp.payload;
    out["event_type"]  = resp.event_type;
    out["source_device"] = resp.source_device;
    std::cout << out.dump(2) << "\n";
    return resp.ok ? 0 : 1;
}

// ─── proactive (Phase 6 commit 1) ────────────────────────────────────────

arise::FeedbackDb::Config defaultFeedbackCfg() {
    arise::paths::ensureLayout();
    arise::FeedbackDb::Config c;
    c.db_path = arise::paths::feedbackDbPath();
    return c;
}

json suggestionRowToJson(const arise::SuggestionRow& r) {
    auto t = std::chrono::system_clock::to_time_t(r.proposed_at);
    std::tm tm_buf{}; localtime_r(&t, &tm_buf);
    char buf[24]; std::strftime(buf, sizeof buf, "%F %T", &tm_buf);

    json j;
    j["id"]                 = r.id;
    j["tier"]               = arise::tierToString(r.tier);
    j["category"]           = r.category;
    j["text"]               = r.text;
    j["source_topic"]       = r.source_topic;
    j["proposed_at"]        = std::string(buf);
    j["decision"]           = arise::decisionToString(r.decision);
    if (r.decided_at) {
        auto t2 = std::chrono::system_clock::to_time_t(*r.decided_at);
        std::tm tm2{}; localtime_r(&t2, &tm2);
        char b2[24]; std::strftime(b2, sizeof b2, "%F %T", &tm2);
        j["decided_at"] = std::string(b2);
    }
    return j;
}

int cmdProactiveStart(const Args& a) {
    arise::Blackboard   bb;
    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::FeedbackDb   feedback(defaultFeedbackCfg());

    arise::ProactiveEngine::Config pc;
    pc.bb       = &bb;
    pc.feedback = &feedback;
    pc.cortex   = &cortex;
    if (a.has("ambient-min-sec"))
        pc.gate.min_interval_ambient = std::chrono::seconds(
            std::atoll(a.get("ambient-min-sec").c_str()));
    if (a.has("active-min-sec"))
        pc.gate.min_interval_active = std::chrono::seconds(
            std::atoll(a.get("active-min-sec").c_str()));
    if (a.has("quiet-hours")) pc.gate.quiet_hours_enabled = true;
    if (a.has("quiet-start")) pc.gate.quiet_start_hour = std::atoi(a.get("quiet-start").c_str());
    if (a.has("quiet-end"))   pc.gate.quiet_end_hour   = std::atoi(a.get("quiet-end").c_str());
    if (a.has("mute-after"))  pc.gate.mute_after_rejects = std::atoi(a.get("mute-after").c_str());
    if (a.has("mute-hours"))  pc.gate.mute_window = std::chrono::seconds(
        std::atoll(a.get("mute-hours").c_str()) * 3600);
    pc.publish_dropped = a.has("publish-dropped");

    arise::ProactiveEngine engine(pc);

    // Optional: also boot a Watcher so vision/audio/system signals get
    // judged into agent.watcher.notice and the proactive engine has work
    // to do without running the full agents-start daemon.
    std::unique_ptr<arise::GoalStore> goals;
    std::unique_ptr<arise::Watcher>  watcher;
    if (a.has("with-watcher")) {
        goals = std::make_unique<arise::GoalStore>(defaultGoalsConfig(&cortex));
        arise::Watcher::Config wc;
        wc.bb     = &bb;
        wc.goals  = goals.get();
        wc.cortex = &cortex;
        watcher = std::make_unique<arise::Watcher>(wc);
    }

    auto sub = bb.subscribe("");
    engine.start();
    if (watcher) watcher->start();

    std::cout << "proactive: ambient_min="
              << pc.gate.min_interval_ambient.count() << "s"
              << " active_min=" << pc.gate.min_interval_active.count() << "s"
              << " quiet=" << (pc.gate.quiet_hours_enabled ? "on" : "off")
              << " mute_after=" << pc.gate.mute_after_rejects
              << " watcher=" << (watcher ? "on" : "off")
              << "  (Ctrl-C to stop)\n";

    std::signal(SIGINT,  onSig);
    std::signal(SIGTERM, onSig);

    int seconds = a.has("seconds") ? std::atoi(a.get("seconds").c_str()) : -1;
    auto deadline = system_clock::now() + std::chrono::seconds(seconds);

    while (!g_stop.load()) {
        if (seconds > 0 && system_clock::now() >= deadline) break;
        auto ev = sub.next(std::chrono::milliseconds(250));
        if (!ev) continue;
        if (ev->topic.rfind("proactive.", 0) != 0) continue;
        std::cout << "[" << fmtClock(ev->ts) << "] " << ev->topic
                  << "  " << truncatePayload(ev->payload.dump(), 240) << "\n";
        std::cout.flush();
    }

    sub.stop();
    if (watcher) watcher->stop();
    engine.stop();

    auto st = engine.stats();
    std::cout << "\n-- proactive stats --\n"
              << "signals_seen      = " << st.signals_seen      << "\n"
              << "suggested         = " << st.suggested         << "\n"
              << "passed            = " << st.passed            << "\n"
              << "blocked_rate      = " << st.blocked_rate      << "\n"
              << "blocked_quiet     = " << st.blocked_quiet     << "\n"
              << "blocked_muted     = " << st.blocked_muted     << "\n"
              << "blocked_silent    = " << st.blocked_silent    << "\n";
    return 0;
}

int cmdProactiveList(const Args& a) {
    arise::FeedbackDb fb(defaultFeedbackCfg());
    arise::FeedbackQuery q;
    q.limit = a.has("limit") ? std::atoi(a.get("limit").c_str()) : 25;
    if (a.has("decision")) {
        if (auto d = arise::decisionFromString(a.get("decision"))) q.decision = d;
    }
    if (a.has("tier")) {
        if (auto t = arise::tierFromString(a.get("tier"))) q.tier = t;
    }
    if (a.has("category")) q.category = a.get("category");
    auto rows = fb.list(q);
    if (rows.empty()) { std::cout << "(none)\n"; return 0; }
    for (auto& r : rows) {
        std::cout << suggestionRowToJson(r).dump() << "\n";
    }
    return 0;
}

int cmdProactiveDecide(const Args& a) {
    if (a.pos.empty()) { std::cerr << "proactive decide: ID required\n"; return 2; }
    auto id = std::atoll(a.pos[0].c_str());
    arise::Decision d = arise::Decision::Pending;
    if      (a.has("accept")) d = arise::Decision::Accepted;
    else if (a.has("reject")) d = arise::Decision::Rejected;
    else if (a.has("ignore")) d = arise::Decision::Ignored;
    else {
        std::cerr << "proactive decide: --accept | --reject | --ignore required\n";
        return 2;
    }
    arise::FeedbackDb fb(defaultFeedbackCfg());
    if (!fb.recordDecision(id, d)) {
        std::cerr << "proactive decide: failed (id missing or already terminal)\n";
        return 1;
    }
    std::cout << "ok\n"; return 0;
}

int cmdProactiveMute(const Args& a) {
    if (a.pos.empty()) { std::cerr << "proactive mute: CATEGORY required\n"; return 2; }
    // The gate's mute is in-memory; persist by inserting fake-rejected rows
    // would be a richer follow-up. For now this command is informational —
    // tells the user the streak / suggests `proactive timeout` to recompute.
    auto cat = a.pos[0];
    arise::FeedbackDb fb(defaultFeedbackCfg());
    int streak = fb.consecutiveRejects(cat);
    int rejected = fb.categoryCount(cat, arise::Decision::Rejected);
    int accepted = fb.categoryCount(cat, arise::Decision::Accepted);
    json out;
    out["category"]              = cat;
    out["consecutive_rejects"]   = streak;
    out["total_rejected"]        = rejected;
    out["total_accepted"]        = accepted;
    out["note"]                  = std::string(
        "auto-mute fires when the running engine sees consecutive_rejects "
        "≥ mute_after_rejects (default 5). Rerun `proactive start` to re-arm.");
    std::cout << out.dump(2) << "\n";
    return 0;
}

int cmdProactiveTimeout(const Args& a) {
    int sec = a.has("older-than-sec")
        ? std::atoi(a.get("older-than-sec").c_str()) : 24 * 3600;
    arise::FeedbackDb fb(defaultFeedbackCfg());
    int n = fb.timeoutPending(std::chrono::seconds(sec));
    json out; out["timed_out"] = n; out["older_than_sec"] = sec;
    std::cout << out.dump(2) << "\n";
    return 0;
}

int cmdToolsForge(const Args& a) {
    if (!a.has("description")) {
        std::cerr << "tools forge: --description required\n"; return 2;
    }
    arise::paths::ensureLayout();

    arise::SubAgent::Config llm_cfg;
    llm_cfg.role        = "forge";
    llm_cfg.format_json = true;
    llm_cfg.timeout_sec = 90;
    llm_cfg.max_predict = 1500;
    if (a.has("model"))      llm_cfg.model      = a.get("model");
    if (a.has("ollama-url")) llm_cfg.ollama_url = a.get("ollama-url");
    arise::SubAgent llm(llm_cfg);

    arise::Critic::Config cc;
    arise::Critic critic(cc);

    arise::ToolRegistry reg(defaultRegistryConfig());

    arise::ForgeTool::Config fc;
    fc.coder_llm     = &llm;
    fc.critic        = &critic;
    fc.registry      = &reg;
    fc.sandbox_root  = reg.sandboxDir();
    fc.learned_root  = reg.learnedDir();
    arise::ForgeTool forge(fc);

    json schema = json::object();
    if (a.has("args-schema")) {
        try { schema = json::parse(a.get("args-schema")); }
        catch (const std::exception& e) {
            std::cerr << "tools forge: --args-schema parse: " << e.what() << "\n";
            return 2;
        }
    }
    std::optional<json> example;
    if (a.has("example-args")) {
        try { example = json::parse(a.get("example-args")); }
        catch (const std::exception& e) {
            std::cerr << "tools forge: --example-args parse: " << e.what() << "\n";
            return 2;
        }
    }

    auto requested_id = a.has("id") ? a.get("id") : std::string{};
    auto p = forge.propose(a.get("description"), schema, example, requested_id);

    json out;
    out["id"]                = p.id;
    out["ok"]                = p.ok;
    out["budget_hit"]        = p.budget_hit;
    out["dry_run_skipped"]   = p.dry_run_skipped;
    out["draft_dir"]         = p.draft_dir;
    out["entry_path"]        = p.entry_path;
    out["interpreter"]       = p.interpreter;
    out["summary"]           = p.summary;
    if (!p.error.empty()) out["error"] = p.error;
    out["files"]             = p.files;
    out["critic"]            = {
        {"approved", p.critic_review.approved},
        {"verdict",  p.critic_review.verdict},
        {"matches",  p.critic_review.matches},
    };
    if (!p.dry_run_skipped) {
        out["dry_run"] = {
            {"ok",          p.dry_run.ok},
            {"exit_code",   p.dry_run.exit_code},
            {"timed_out",   p.dry_run.timed_out},
            {"duration_ms", p.dry_run.duration_ms},
            {"stdout",      p.dry_run.stdout_text},
            {"stderr",      p.dry_run.stderr_text},
        };
    }

    bool dry_run_only = a.has("dry-run-only");
    if (p.ok && !dry_run_only) {
        if (forge.stage(p, a.has("auto-approve"))) {
            out["staged"] = true;
            out["registered_as"] = p.id;
        } else {
            out["staged"] = false;
        }
    } else {
        out["staged"] = false;
    }

    std::cout << out.dump(2) << "\n";
    return p.ok ? 0 : 1;
}

int cmdToolsSweep(const Args& a) {
    int days = a.has("days") ? std::atoi(a.get("days").c_str()) : 90;
    arise::ToolRegistry reg(defaultRegistryConfig());
    bool dry_run = a.has("dry-run");
    int n = reg.sweepStale(days, dry_run);
    json out;
    out["days"]    = days;
    out["dry_run"] = dry_run;
    out["count"]   = n;
    std::cout << out.dump(2) << "\n";
    return 0;
}

int cmdToolsRegister(const Args& a) {
    if (!a.has("id") || !a.has("script")) {
        std::cerr << "tools register: --id and --script required\n"; return 2;
    }
    arise::ToolInfo t;
    t.id           = a.get("id");
    t.description  = a.has("description") ? a.get("description") : "";
    t.interpreter  = a.has("interpreter") ? a.get("interpreter") : "";
    t.script_path  = a.get("script");
    t.allow_network = a.has("allow-network");
    if (a.has("writable")) t.writable_paths.push_back(a.get("writable"));
    if (a.has("args-schema")) {
        try { t.args_schema = json::parse(a.get("args-schema")); }
        catch (const std::exception& e) {
            std::cerr << "tools register: --args-schema parse: " << e.what() << "\n";
            return 2;
        }
    }
    if (!arise::paths::fileExists(t.script_path)) {
        std::cerr << "tools register: --script not found on disk: "
                  << t.script_path << "\n";
        return 1;
    }
    arise::ToolRegistry reg(defaultRegistryConfig());
    if (!reg.addLearned(std::move(t))) {
        std::cerr << "tools register: failed (id reserved or save error)\n";
        return 1;
    }
    std::cout << "registered (unapproved)\n"; return 0;
}

int cmdCriticReview(const Args& a) {
    std::string content;
    if (a.has("file"))         content = slurp(a.get("file"));
    else if (a.has("content")) content = a.get("content");
    else {
        std::cerr << "critic review: --file FILE or --content TEXT required\n";
        return 2;
    }
    if (content.empty()) {
        std::cerr << "critic review: empty content\n"; return 1;
    }

    std::unique_ptr<arise::SubAgent> llm;
    arise::Critic::Config cc;
    cc.require_llm_for_approval = a.has("require-llm");
    if (a.has("llm") || cc.require_llm_for_approval) {
        arise::SubAgent::Config lc;
        lc.role        = "critic";
        lc.format_json = true;
        if (a.has("model"))      lc.model      = a.get("model");
        if (a.has("ollama-url")) lc.ollama_url = a.get("ollama-url");
        llm = std::make_unique<arise::SubAgent>(lc);
        cc.llm = llm.get();
    }

    arise::Critic critic(cc);
    auto rev = critic.reviewContent(content, a.has("kind") ? a.get("kind") : "content");
    json out;
    out["approved"] = rev.approved;
    out["verdict"]  = rev.verdict;
    out["matches"]  = rev.matches;
    out["from_llm"] = rev.from_llm;
    if (!rev.llm_raw.empty() && rev.llm_raw.size() < 800) {
        out["llm_raw"] = rev.llm_raw;
    }
    std::cout << out.dump(2) << "\n";
    return rev.approved ? 0 : 1;
}

int cmdCuratorAbsorb(const Args& a) {
    if (!a.has("transcript")) {
        std::cerr << "curator absorb: --transcript FILE required\n"; return 2;
    }
    auto blob = slurp(a.get("transcript"));
    if (blob.empty()) {
        std::cerr << "curator absorb: empty/missing transcript file\n"; return 1;
    }

    arise::SubAgent::Config llm_cfg;
    llm_cfg.role         = "curator";
    llm_cfg.format_json  = true;
    llm_cfg.max_predict  = 800;
    llm_cfg.timeout_sec  = 60;
    if (a.has("model"))      llm_cfg.model      = a.get("model");
    if (a.has("ollama-url")) llm_cfg.ollama_url = a.get("ollama-url");
    arise::SubAgent llm(llm_cfg);

    arise::MemoryCortex cortex(defaultCortexConfig());

    arise::Curator::Config cc;
    cc.cortex = &cortex;
    cc.llm    = &llm;
    if (a.has("min-confidence"))
        cc.min_confidence = std::atof(a.get("min-confidence").c_str());

    arise::Curator cur(cc);

    std::vector<arise::Curator::ExtractedFact> facts;
    int upserted = cur.absorbConversation(blob, &facts);
    if (a.has("no-upsert")) upserted = 0;

    json out;
    out["upserted"] = upserted;
    json arr = json::array();
    for (auto& f : facts) {
        arr.push_back({{"subject", f.subject}, {"predicate", f.predicate},
                       {"object", f.object},   {"confidence", f.confidence}});
    }
    out["facts"] = std::move(arr);
    auto s = cur.stats();
    out["stats"] = {
        {"llm_calls", s.llm_calls}, {"llm_failures", s.llm_failures},
        {"facts_extracted", s.facts_extracted},
        {"facts_upserted",  s.facts_upserted},
        {"facts_rejected",  s.facts_rejected},
    };
    std::cout << out.dump(2) << "\n";
    return 0;
}

// ─── identity ────────────────────────────────────────────────────────────

int cmdIdentityShow() {
    arise::IdentityStore ids(arise::paths::identityDir());
    auto r = ids.get();
    json j;
    j["name"]            = r.name;
    j["pronouns"]        = r.pronouns;
    j["persona_summary"] = r.persona_summary;
    j["baseline_mood"]   = r.baseline_mood;
    if (r.voice_profile) j["voice_profile"] = *r.voice_profile;
    else                 j["voice_profile"] = nullptr;
    j["do"]   = r.do_list;
    j["dont"] = r.dont_list;
    std::cout << j.dump(2) << "\n";
    return 0;
}

int cmdIdentitySet(const Args& a) {
    arise::IdentityStore ids(arise::paths::identityDir());
    auto r = ids.get();
    if (a.has("name"))     r.name     = a.get("name");
    if (a.has("pronouns")) r.pronouns = a.get("pronouns");
    if (a.has("persona"))  r.persona_summary = a.get("persona");
    if (a.has("add-do"))   r.do_list.push_back(a.get("add-do"));
    if (a.has("add-dont")) r.dont_list.push_back(a.get("add-dont"));
    if (a.has("baseline-mood")) r.baseline_mood = a.get("baseline-mood");
    ids.set(std::move(r), a.get("msg", "update identity via cli"));
    std::cout << "saved.\n";
    return 0;
}

// ─── import-aria ─────────────────────────────────────────────────────────

int cmdImportAria(const Args& a) {
    auto src = a.has("source") ? a.get("source")
                               : arise::paths::expandHome("~/.aria_memory.db");
    if (!arise::paths::fileExists(src)) {
        std::cerr << "import-aria: source not found: " << src << "\n";
        return 1;
    }
    bool dryRun = a.has("dry-run");

    sqlite3* db = nullptr;
    if (sqlite3_open_v2(src.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cerr << "import-aria: open " << src << " (readonly): "
                  << sqlite3_errmsg(db) << "\n";
        if (db) sqlite3_close(db);
        return 1;
    }

    arise::MemoryCortex cortex(defaultCortexConfig());

    int conv_count = 0, fact_count = 0;
    long long max_seen_aria = 0;

    // ── conversations → episodic ──
    {
        sqlite3_stmt* st = nullptr;
        const char* q = "SELECT id, role, content, timestamp FROM conversations "
                        "ORDER BY id ASC";
        if (sqlite3_prepare_v2(db, q, -1, &st, nullptr) == SQLITE_OK) {
            while (sqlite3_step(st) == SQLITE_ROW) {
                long long aria_id = sqlite3_column_int64(st, 0);
                std::string role = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
                std::string content = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
                std::string ts = reinterpret_cast<const char*>(sqlite3_column_text(st, 3));
                if (content.empty()) continue;

                if (!dryRun) {
                    arise::EpisodicEvent ev;
                    ev.kind    = "conversation_turn";
                    ev.summary = content;
                    json p;
                    p["role"]            = role;
                    p["aria_origin"]     = {{"table", "conversations"}, {"rowid", aria_id}};
                    p["timestamp_local"] = ts;
                    ev.payload_json = p.dump();
                    // Parse ARIA's "%Y-%m-%d %H:%M:%S" if possible.
                    std::tm tm{};
                    if (strptime(ts.c_str(), "%Y-%m-%d %H:%M:%S", &tm)) {
                        tm.tm_isdst = -1;
                        ev.ts = system_clock::from_time_t(std::mktime(&tm));
                    }
                    cortex.recordEvent(std::move(ev));
                }
                ++conv_count;
                if (aria_id > max_seen_aria) max_seen_aria = aria_id;
            }
            sqlite3_finalize(st);
        }
    }

    // ── user_facts → semantic ──
    {
        sqlite3_stmt* st = nullptr;
        const char* q = "SELECT key, value FROM user_facts";
        if (sqlite3_prepare_v2(db, q, -1, &st, nullptr) == SQLITE_OK) {
            while (sqlite3_step(st) == SQLITE_ROW) {
                std::string key = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
                std::string val = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
                if (key.empty() || val.empty()) continue;

                if (!dryRun) {
                    arise::SemanticFact f;
                    f.subject   = "user";
                    f.predicate = key;
                    f.object    = val;
                    f.confidence = 1.0;
                    cortex.upsertFact(std::move(f));
                }
                ++fact_count;
            }
            sqlite3_finalize(st);
        }
    }

    sqlite3_close(db);

    std::cout << (dryRun ? "[dry-run] " : "")
              << "conversations imported: " << conv_count
              << ", facts: "                 << fact_count
              << " (max aria_id=" << max_seen_aria << ")\n";
    return 0;
}

// ─── perceive / system / privacy / blackboard ────────────────────────────

std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');) {
        // trim
        std::size_t i = 0;
        while (i < tok.size() && std::isspace(static_cast<unsigned char>(tok[i]))) ++i;
        std::size_t j = tok.size();
        while (j > i && std::isspace(static_cast<unsigned char>(tok[j-1]))) --j;
        if (j > i) out.push_back(tok.substr(i, j - i));
    }
    return out;
}

std::string truncatePayload(const std::string& s, std::size_t cap = 200) {
    if (s.size() <= cap) return s;
    return s.substr(0, cap) + "…";
}

std::string fmtClock(arise::BBTimestamp ts) {
    auto t = system_clock::to_time_t(ts);
    std::tm tm_buf{}; localtime_r(&t, &tm_buf);
    char buf[16];
    std::strftime(buf, sizeof buf, "%H:%M:%S", &tm_buf);
    return buf;
}

std::atomic<bool> g_stop{false};
extern "C" void onSig(int) { g_stop.store(true); }

int cmdPerceive(const Args& a) {
    arise::Perception::Config pc;
    pc.vision_interval_ms = a.has("vision-fps")
        ? std::max(50, int(1000.0 / std::max(0.001, std::atof(a.get("vision-fps").c_str()))))
        : 1000;
    if (a.has("vision-interval-ms"))
        pc.vision_interval_ms = std::atoi(a.get("vision-interval-ms").c_str());
    if (a.has("system-interval-ms"))
        pc.system_interval_ms = std::atoi(a.get("system-interval-ms").c_str());
    if (a.has("idle-threshold-ms"))
        pc.idle_threshold_ms = std::atoi(a.get("idle-threshold-ms").c_str());
    if (a.has("threshold"))
        pc.vision_diff_threshold = std::atoi(a.get("threshold").c_str());
    if (a.has("caption-cooldown-ms"))
        pc.caption_cooldown_ms = std::atoi(a.get("caption-cooldown-ms").c_str());
    if (a.has("private"))
        pc.private_apps = splitCsv(a.get("private"));
    pc.failsafe_private_on_probe_error = a.has("strict-privacy");
    pc.episodic_writes = a.has("episodic");
    pc.emit_unchanged_frames = a.has("verbose");

    if (a.has("no-vision")) pc.vision_interval_ms = 0;
    if (a.has("no-system")) pc.system_interval_ms = 0;
    if (a.has("no-idle"))   pc.idle_threshold_ms = 0;

    arise::Blackboard bb;
    pc.bb = &bb;

    // ── optional cortex (episodic writes) ──
    std::unique_ptr<arise::MemoryCortex> cortex;
    if (pc.episodic_writes) {
        cortex = std::make_unique<arise::MemoryCortex>(defaultCortexConfig());
        pc.cortex = cortex.get();
    }

    // ── optional captioner + salience scorer ──
    std::unique_ptr<arise::VisionClient>   captioner;
    std::unique_ptr<arise::SalienceScorer> scorer;
    if (a.has("captioner")) {
        arise::VisionClient::Config vc;
        if (const char* u = std::getenv("ARIA_OLLAMA_URL")) vc.ollama_url = u;
        if (a.has("vision-model")) vc.model = a.get("vision-model");
        captioner = std::make_unique<arise::VisionClient>(vc);
        pc.captioner = captioner.get();

        if (!a.has("no-salience")) {
            arise::SalienceScorer::Config sc;
            if (const char* u = std::getenv("ARIA_OLLAMA_URL")) sc.ollama_url = u;
            if (a.has("salience-model")) sc.model = a.get("salience-model");
            scorer = std::make_unique<arise::SalienceScorer>(sc);
            pc.salience = scorer.get();
        }
    }

    // ── optional audio scene classifier + mic ──
    std::unique_ptr<arise::audio::SceneClassifier> audio_scene;
    std::unique_ptr<arise::audio::MicCapture>      mic;
    if (a.has("audio-scene")) {
        arise::audio::SceneClassifier::Config ac;
        ac.model_path  = a.has("yamnet-model")
            ? a.get("yamnet-model")
            : arise::paths::expandHome("~/.local/share/arise/models/yamnet.onnx");
        ac.labels_path = a.has("yamnet-labels")
            ? a.get("yamnet-labels")
            : arise::paths::expandHome("~/.local/share/arise/models/yamnet_class_map.csv");
        if (a.has("yamnet-min-score"))
            ac.min_score = float(std::atof(a.get("yamnet-min-score").c_str()));

        audio_scene = std::make_unique<arise::audio::SceneClassifier>();
        if (audio_scene->init(ac)) {
            mic = std::make_unique<arise::audio::MicCapture>();
            pc.audio_scene = audio_scene.get();
            pc.mic         = mic.get();
            if (a.has("audio-device")) pc.mic_device = a.get("audio-device");
        } else {
            std::cerr << "audio-scene: failed to init YAMNet from "
                      << ac.model_path << " / " << ac.labels_path
                      << " — audio loop disabled\n";
            audio_scene.reset();
        }
    }

    std::cout << "perceive: vision=" << (pc.vision_interval_ms ? "on" : "off")
              << " system=" << (pc.system_interval_ms ? "on" : "off")
              << " idle="   << (pc.idle_threshold_ms ? "on" : "off")
              << " caption=" << (pc.captioner ? "on" : "off")
              << " salience=" << (pc.salience ? "on" : "off")
              << " audio="  << (pc.audio_scene ? "on" : "off")
              << " episodic=" << (pc.episodic_writes ? "on" : "off")
              << "  (Ctrl-C to stop)\n";

    arise::Perception perc(pc);
    auto sub = bb.subscribe("");          // wildcard tail
    perc.start();

    std::signal(SIGINT,  onSig);
    std::signal(SIGTERM, onSig);

    int seconds = a.has("seconds") ? std::atoi(a.get("seconds").c_str()) : -1;
    auto deadline = system_clock::now() + std::chrono::seconds(seconds);

    while (!g_stop.load()) {
        if (seconds > 0 && system_clock::now() >= deadline) break;
        auto ev = sub.next(std::chrono::milliseconds(250));
        if (!ev) continue;
        std::cout << "[" << fmtClock(ev->ts) << "] " << ev->topic
                  << "  " << truncatePayload(ev->payload.dump()) << "\n";
        std::cout.flush();
    }

    sub.stop();
    perc.stop();

    auto st = perc.stats();
    std::cout << "\n-- stats --\n"
              << "frames_captured     = " << st.frames_captured     << "\n"
              << "frames_changed      = " << st.frames_changed      << "\n"
              << "frames_unchanged    = " << st.frames_unchanged    << "\n"
              << "frames_failed       = " << st.frames_failed       << "\n"
              << "captions_attempted  = " << st.captions_attempted  << "\n"
              << "captions_ok         = " << st.captions_ok         << "\n"
              << "captions_failed     = " << st.captions_failed     << "\n"
              << "captions_throttled  = " << st.captions_throttled  << "\n"
              << "system_samples      = " << st.system_samples      << "\n"
              << "system_deltas       = " << st.system_deltas       << "\n"
              << "privacy_holds       = " << st.privacy_holds       << "\n"
              << "idle_entries        = " << st.idle_entries        << "\n"
              << "idle_exits          = " << st.idle_exits          << "\n"
              << "audio_windows       = " << st.audio_windows       << "\n"
              << "audio_scene_changes = " << st.audio_scene_changes << "\n"
              << "events_total        = " << bb.totalPublished()    << "\n";
    return 0;
}

int cmdSystemSnapshot() {
    auto s = arise::sys::take();
    std::cout << arise::sys::toJson(s).dump(2) << "\n";
    return 0;
}

int cmdPrivacyCheck(const Args& a) {
    arise::PrivacyGate::Config c;
    c.private_apps = splitCsv(a.get("private"));
    c.failsafe_private_on_probe_error = a.has("strict");
    arise::PrivacyGate gate(c);

    auto cur_app = arise::sys::take().active_app.value_or("");
    bool blocked = gate.isPrivate();

    json out;
    out["active_app"]    = cur_app.empty() ? json(nullptr) : json(cur_app);
    out["private_apps"]  = c.private_apps;
    out["would_block"]   = blocked;
    out["matched_app"]   = gate.lastMatched();
    std::cout << out.dump(2) << "\n";
    return 0;
}

int cmdBlackboardTail() {
    std::cerr <<
"blackboard tail: ARISE phase 1 has no daemon yet, so the blackboard is\n"
"in-process only. Use `arise perceive` to see live events as perception\n"
"emits them; a daemonised tap arrives in phase 2 commit 2.\n";
    return 2;
}

// ─── self-model (phase 9): privacy scrubber, curator, adapter registry ────
//
// These wrap already-shipped library code (PrivacyFilter, TrainingCurator,
// AdapterRegistry) that the help text advertised but the dispatcher never
// reached — `arise self ...` used to print "unknown command: self".

static std::string fmtTp(std::chrono::system_clock::time_point tp) {
    if (tp.time_since_epoch().count() == 0) return "-";
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_buf{}; localtime_r(&t, &tm_buf);
    char buf[24]; std::strftime(buf, sizeof buf, "%F %T", &tm_buf);
    return buf;
}

int cmdSelfPrivacyCheck(const Args& a) {
    std::string text;
    if (a.has("file")) {
        text = slurp(a.get("file"));
        if (text.empty()) {
            std::cerr << "self privacy-check: --file empty or unreadable\n";
            return 2;
        }
    } else if (a.has("text")) {
        text = a.get("text");
    } else {
        std::cerr << "self privacy-check: --text T or --file F required\n";
        return 2;
    }

    arise::PrivacyFilter filter{arise::PrivacyFilter::Config{}};
    auto res = filter.scan(text);

    json out;
    out["passed"]    = res.passed;           // true ⇒ nothing sensitive found
    out["hit_count"] = res.hits.size();
    json hits = json::array();
    for (const auto& h : res.hits) {
        hits.push_back({
            {"reason",  arise::PrivacyFilter::reasonToString(h.reason)},
            {"start",   h.start},
            {"length",  h.length},
            {"preview", h.preview},
        });
    }
    out["hits"]     = hits;
    out["redacted"] = res.redacted_text;
    std::cout << out.dump(2) << "\n";
    // Non-zero exit when something sensitive was found, so scripts can gate.
    return res.passed ? 0 : 1;
}

int cmdSelfCurate(const Args& a) {
    arise::TrainingCurator::Config cfg;
    if (a.has("lookback-hours"))
        cfg.replay.lookback =
            std::chrono::hours(std::atoi(a.get("lookback-hours").c_str()));
    if (a.has("top-n"))
        cfg.top_n = std::atoi(a.get("top-n").c_str());
    if (a.flag.count("no-llm"))
        cfg.skip_llm_scoring = true;
    if (const char* u = std::getenv("ARIA_OLLAMA_URL"))
        cfg.scorer.ollama_url = u;

    arise::MemoryCortex cortex(defaultCortexConfig());
    arise::FeedbackDb   feedback(defaultFeedbackCfg());

    arise::TrainingCurator curator(cfg);
    arise::TrainingCurator::Stats stats;
    auto examples = curator.curate(cortex, feedback, &stats);

    // --out is an output directory (writeJsonl names the file <slug>.jsonl).
    std::string dir = a.has("out") ? a.get("out") : arise::paths::trainingDir();
    std::string path;
    if (!examples.empty())
        path = arise::TrainingCurator::writeJsonl(examples, dir);

    json out;
    out["replayed_total"]    = stats.replayed_total;
    out["scored"]            = stats.scored;
    out["dropped_low_score"] = stats.dropped_low_score;
    out["dropped_privacy"]   = stats.dropped_privacy;
    out["dropped_empty"]     = stats.dropped_empty;
    out["kept"]              = stats.kept;
    out["written"]           = path;   // empty if nothing kept
    std::cout << out.dump(2) << "\n";
    return 0;
}

int cmdSelfAdapter(const Args& a) {
    if (a.pos.empty()) {
        std::cerr << "self adapter: subcommand required "
                     "(list|register|promote|rollback|rotate)\n";
        return 2;
    }
    std::string sub = a.pos[0];

    arise::AdapterRegistry::Config cfg;
    cfg.manifest_path = arise::paths::adaptersManifestPath();
    if (a.has("keep")) cfg.keep = std::atoi(a.get("keep").c_str());
    arise::AdapterRegistry reg(cfg);
    reg.load();

    auto infoToJson = [](const arise::AdapterInfo& i) {
        return json{
            {"id",             i.id},
            {"base_model",     i.base_model},
            {"path",           i.path},
            {"eval_score",     i.eval_score},
            {"baseline_score", i.baseline_score},
            {"deployed",       i.deployed},
            {"rolled_back",    i.rolled_back},
            {"created_at",     fmtTp(i.created_at)},
            {"evaluated_at",   fmtTp(i.evaluated_at)},
            {"note",           i.note},
        };
    };

    if (sub == "list") {
        json arr = json::array();
        for (const auto& i : reg.list()) arr.push_back(infoToJson(i));
        std::cout << arr.dump(2) << "\n";
        return 0;
    }
    if (sub == "register") {
        if (!a.has("id") || !a.has("path") || !a.has("base-model")) {
            std::cerr << "self adapter register: --id, --path and "
                         "--base-model are required\n";
            return 2;
        }
        arise::AdapterInfo info;
        info.id             = a.get("id");
        info.path           = a.get("path");
        info.base_model     = a.get("base-model");
        info.eval_score     = a.has("score") ? std::atof(a.get("score").c_str()) : 0.0;
        info.note           = a.get("note");
        if (!reg.registerAdapter(info)) {
            std::cerr << "self adapter register: failed to save manifest\n";
            return 1;
        }
        std::cout << "registered " << info.id << "\n";
        return 0;
    }
    if (sub == "promote" || sub == "rollback") {
        if (a.pos.size() < 2) {
            std::cerr << "self adapter " << sub << ": ID required\n";
            return 2;
        }
        std::string id = a.pos[1];
        bool ok = (sub == "promote") ? reg.promote(id) : reg.rollback(id);
        if (!ok) { std::cerr << "self adapter " << sub << ": no such id '" << id << "'\n"; return 1; }
        std::cout << sub << " " << id << " ok\n";
        return 0;
    }
    if (sub == "rotate") {
        int archived = reg.rotate();
        std::cout << "rotated: archived " << archived
                  << " adapter(s), keeping " << cfg.keep << "\n";
        return 0;
    }
    std::cerr << "self adapter: unknown subcommand '" << sub << "'\n";
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return cmdHelp();
    std::string cmd = argv[1];
    Args a = parseArgs(argc, argv, 2);

    arise::log::init(arise::paths::logsDir() + "/arise.log");
    arise::log::setLevel(arise::log::Level::Info);
    arise::paths::ensureLayout();

    if (cmd == "help" || cmd == "-h" || cmd == "--help") return cmdHelp();
    if (cmd == "init") return cmdInit();

    if (cmd == "mem") {
        if (a.pos.empty()) { std::cerr << "mem: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a;
        sa.pos.erase(sa.pos.begin());
        if (sub == "record") return cmdMemRecord(sa);
        if (sub == "recall") return cmdMemRecall(sa);
        if (sub == "dump")   return cmdMemDump(sa);
        if (sub == "mood")   return cmdMemMood(sa);
        if (sub == "purge")  return cmdMemPurge();
        if (sub == "fact")   return cmdMemFact(sa);
        if (sub == "facts")  return cmdMemFacts(sa);
        std::cerr << "mem: unknown subcommand '" << sub << "'\n";
        return 2;
    }

    if (cmd == "identity") {
        if (a.pos.empty()) { std::cerr << "identity: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "show") return cmdIdentityShow();
        if (sub == "set")  return cmdIdentitySet(sa);
        std::cerr << "identity: unknown subcommand '" << sub << "'\n";
        return 2;
    }

    if (cmd == "import-aria") return cmdImportAria(a);

    if (cmd == "goal") return cmdGoalDispatch(a);

    if (cmd == "agents") {
        if (a.pos.empty()) { std::cerr << "agents: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "start") return cmdAgentsStart(sa);
        if (sub == "spawn") return cmdAgentsSpawn(sa);
        std::cerr << "agents: unknown subcommand '" << sub << "'\n";
        return 2;
    }
    if (cmd == "critic") {
        if (a.pos.empty()) { std::cerr << "critic: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "review") return cmdCriticReview(sa);
        std::cerr << "critic: unknown subcommand '" << sub << "'\n";
        return 2;
    }
    if (cmd == "speak") return cmdSpeak(a);
    if (cmd == "persona") {
        if (a.pos.empty()) { std::cerr << "persona: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "prompt") return cmdPersonaPrompt(sa);
        std::cerr << "persona: unknown subcommand '" << sub << "'\n";
        return 2;
    }

    if (cmd == "federation") {
        if (a.pos.empty()) { std::cerr << "federation: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "pair")    return cmdFederationPair(sa);
        if (sub == "list")    return cmdFederationList(sa);
        if (sub == "revoke")  return cmdFederationRevoke(sa);
        if (sub == "ingest")  return cmdFederationIngest(sa);
        std::cerr << "federation: unknown subcommand '" << sub << "'\n";
        return 2;
    }
    if (cmd == "proactive") {
        if (a.pos.empty()) { std::cerr << "proactive: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "start")   return cmdProactiveStart(sa);
        if (sub == "list")    return cmdProactiveList(sa);
        if (sub == "decide")  return cmdProactiveDecide(sa);
        if (sub == "mute")    return cmdProactiveMute(sa);
        if (sub == "timeout") return cmdProactiveTimeout(sa);
        std::cerr << "proactive: unknown subcommand '" << sub << "'\n";
        return 2;
    }
    if (cmd == "tools") {
        if (a.pos.empty()) { std::cerr << "tools: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "list")          return cmdToolsList(sa);
        if (sub == "show")          return cmdToolsShow(sa);
        if (sub == "run")           return cmdToolsRun(sa);
        if (sub == "approve")       return cmdToolsApprove(sa);
        if (sub == "archive")       return cmdToolsArchive(sa);
        if (sub == "remove")        return cmdToolsRemove(sa);
        if (sub == "sandbox-exec")  return cmdToolsSandboxExec(sa);
        if (sub == "register")      return cmdToolsRegister(sa);
        if (sub == "forge")         return cmdToolsForge(sa);
        if (sub == "sweep")         return cmdToolsSweep(sa);
        std::cerr << "tools: unknown subcommand '" << sub << "'\n";
        return 2;
    }
    if (cmd == "watcher") {
        if (a.pos.empty()) { std::cerr << "watcher: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "fire") return cmdWatcherFire(sa);
        std::cerr << "watcher: unknown subcommand '" << sub << "'\n";
        return 2;
    }
    if (cmd == "curator") {
        if (a.pos.empty()) { std::cerr << "curator: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "absorb") return cmdCuratorAbsorb(sa);
        std::cerr << "curator: unknown subcommand '" << sub << "'\n";
        return 2;
    }
    if (cmd == "self") {
        if (a.pos.empty()) { std::cerr << "self: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "curate")        return cmdSelfCurate(sa);
        if (sub == "privacy-check") return cmdSelfPrivacyCheck(sa);
        if (sub == "adapter")       return cmdSelfAdapter(sa);
        std::cerr << "self: unknown subcommand '" << sub << "'\n";
        return 2;
    }

    if (cmd == "perceive") return cmdPerceive(a);

    if (cmd == "system") {
        if (a.pos.empty()) { std::cerr << "system: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "snapshot") return cmdSystemSnapshot();
        std::cerr << "system: unknown subcommand '" << sub << "'\n";
        return 2;
    }

    if (cmd == "privacy") {
        if (a.pos.empty()) { std::cerr << "privacy: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        Args sa = a; sa.pos.erase(sa.pos.begin());
        if (sub == "check") return cmdPrivacyCheck(sa);
        std::cerr << "privacy: unknown subcommand '" << sub << "'\n";
        return 2;
    }

    if (cmd == "blackboard") {
        if (a.pos.empty()) { std::cerr << "blackboard: subcommand required\n"; return 2; }
        std::string sub = a.pos[0];
        if (sub == "tail") return cmdBlackboardTail();
        std::cerr << "blackboard: unknown subcommand '" << sub << "'\n";
        return 2;
    }

    std::cerr << "unknown command: " << cmd << "\n";
    return cmdHelp();
}
