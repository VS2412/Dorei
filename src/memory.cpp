#include "memory.hpp"
#include "logger.hpp"
#include "config.hpp"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Small curl write-sink used for the embedding endpoint.
static size_t embedWriteCb(void* ptr, size_t sz, size_t nmemb, std::string* out) {
    out->append(reinterpret_cast<char*>(ptr), sz * nmemb);
    return sz * nmemb;
}

// Unit-normalize in place so L2 distance on vec0 ranks by cosine.
static void l2Normalize(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += double(x) * double(x);
    double n = std::sqrt(s);
    if (n <= 1e-12) return;
    float inv = static_cast<float>(1.0 / n);
    for (float& x : v) x *= inv;
}

// Convert vec0 L2 distance (on unit vectors) to cosine similarity in [-1, 1].
//   ||a-b||^2 = 2 - 2 cos(θ)  ⇒  cos = 1 - d²/2
static double l2ToCosSim(double d) {
    double cs = 1.0 - (d * d) / 2.0;
    if (cs >  1.0) cs =  1.0;
    if (cs < -1.0) cs = -1.0;
    return cs;
}

// Parse 'YYYY-MM-DD HH:MM:SS' localtime into epoch seconds. 0 on failure.
static long long parseTimestamp(const std::string& ts) {
    std::tm tm{};
    if (strptime(ts.c_str(), "%Y-%m-%d %H:%M:%S", &tm) == nullptr) return 0;
    tm.tm_isdst = -1;
    return static_cast<long long>(std::mktime(&tm));
}

Memory::Memory(const std::string& dbPath) {
    const auto& cfg = Config::get();
    embedDim_   = cfg.embedding_dim;
    embedModel_ = cfg.embedding_model;
    ollamaUrl_  = cfg.ollama_url;

    if (sqlite3_open(dbPath.c_str(), &db_) != SQLITE_OK) {
        Logger::error("Memory: cannot open DB: " + dbPath);
        db_ = nullptr;
        return;
    }

    // Core tables
    exec(R"(
        CREATE TABLE IF NOT EXISTS conversations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now','localtime')),
            role      TEXT NOT NULL,
            content   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_facts (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS summaries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT DEFAULT (datetime('now','localtime')),
            content    TEXT NOT NULL,
            turn_start INTEGER,
            turn_end   INTEGER
        );
        CREATE TABLE IF NOT EXISTS plans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            goal       TEXT NOT NULL,
            steps_json TEXT NOT NULL,
            outcome    TEXT,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_plans_goal ON plans(goal);
    )");

    // FTS5 virtual table synced with conversations
    exec(R"(
        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
            content, content='conversations', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
            INSERT INTO conversations_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
            INSERT INTO conversations_fts(conversations_fts, rowid, content)
                VALUES('delete', old.id, old.content);
        END;
        CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
            INSERT INTO conversations_fts(conversations_fts, rowid, content)
                VALUES('delete', old.id, old.content);
            INSERT INTO conversations_fts(rowid, content) VALUES (new.id, new.content);
        END;
    )");

    // One-time backfill: rebuild FTS index from conversations if the index is empty.
    {
        sqlite3_stmt* stmt;
        int indexed = 0;
        if (sqlite3_prepare_v2(db_,
                "SELECT COUNT(*) FROM conversations_fts_docsize",
                -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) indexed = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
        }
        int total = 0;
        if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM conversations",
                               -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) total = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
        }
        if (indexed < total) {
            exec("INSERT INTO conversations_fts(conversations_fts) VALUES('rebuild');");
            Logger::info("Memory: rebuilt FTS index (" + std::to_string(total) + " rows)");
        }
    }

    // ─── Phase 8: vec extension + schema ───
    loadVecExtension(cfg.sqlite_vec_path);
    if (vecEnabled_) ensureVecSchema();

    Logger::info("Memory: loaded from " + dbPath +
                 (vecEnabled_ ? " (+sqlite-vec)" : " (vec disabled)"));

    // Background backfill: embed any conversations/summaries missing a vec row.
    if (vecEnabled_ && cfg.vec_backfill) {
        backfillThread_ = std::thread(&Memory::backfillLoop, this);
    }
}

Memory::~Memory() {
    stopping_ = true;
    if (backfillThread_.joinable()) backfillThread_.join();
    if (db_) sqlite3_close(db_);
}

void Memory::exec(const std::string& sql) {
    if (!db_) return;
    char* err = nullptr;
    sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err);
    if (err) { Logger::error("Memory SQL: " + std::string(err)); sqlite3_free(err); }
}

void Memory::loadVecExtension(const std::string& path) {
    if (!db_) return;
    if (sqlite3_enable_load_extension(db_, 1) != SQLITE_OK) {
        Logger::warn("Memory: sqlite3_enable_load_extension failed; vec disabled");
        return;
    }
    char* err = nullptr;
    int rc = sqlite3_load_extension(db_, path.c_str(), nullptr, &err);
    sqlite3_enable_load_extension(db_, 0);  // re-seal for safety
    if (rc != SQLITE_OK) {
        Logger::warn("Memory: load vec0.so (" + path + ") failed: " +
                     (err ? err : "unknown") + "  — falling back to FTS-only");
        if (err) sqlite3_free(err);
        return;
    }
    vecEnabled_ = true;
}

void Memory::ensureVecSchema() {
    const std::string vecDim = std::to_string(embedDim_);
    // vec0 virtual tables: rowid is the join key back to conversations/summaries
    exec("CREATE VIRTUAL TABLE IF NOT EXISTS conversations_vec USING vec0("
         "embedding float[" + vecDim + "]);");
    exec("CREATE VIRTUAL TABLE IF NOT EXISTS summary_vec USING vec0("
         "embedding float[" + vecDim + "]);");

    // Entity graph. Kept as regular tables; vec search on entity names can come later.
    exec(R"(
        CREATE TABLE IF NOT EXISTS entities (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            type       TEXT NOT NULL,
            first_seen TEXT DEFAULT (datetime('now','localtime')),
            last_seen  TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(name, type)
        );
        CREATE TABLE IF NOT EXISTS entity_relations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            entity1_id INTEGER NOT NULL,
            relation   TEXT NOT NULL,
            entity2_id INTEGER,
            context    TEXT,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (entity1_id) REFERENCES entities(id),
            FOREIGN KEY (entity2_id) REFERENCES entities(id)
        );
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
    )");
}

// ─── Embedding client ───
std::vector<float> Memory::embed(const std::string& text) const {
    std::vector<float> out;
    if (text.empty()) return out;

    CURL* curl = curl_easy_init();
    if (!curl) return out;

    const std::string url = ollamaUrl_ + "/api/embeddings";
    // num_gpu:0 — embeddings run per conversation save; letting nomic grab
    // VRAM can evict the pinned primary brain on the 6GB card.
    json body = { {"model", embedModel_}, {"prompt", text},
                  {"options", {{"num_gpu", 0}}} };
    std::string payload = body.dump();
    std::string response;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    payload.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, embedWriteCb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,       10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 2L);

    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK) {
        Logger::warn("Memory: embed curl error " + std::string(curl_easy_strerror(rc)));
        return out;
    }

    try {
        auto j = json::parse(response);
        if (!j.contains("embedding") || !j["embedding"].is_array()) {
            Logger::warn("Memory: embed reply missing 'embedding'");
            return out;
        }
        out.reserve(j["embedding"].size());
        for (const auto& v : j["embedding"]) out.push_back(v.get<float>());
        if (static_cast<int>(out.size()) != embedDim_) {
            Logger::warn("Memory: embed dim mismatch " + std::to_string(out.size()) +
                         " != " + std::to_string(embedDim_));
            out.clear();
            return out;
        }
        l2Normalize(out);
    } catch (const std::exception& e) {
        Logger::warn("Memory: embed parse error " + std::string(e.what()));
        out.clear();
    }
    return out;
}

void Memory::storeConversationVec(long long rowid, const std::string& content) {
    if (!vecEnabled_ || !db_) return;
    auto v = embed(content);
    if (v.empty()) return;

    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "INSERT OR REPLACE INTO conversations_vec(rowid, embedding) VALUES (?, ?)";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int64(stmt, 1, rowid);
        sqlite3_bind_blob (stmt, 2, v.data(),
                           static_cast<int>(v.size() * sizeof(float)),
                           SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

void Memory::storeSummaryVec(long long rowid, const std::string& content) {
    if (!vecEnabled_ || !db_) return;
    auto v = embed(content);
    if (v.empty()) return;

    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "INSERT OR REPLACE INTO summary_vec(rowid, embedding) VALUES (?, ?)";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int64(stmt, 1, rowid);
        sqlite3_bind_blob (stmt, 2, v.data(),
                           static_cast<int>(v.size() * sizeof(float)),
                           SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

void Memory::save(const std::string& role, const std::string& content) {
    if (!db_) return;
    long long newId = 0;
    {
        std::lock_guard<std::mutex> lk(dbMutex_);
        sqlite3_stmt* stmt;
        const char* q = "INSERT INTO conversations (role, content) VALUES (?, ?)";
        if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, role.c_str(),    -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, content.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
            newId = sqlite3_last_insert_rowid(db_);
        }
    }
    // Don't embed action-log lines — they're noisy ("open_application:{...}") and low-value
    // for semantic recall. Role-gated: user + assistant conversational turns only.
    if (vecEnabled_ && newId > 0 && (role == "user" || role == "assistant")) {
        // Skip structured-action dumps that assistant rows sometimes carry.
        bool looksStructured = content.find(":{") != std::string::npos &&
                               content.find('}')  != std::string::npos;
        if (!looksStructured) storeConversationVec(newId, content);
    }
}

void Memory::setFact(const std::string& key, const std::string& value) {
    if (!db_) return;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "INSERT OR REPLACE INTO user_facts (key, value, updated_at) "
                    "VALUES (?, ?, datetime('now','localtime'))";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key.c_str(),   -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

std::string Memory::getFact(const std::string& key) const {
    if (!db_) return "";
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "SELECT value FROM user_facts WHERE key = ?";
    std::string result;
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW)
            result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        sqlite3_finalize(stmt);
    }
    return result;
}

std::vector<MemoryEntry> Memory::getRecent(int n) const {
    std::vector<MemoryEntry> entries;
    if (!db_) return entries;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "SELECT id, role, content, timestamp FROM conversations "
                    "ORDER BY id DESC LIMIT ?";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, n);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            MemoryEntry e;
            e.id        = sqlite3_column_int(stmt, 0);
            e.role      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            e.content   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            e.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            entries.push_back(e);
        }
        sqlite3_finalize(stmt);
    }
    std::reverse(entries.begin(), entries.end());
    return entries;
}

// ─── FTS5 search ───
std::vector<MemoryEntry> Memory::search(const std::string& query, int limit) const {
    std::vector<MemoryEntry> results;
    if (!db_ || query.empty()) return results;

    // Build safe FTS5 query — quote each alphanumeric token
    std::string ftsQuery;
    std::istringstream iss(query);
    std::string token;
    while (iss >> token) {
        std::string safe;
        for (char c : token) if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') safe += c;
        if (safe.empty()) continue;
        if (!ftsQuery.empty()) ftsQuery += " OR ";
        ftsQuery += "\"" + safe + "\"";
    }
    if (ftsQuery.empty()) return results;

    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q =
        "SELECT c.id, c.role, c.content, c.timestamp "
        "FROM conversations c JOIN conversations_fts f ON c.id = f.rowid "
        "WHERE conversations_fts MATCH ? "
        "ORDER BY rank LIMIT ?";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, ftsQuery.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 2, limit);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            MemoryEntry e;
            e.id        = sqlite3_column_int(stmt, 0);
            e.role      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            e.content   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            e.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            results.push_back(e);
        }
        sqlite3_finalize(stmt);
    }
    return results;
}

// ─── Phase 8: hybrid retrieval (FTS ∪ Vec → RRF + recency rerank) ───
std::vector<SearchResult> Memory::hybridSearch(const std::string& query, int limit) const {
    std::vector<SearchResult> out;
    if (!db_ || query.empty()) return out;

    const int perSource = std::max(limit * 3, 10);

    // 1) FTS candidates (fast path; never skipped — it's the keyword baseline)
    std::vector<MemoryEntry> fts = search(query, perSource);

    // 2) Vec candidates (only if enabled and embedding succeeds)
    struct VecHit { long long rowid; double distance; };
    std::vector<VecHit> vec;
    std::vector<float> qv;
    if (vecEnabled_) {
        qv = embed(query);
        if (!qv.empty()) {
            std::lock_guard<std::mutex> lk(dbMutex_);
            sqlite3_stmt* stmt;
            const char* q =
                "SELECT rowid, distance FROM conversations_vec "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT ?";
            if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
                sqlite3_bind_blob(stmt, 1, qv.data(),
                                  static_cast<int>(qv.size() * sizeof(float)),
                                  SQLITE_TRANSIENT);
                sqlite3_bind_int (stmt, 2, perSource);
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    vec.push_back({sqlite3_column_int64(stmt, 0),
                                   sqlite3_column_double(stmt, 1)});
                }
                sqlite3_finalize(stmt);
            }
        }
    }

    // 3) Load full rows for any vec hits not already in fts list
    std::unordered_map<long long, SearchResult> merged;
    // FTS pass — rank from 1
    for (size_t i = 0; i < fts.size(); ++i) {
        SearchResult sr;
        sr.entry  = fts[i];
        sr.source = "fts";
        sr.fts    = 1.0 / (60.0 + static_cast<double>(i + 1));  // RRF
        merged[fts[i].id] = sr;
    }
    // Vec pass
    for (size_t i = 0; i < vec.size(); ++i) {
        long long id   = vec[i].rowid;
        double   sim   = l2ToCosSim(vec[i].distance);
        double   rrf_v = 1.0 / (60.0 + static_cast<double>(i + 1));

        auto it = merged.find(id);
        if (it != merged.end()) {
            it->second.vecSim  = sim;
            it->second.score  += rrf_v;   // RRF from vec side
            it->second.source  = "both";
        } else {
            // Fetch the row so we can show it.
            std::lock_guard<std::mutex> lk(dbMutex_);
            sqlite3_stmt* stmt;
            if (sqlite3_prepare_v2(db_,
                    "SELECT id, role, content, timestamp FROM conversations WHERE id = ?",
                    -1, &stmt, nullptr) == SQLITE_OK) {
                sqlite3_bind_int64(stmt, 1, id);
                if (sqlite3_step(stmt) == SQLITE_ROW) {
                    SearchResult sr;
                    sr.entry.id        = sqlite3_column_int(stmt, 0);
                    sr.entry.role      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                    sr.entry.content   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                    sr.entry.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
                    sr.vecSim = sim;
                    sr.score  = rrf_v;
                    sr.source = "vec";
                    merged[id] = sr;
                }
                sqlite3_finalize(stmt);
            }
        }
    }

    // 4) Finalize scores = RRF(fts + vec) + recency decay
    const long long now = static_cast<long long>(std::time(nullptr));
    for (auto& [id, sr] : merged) {
        sr.score += sr.fts;  // fts RRF accumulated separately above
        long long ts = parseTimestamp(sr.entry.timestamp);
        double ageDays = ts > 0 ? double(now - ts) / 86400.0 : 365.0;
        double recency = std::exp(-ageDays / 30.0);  // half-life ~21 days
        // Plan weighting: 0.6 * similarity + 0.4 * recency. RRF already captures
        // relevance, so treat score as the relevance axis and blend with recency.
        sr.score = 0.6 * sr.score + 0.4 * recency * 0.03;
        //                                    ^^ RRF max is 1/(60+1) ≈ 0.016 per source;
        // scaling recency to ~0.03 keeps recency a meaningful tiebreaker without
        // dominating when there's a strong semantic hit.
        out.push_back(sr);
    }

    std::sort(out.begin(), out.end(),
              [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
    if (static_cast<int>(out.size()) > limit) out.resize(limit);
    return out;
}

// ─── Phase 8: entity graph ───
int Memory::upsertEntity(const std::string& name, const std::string& type) {
    if (!db_ || name.empty() || type.empty()) return 0;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* upsert =
        "INSERT INTO entities(name, type) VALUES(?, ?) "
        "ON CONFLICT(name, type) DO UPDATE SET last_seen = datetime('now','localtime') "
        "RETURNING id";
    int id = 0;
    if (sqlite3_prepare_v2(db_, upsert, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, type.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) id = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
    }
    return id;
}

void Memory::addEntityRelation(int entity1Id, const std::string& relation,
                               int entity2Id, const std::string& context) {
    if (!db_ || entity1Id <= 0 || relation.empty()) return;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q =
        "INSERT INTO entity_relations(entity1_id, relation, entity2_id, context) "
        "VALUES(?, ?, ?, ?)";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int (stmt, 1, entity1Id);
        sqlite3_bind_text(stmt, 2, relation.c_str(), -1, SQLITE_TRANSIENT);
        if (entity2Id > 0) sqlite3_bind_int (stmt, 3, entity2Id);
        else               sqlite3_bind_null(stmt, 3);
        sqlite3_bind_text(stmt, 4, context.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

std::vector<EntityRow> Memory::listEntities(const std::string& type, int limit) const {
    std::vector<EntityRow> out;
    if (!db_) return out;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const std::string q = type.empty()
        ? "SELECT id, name, type, first_seen, last_seen FROM entities "
          "ORDER BY last_seen DESC LIMIT ?"
        : "SELECT id, name, type, first_seen, last_seen FROM entities "
          "WHERE type = ? ORDER BY last_seen DESC LIMIT ?";
    if (sqlite3_prepare_v2(db_, q.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        if (type.empty()) {
            sqlite3_bind_int(stmt, 1, limit);
        } else {
            sqlite3_bind_text(stmt, 1, type.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int (stmt, 2, limit);
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            EntityRow r;
            r.id        = sqlite3_column_int(stmt, 0);
            r.name      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            r.type      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            r.firstSeen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            r.lastSeen  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            out.push_back(r);
        }
        sqlite3_finalize(stmt);
    }
    return out;
}

// ─── Phase 8: backfill ───
// Embeds any conversation/summary rows that don't yet have a vec row.
// Runs as a detached thread off the Memory ctor; polite (sleeps between rows)
// to avoid starving foreground latency on the single Ollama worker.
void Memory::backfillLoop() {
    if (!db_ || !vecEnabled_) return;
    // Give the daemon a moment to finish startup before hammering /api/embeddings.
    std::this_thread::sleep_for(std::chrono::seconds(3));
    if (stopping_) return;

    // Gather target IDs in one shot so we don't hold the mutex during network calls.
    std::vector<std::pair<long long, std::string>> todo;
    {
        std::lock_guard<std::mutex> lk(dbMutex_);
        sqlite3_stmt* stmt;
        const char* q =
            "SELECT c.id, c.content FROM conversations c "
            "LEFT JOIN conversations_vec v ON v.rowid = c.id "
            "WHERE v.rowid IS NULL AND c.role IN ('user','assistant') "
            "ORDER BY c.id ASC";
        if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                long long id = sqlite3_column_int64(stmt, 0);
                std::string c = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                if (!c.empty() && c.find(":{") == std::string::npos)
                    todo.emplace_back(id, std::move(c));
            }
            sqlite3_finalize(stmt);
        }
    }

    if (todo.empty()) return;
    Logger::info("Memory: backfilling " + std::to_string(todo.size()) + " conversation embeddings");

    int done = 0;
    for (auto& [id, content] : todo) {
        if (stopping_) break;
        storeConversationVec(id, content);
        ++done;
        if ((done % 50) == 0)
            Logger::info("Memory: backfill " + std::to_string(done) + "/" +
                         std::to_string(todo.size()));
        // Pace: nomic-embed-text is quick but we don't want to starve live queries.
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }

    // Summaries
    std::vector<std::pair<long long, std::string>> sumsTodo;
    {
        std::lock_guard<std::mutex> lk(dbMutex_);
        sqlite3_stmt* stmt;
        const char* q =
            "SELECT s.id, s.content FROM summaries s "
            "LEFT JOIN summary_vec v ON v.rowid = s.id "
            "WHERE v.rowid IS NULL ORDER BY s.id ASC";
        if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                long long id = sqlite3_column_int64(stmt, 0);
                std::string c = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                sumsTodo.emplace_back(id, std::move(c));
            }
            sqlite3_finalize(stmt);
        }
    }
    for (auto& [id, content] : sumsTodo) {
        if (stopping_) break;
        storeSummaryVec(id, content);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }

    if (!stopping_)
        Logger::info("Memory: backfill complete (" + std::to_string(done) +
                     " conversations, " + std::to_string(sumsTodo.size()) + " summaries)");
}

// ─── Summaries ───
int Memory::lastConversationId() const {
    if (!db_) return 0;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    int maxId = 0;
    if (sqlite3_prepare_v2(db_, "SELECT COALESCE(MAX(id),0) FROM conversations",
                           -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) maxId = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
    }
    return maxId;
}

int Memory::lastSummarizedId() const {
    if (!db_) return 0;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    int lastEnd = 0;
    if (sqlite3_prepare_v2(db_, "SELECT COALESCE(MAX(turn_end),0) FROM summaries",
                           -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) lastEnd = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
    }
    return lastEnd;
}

void Memory::saveSummary(const std::string& content, int turnStart, int turnEnd) {
    if (!db_) return;
    long long newId = 0;
    {
        std::lock_guard<std::mutex> lk(dbMutex_);
        sqlite3_stmt* stmt;
        const char* q = "INSERT INTO summaries (content, turn_start, turn_end) VALUES (?, ?, ?)";
        if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, content.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int (stmt, 2, turnStart);
            sqlite3_bind_int (stmt, 3, turnEnd);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
            newId = sqlite3_last_insert_rowid(db_);
        }
    }
    if (vecEnabled_ && newId > 0) storeSummaryVec(newId, content);
}

std::vector<SummaryEntry> Memory::getSummaries(int n) const {
    std::vector<SummaryEntry> results;
    if (!db_) return results;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "SELECT content, timestamp, turn_start, turn_end FROM summaries "
                    "ORDER BY id DESC LIMIT ?";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, n);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            SummaryEntry e;
            e.content   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            e.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            e.turnStart = sqlite3_column_int(stmt, 2);
            e.turnEnd   = sqlite3_column_int(stmt, 3);
            results.push_back(e);
        }
        sqlite3_finalize(stmt);
    }
    std::reverse(results.begin(), results.end()); // oldest first
    return results;
}

std::vector<MemoryEntry> Memory::getRange(int minIdExclusive, int maxIdInclusive) const {
    std::vector<MemoryEntry> entries;
    if (!db_) return entries;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "SELECT id, role, content, timestamp FROM conversations "
                    "WHERE id > ? AND id <= ? ORDER BY id ASC";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, minIdExclusive);
        sqlite3_bind_int(stmt, 2, maxIdInclusive);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            MemoryEntry e;
            e.id        = sqlite3_column_int(stmt, 0);
            e.role      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            e.content   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            e.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            entries.push_back(e);
        }
        sqlite3_finalize(stmt);
    }
    return entries;
}

std::string Memory::getSummary() const {
    if (!db_) return "";
    std::ostringstream out;

    // Facts about the user — persistent memory
    {
        std::lock_guard<std::mutex> lk(dbMutex_);
        sqlite3_stmt* stmt;
        const char* q1 = "SELECT key, value FROM user_facts ORDER BY updated_at DESC LIMIT 15";
        if (sqlite3_prepare_v2(db_, q1, -1, &stmt, nullptr) == SQLITE_OK) {
            bool any = false;
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                if (!any) { out << "Known about user: "; any = true; }
                out << sqlite3_column_text(stmt, 0) << "="
                    << sqlite3_column_text(stmt, 1) << "; ";
            }
            sqlite3_finalize(stmt);
            if (any) out << "\n";
        }
    }

    // Older conversation summaries — long-term memory window
    auto summaries = getSummaries(3);
    if (!summaries.empty()) {
        out << "Earlier conversations:\n";
        for (const auto& s : summaries)
            out << "[" << s.timestamp << "] " << s.content << "\n";
    }

    // Deliberately NOT dumping recent raw turns here — the LLM maintains its
    // own rolling message history inside the chat request, and re-injecting
    // the DB copy was leaking pre-Phase-7 persona breakage ("I'm Qwen", "I
    // don't have access") back into the system prompt and causing the model
    // to mimic its own older bad output. Facts + summaries are enough.

    return out.str();
}

// ─── Phase 11: agentic plans ───

int Memory::savePlan(const std::string& goal, const std::string& stepsJson,
                      const std::string& outcome) {
    if (!db_) return 0;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "INSERT INTO plans (goal, steps_json, outcome) VALUES (?, ?, ?)";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) != SQLITE_OK) {
        Logger::error("Memory: savePlan prepare failed");
        return 0;
    }
    sqlite3_bind_text(stmt, 1, goal.c_str(),      -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, stepsJson.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, outcome.c_str(),   -1, SQLITE_TRANSIENT);
    int id = 0;
    if (sqlite3_step(stmt) == SQLITE_DONE)
        id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);
    return id;
}

void Memory::updatePlanOutcome(int planId, const std::string& outcome,
                                const std::string& stepsJson) {
    if (!db_ || planId <= 0) return;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = stepsJson.empty()
        ? "UPDATE plans SET outcome = ? WHERE id = ?"
        : "UPDATE plans SET outcome = ?, steps_json = ? WHERE id = ?";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) != SQLITE_OK) return;
    sqlite3_bind_text(stmt, 1, outcome.c_str(), -1, SQLITE_TRANSIENT);
    if (stepsJson.empty()) {
        sqlite3_bind_int(stmt, 2, planId);
    } else {
        sqlite3_bind_text(stmt, 2, stepsJson.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 3, planId);
    }
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::vector<PlanRow> Memory::recentPlans(int n) const {
    std::vector<PlanRow> out;
    if (!db_) return out;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "SELECT id, goal, steps_json, outcome, created_at "
                    "FROM plans ORDER BY id DESC LIMIT ?";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) != SQLITE_OK) return out;
    sqlite3_bind_int(stmt, 1, n);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        PlanRow p;
        p.id        = sqlite3_column_int(stmt, 0);
        p.goal      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        p.stepsJson = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        if (sqlite3_column_type(stmt, 3) != SQLITE_NULL)
            p.outcome = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        p.createdAt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        out.push_back(std::move(p));
    }
    sqlite3_finalize(stmt);
    return out;
}

std::vector<PlanRow> Memory::findPlansByGoal(const std::string& goalLike, int n) const {
    std::vector<PlanRow> out;
    if (!db_ || goalLike.empty()) return out;
    std::lock_guard<std::mutex> lk(dbMutex_);
    sqlite3_stmt* stmt;
    const char* q = "SELECT id, goal, steps_json, outcome, created_at "
                    "FROM plans WHERE goal LIKE ? ORDER BY id DESC LIMIT ?";
    if (sqlite3_prepare_v2(db_, q, -1, &stmt, nullptr) != SQLITE_OK) return out;
    std::string pattern = "%" + goalLike + "%";
    sqlite3_bind_text(stmt, 1, pattern.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (stmt, 2, n);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        PlanRow p;
        p.id        = sqlite3_column_int(stmt, 0);
        p.goal      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        p.stepsJson = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        if (sqlite3_column_type(stmt, 3) != SQLITE_NULL)
            p.outcome = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        p.createdAt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        out.push_back(std::move(p));
    }
    sqlite3_finalize(stmt);
    return out;
}
