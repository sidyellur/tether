"""store.py - the memory store. Owns ALL SQL.

One table (`memories`) plus an external-content FTS5 index kept in sync by
triggers. The four verbs and the boot index are the only public surface;
nothing outside this module speaks SQL.
"""

import json
import re
import struct
import warnings
from datetime import datetime, timezone

from . import graph
from .graph import Graph

VALID_TYPES = ("user", "feedback", "project", "reference")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    type       TEXT NOT NULL CHECK (type IN ('user','feedback','project','reference')),
    title      TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    body       TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '',
    links      TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    device_id  TEXT NOT NULL DEFAULT ''
);
-- idx_memories_dedup is created/upgraded in _ensure_dedup_unique_index(), not
-- here: it needs to become a partial UNIQUE index (#41) but must degrade
-- gracefully on a live DB that already has duplicate current rows, which
-- plain executescript() can't express.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(title, body, tags, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, title, body, tags)
        VALUES (new.id, new.title, new.body, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, body, tags)
        VALUES ('delete', old.id, old.title, old.body, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, body, tags)
        VALUES ('delete', old.id, old.title, old.body, old.tags);
    INSERT INTO memories_fts(rowid, title, body, tags)
        VALUES (new.id, new.title, new.body, new.tags);
END;
"""

_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(title: str) -> str:
    """Normalize a title for the dedup probe: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", title.strip().lower())


def _tags_to_str(tags) -> str:
    if not tags:
        return ""
    if isinstance(tags, str):
        parts = tags.split(",")
    else:
        parts = list(tags)
    return ",".join(t.strip() for t in parts if t and t.strip())


def _parse_tags(tags) -> list:
    """Split a tag filter (comma-separated string or iterable) into normalized
    tokens. [] when there is nothing to filter on."""
    if not tags:
        return []
    if isinstance(tags, str):
        parts = tags.split(",")
    else:
        parts = list(tags)
    return [t.strip() for t in parts if t and t.strip()]


def _tags_match(stored_tags: str, required: list) -> bool:
    """Exact membership check: every tag in `required` must be one of the
    stored tags, split on commas - never a substring/LIKE match, so
    "proj:tether" never matches "proj:tether2"."""
    if not required:
        return True
    stored = {t.strip() for t in stored_tags.split(",") if t.strip()}
    return all(t in stored for t in required)


def _dedupe_links(links) -> list:
    """Order-preserving de-dupe of a links list, tolerant of None (#47)."""
    seen = set()
    out = []
    for x in (links or []):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _fts_query(raw: str):
    """Turn a free-text query into a safe FTS5 MATCH string.

    Each whitespace token is escaped and double-quoted so punctuation in the
    query can never produce an FTS5 syntax error (degrade, never throw).
    Returns None when the query has no usable tokens.
    """
    toks = [t for t in re.split(r"\s+", raw.strip()) if t]
    if not toks:
        return None
    return " ".join('"' + t.replace('"', '""') + '"' for t in toks)


def _embed_text(title: str, body: str) -> str:
    """The text an embedding represents: title and body together."""
    return f"{title}\n{body}"


def _pack(vec) -> bytes:
    """Serialize a float vector as little-endian float32 bytes."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> tuple:
    """Inverse of _pack: bytes -> tuple of floats."""
    return struct.unpack(f"<{len(blob) // 4}f", blob)


_RECENCY_WEIGHT = 0.25
_PRIMING_WEIGHT = 0.25
# Associative ranking protects the head and re-ranks the tail. _PROTECT_HEAD
# locks that many top v0.2 hits in place (so a direct hit can't be buried by
# spreading - the #25 regression), and everything below is re-ranked by spread
# activation to surface connected-but-weakly-matched memories. Larger == more
# protection, less upside. 8 predates the #15 seed floor, which bounds the seed
# set to genuinely-relevant hits (so `head` is now "protect the real seeds", not
# "protect the top 8 of the whole store"); the default likely wants re-tuning
# downward now that spread governs real slots -- tracked as a bench follow-up.
_PROTECT_HEAD = 8
# Minimum cosine a vector hit needs to seed an associative walk (#15). Without
# it, _vector_ids returns the whole store as near-tied seeds, so the associative
# tier can never label an edge-reached memory (everything is already a seed) and
# protect-head guards a meaningless order. 0.35 sits in the clean gap the bench
# corpus shows between entry targets (>=0.49) and distant golds (<=0.29 -- these
# should be reached by edges, not seeded); it leaves margin for real paraphrases.
_SEED_FLOOR = 0.35


def _rrf_scores(ranked_lists, k=60):
    """Reciprocal Rank Fusion as a {id: score} map (deterministic)."""
    scores = {}
    for lst in ranked_lists:
        for rank, mid in enumerate(lst):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _rrf_fuse(ranked_lists, k=60):
    """Reciprocal Rank Fusion: merge several ranked id-lists into one order
    without needing comparable scores across lists. Deterministic - ties break
    by ascending id."""
    return [mid for mid, _ in sorted(
        _rrf_scores(ranked_lists, k).items(), key=lambda kv: (-kv[1], kv[0]))]


def _age_days(iso: str, now_iso: str) -> float:
    a = datetime.fromisoformat(iso)
    b = datetime.fromisoformat(now_iso)
    return max(0.0, (b - a).total_seconds() / 86400.0)


def _decay_factor(age_days: float, half_life_days: float) -> float:
    return 0.5 ** (age_days / half_life_days)


class Store:
    def __init__(self, conn, device_id: str, sync_now, embedder=None,
                 author="", consolidate=False, dedup_threshold=0.92,
                 decay_half_life_days=None, assoc=False, recall_budget=8,
                 protect_head=_PROTECT_HEAD, seed_floor=_SEED_FLOOR,
                 crystallize=False,
                 boot_index_cap=50, forget=False, forget_age_days=90,
                 forget_interval=20, forget_max_per_sweep=10):
        self._conn = conn
        self._device_id = device_id
        self._sync_now = sync_now
        self._embedder = embedder
        self._author = author
        self._consolidate = consolidate
        self._dedup_threshold = dedup_threshold
        self._decay_half_life_days = decay_half_life_days
        self._recall_budget = recall_budget
        self._protect_head = protect_head
        self._seed_floor = seed_floor
        self._crystallize = crystallize
        self._cryst_sig = None
        self._cryst_cache = []
        self._graph = Graph(conn, enabled=assoc)
        # Set for real by migrate()'s _ensure_dedup_unique_index(); defaults
        # to the conservative (locking) path if migrate() is somehow skipped.
        self._has_unique_dedup_index = False
        self._boot_index_cap = boot_index_cap
        self._forget = forget
        self._forget_age_days = forget_age_days
        self._forget_interval = forget_interval
        self._forget_max_per_sweep = forget_max_per_sweep

    def migrate(self) -> None:
        fts_existed = self._table_exists("memories_fts")
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_META_SCHEMA)
        self._ensure_embedding_column()
        if not fts_existed:
            # FTS5 external-content tables don't auto-index pre-existing rows
            # in the content table; rebuild so a DB that predates the FTS5
            # table (or embedding column) isn't left with a stale/empty index.
            # Must happen before any UPDATE touches `memories` (e.g. the
            # valid_from backfill below) - an UPDATE trigger firing against a
            # not-yet-rebuilt FTS5 shadow index corrupts it.
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        self._ensure_consolidation_columns()
        self._ensure_dedup_unique_index()
        self._graph.migrate()
        if self._graph.enabled:
            pairs = []
            for (rid, links_json) in self._conn.execute(
                    "SELECT id, links FROM memories").fetchall():
                for other in json.loads(links_json or "[]"):
                    pairs.append((rid, other))
            self._graph.backfill_explicit(pairs)
        self._conn.commit()

    def _table_exists(self, name) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,)).fetchone() is not None

    def _ensure_embedding_column(self) -> None:
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(memories)").fetchall()}
        if "embedding" not in cols:
            self._conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")

    def _ensure_consolidation_columns(self) -> None:
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(memories)").fetchall()}
        adds = [
            ("author", "ALTER TABLE memories ADD COLUMN author TEXT NOT NULL DEFAULT ''"),
            ("valid_from", "ALTER TABLE memories ADD COLUMN valid_from TEXT"),
            ("valid_to", "ALTER TABLE memories ADD COLUMN valid_to TEXT"),
            ("superseded_by", "ALTER TABLE memories ADD COLUMN superseded_by INTEGER"),
        ]
        for name, ddl in adds:
            if name not in cols:
                self._conn.execute(ddl)
        # heal any row missing valid_from (legacy or a NULL'd column)
        self._conn.execute(
            "UPDATE memories SET valid_from = created_at WHERE valid_from IS NULL")

    def _ensure_dedup_unique_index(self) -> None:
        """Make idx_memories_dedup a partial UNIQUE index on
        (type, title_norm) WHERE valid_to IS NULL, so remember()'s upsert can
        rely on `INSERT ... ON CONFLICT` for true cross-connection atomicity
        (#41) instead of a racy probe-SELECT-then-INSERT.

        Sets self._has_unique_dedup_index so remember() knows which upsert
        strategy is safe to use. Must run after _ensure_consolidation_columns
        (needs the valid_to column) and requires only executescript-level DDL,
        so it can't live in _SCHEMA: it degrades instead of crashing when a
        live DB already has duplicate CURRENT rows for some (type,
        title_norm) - only reachable via the very race this fix closes - by
        warning and keeping the plain index, so remember() falls back to a
        BEGIN IMMEDIATE-guarded probe instead of the DB constraint.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='idx_memories_dedup'").fetchone()
        if row is not None and row[0] and "UNIQUE" not in row[0].upper():
            # Upgrading from a pre-#41 plain index: drop it so the CREATE
            # UNIQUE INDEX below (which is IF NOT EXISTS, so a same-named
            # index would otherwise be left alone) actually takes effect.
            self._conn.execute("DROP INDEX idx_memories_dedup")
            row = None
        if row is not None:
            self._has_unique_dedup_index = True
            return
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_dedup "
                "ON memories(type, title_norm) WHERE valid_to IS NULL")
            self._has_unique_dedup_index = True
        except Exception:
            warnings.warn(
                "tether: duplicate current memories already exist for some "
                "(type, title) - could not create the unique dedup index "
                "(#41). remember() will fall back to a locking upsert until "
                "the duplicates are resolved by hand.",
                RuntimeWarning, stacklevel=2)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_dedup "
                "ON memories(type, title_norm)")
            self._has_unique_dedup_index = False

    def _meta_get(self, key):
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _meta_set(self, key, value):
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))

    def _embed_or_none(self, title, body):
        """Embed title+body, or None if there is no embedder / it fails.
        Embedding must never break a write, so any error degrades to None."""
        if self._embedder is None:
            return None
        try:
            return _pack(self._embedder.embed(_embed_text(title, body)))
        except Exception:
            return None

    def backfill_embeddings(self, batch=200) -> int:
        """Embed rows lacking a vector. If the active model/dims differ from
        what produced the stored vectors, clear them all first so the store
        never mixes incompatible embeddings. Returns rows embedded. No-op
        (returns 0) without an embedder; never raises."""
        if self._embedder is None:
            return 0
        try:
            prev_model = self._meta_get("embedding_model")
            prev_dims = self._meta_get("embedding_dims")
            if (prev_model != self._embedder.name
                    or prev_dims != str(self._embedder.dims)):
                self._conn.execute("UPDATE memories SET embedding=NULL")
                self._meta_set("embedding_model", self._embedder.name)
                self._meta_set("embedding_dims", self._embedder.dims)
                self._conn.commit()
            done = 0
            while True:
                rows = self._conn.execute(
                    "SELECT id, title, body FROM memories "
                    "WHERE embedding IS NULL LIMIT ?", (batch,)).fetchall()
                if not rows:
                    break
                for mid, title, body in rows:
                    blob = self._embed_or_none(title, body)
                    if blob is None:
                        # embedder broke mid-run: stop, leave the rest for later
                        self._conn.commit()
                        return done
                    self._conn.execute(
                        "UPDATE memories SET embedding=? WHERE id=?", (blob, mid))
                    done += 1
                self._conn.commit()
            self._graph.backfill_semantic()
            return done
        except Exception:
            return 0

    def remember(self, type, title, body, tags=None, links=None,
                 crystallizes=None) -> dict:
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got {type!r}")
        now = _now()
        norm = _norm(title)
        tags_s = _tags_to_str(tags)
        incoming_links = _dedupe_links(links)
        emb = self._embed_or_none(title, body)

        if self._has_unique_dedup_index:
            # The partial unique index (#41) makes the upsert itself the
            # source of truth: a probe SELECT here is only an optimization
            # (to decide "created" vs "updated" and whether to run the
            # consolidate check), never a correctness requirement, because
            # the INSERT below uses ON CONFLICT to resolve atomically even
            # if another connection created/removed the row in between.
            existing = self._conn.execute(
                "SELECT id, links FROM memories "
                "WHERE type=? AND title_norm=? AND valid_to IS NULL",
                (type, norm)).fetchone()
            mid, action = self._upsert_via_conflict(
                type, title, norm, body, tags_s, incoming_links, now, emb, existing)
        else:
            # No DB-level guarantee available - a live DB already had
            # duplicate current rows and blocked the unique index at
            # migrate() time (#41). Fall back to bracketing the probe SELECT
            # and the INSERT/UPDATE in a single BEGIN IMMEDIATE transaction
            # so no other writer can interleave between them.
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute(
                "SELECT id, links FROM memories "
                "WHERE type=? AND title_norm=? AND valid_to IS NULL",
                (type, norm)).fetchone()
            mid, action = self._upsert_locked(
                type, title, norm, body, tags_s, incoming_links, now, emb, existing)

        self._graph.on_remember(mid, emb)
        if self._crystallize and crystallizes:
            self._graph.on_crystallize(mid, crystallizes)
        self._conn.commit()
        self._sync_now()
        self._maybe_forget()
        return {"id": mid, "action": action}

    def _merge_links(self, existing_links_json, incoming_links) -> str:
        """#47: union the incoming links with the row's current links rather
        than replacing - re-remembering a memory without re-passing `links`
        must never wipe links recorded by an earlier call (or by link())."""
        existing_links = json.loads(existing_links_json or "[]")
        return json.dumps(_dedupe_links(existing_links + incoming_links))

    def _upsert_via_conflict(self, type, title, norm, body, tags_s,
                              incoming_links, now, emb, existing) -> tuple:
        """Atomic upsert via a partial-unique-index ON CONFLICT target (#41).
        Safe even if `existing` is stale (raced by a concurrent writer since
        the probe SELECT): the DB resolves the real conflict, not us."""
        if existing is not None:
            links_s = self._merge_links(existing[1], incoming_links)
        else:
            links_s = json.dumps(incoming_links)
        superseded = (self._find_near_duplicate(type, emb)
                      if (existing is None and self._consolidate) else None)
        self._conn.execute(
            "INSERT INTO memories(type, title, title_norm, body, tags, links, "
            "created_at, updated_at, device_id, embedding, author, valid_from) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(type, title_norm) WHERE valid_to IS NULL DO UPDATE SET "
            "title=excluded.title, body=excluded.body, tags=excluded.tags, "
            "links=?, updated_at=excluded.updated_at, "
            "device_id=excluded.device_id, author=excluded.author, "
            "embedding=excluded.embedding",
            (type, title, norm, body, tags_s, links_s, now, now,
             self._device_id, emb, self._author, now, links_s))
        # lastrowid isn't reliable across the ON CONFLICT DO UPDATE branch (it
        # only advances on an actual insert), so re-resolve the current row
        # to get both its id and whether this call created it.
        mid, created_at = self._conn.execute(
            "SELECT id, created_at FROM memories "
            "WHERE type=? AND title_norm=? AND valid_to IS NULL",
            (type, norm)).fetchone()
        action = "created" if created_at == now else "updated"
        if action == "created" and superseded is not None:
            self._conn.execute(
                "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                (now, mid, superseded))
            self._graph.unprime(superseded)
            action = "consolidated"
        return mid, action

    def _upsert_locked(self, type, title, norm, body, tags_s,
                        incoming_links, now, emb, existing) -> tuple:
        """Upsert under an already-open BEGIN IMMEDIATE transaction: `existing`
        was probed inside that same transaction, so no other writer could
        have interleaved since - a plain branch on it is safe here."""
        if existing is not None:
            mid, existing_links_json = existing
            links_s = self._merge_links(existing_links_json, incoming_links)
            self._conn.execute(
                "UPDATE memories SET title=?, body=?, tags=?, links=?, updated_at=?, "
                "device_id=?, author=?, embedding=? WHERE id=?",
                (title, body, tags_s, links_s, now, self._device_id,
                 self._author, emb, mid))
            return mid, "updated"
        links_s = json.dumps(incoming_links)
        superseded = self._find_near_duplicate(type, emb) if self._consolidate else None
        cur = self._conn.execute(
            "INSERT INTO memories(type, title, title_norm, body, tags, links, "
            "created_at, updated_at, device_id, embedding, author, valid_from) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (type, title, norm, body, tags_s, links_s, now, now,
             self._device_id, emb, self._author, now))
        mid = cur.lastrowid
        action = "created"
        if superseded is not None:
            self._conn.execute(
                "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                (now, mid, superseded))
            self._graph.unprime(superseded)
            action = "consolidated"
        return mid, action

    def _maybe_forget(self) -> None:
        """Amortized trigger: every forget_interval writes, run one bounded
        sweep. No-op (and no meta writes) when forgetting is disabled."""
        if not self._forget:
            return
        try:
            n = int(self._meta_get("forget_counter") or 0) + 1
            if n >= self._forget_interval:
                self._meta_set("forget_counter", 0)
                self._conn.commit()
                self._run_forgetting_sweep()
            else:
                self._meta_set("forget_counter", n)
                self._conn.commit()
        except Exception:
            return

    def _run_forgetting_sweep(self) -> int:
        """Soft-archive old + behaviorally-isolated memories (opt-in, bounded,
        reversible). Returns the number archived. Never raises."""
        if not self._forget:
            return 0
        try:
            deg = self._graph.degree_map()          # behavioral, {} if unavailable
            if not any(v > 0 for v in deg.values()):
                return 0                            # no live behavioral graph -> refuse
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE valid_to IS NULL").fetchone()[0]
            if count < 2 * self._boot_index_cap:
                return 0                            # store-size floor
            now = _now()
            rows = self._conn.execute(
                "SELECT id, updated_at FROM memories WHERE valid_to IS NULL "
                "ORDER BY updated_at ASC, id ASC").fetchall()   # oldest first
            archived = 0
            for mid, updated_at in rows:
                if archived >= self._forget_max_per_sweep:
                    break
                if _age_days(updated_at, now) <= self._forget_age_days:
                    break                           # rest are younger (ordered) -> done
                if deg.get(mid, 0.0) > 0:
                    continue                        # behaviorally connected -> keep
                self._conn.execute(
                    "UPDATE memories SET valid_to=? WHERE id=?", (now, mid))
                self._graph.unprime(mid)
                archived += 1
            if archived:
                self._conn.commit()
            return archived
        except Exception:
            return 0

    def _find_near_duplicate(self, type, emb):
        """Id of the most-similar CURRENT same-type memory whose cosine
        similarity to `emb` meets the dedup threshold, or None. Degrades to
        None (no consolidation) whenever semantic support is unavailable."""
        if emb is None or self._embedder is None:
            return None
        try:
            import numpy as np

            q = np.frombuffer(emb, dtype="<f4")
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories "
                "WHERE type=? AND valid_to IS NULL AND embedding IS NOT NULL",
                (type,)).fetchall()
            best_id, best_sim = None, -1.0
            for mid, blob in rows:
                sim = float(np.frombuffer(blob, dtype="<f4") @ q)  # both unit-norm
                if sim > best_sim:
                    best_id, best_sim = mid, sim
            return best_id if best_sim >= self._dedup_threshold else None
        except Exception:
            return None

    def _fts_ids(self, query, type=None, limit=200):
        match = _fts_query(query)
        if match is None:
            return []
        sql = ("SELECT m.id FROM memories_fts f JOIN memories m ON m.id = f.rowid "
               "WHERE memories_fts MATCH ? AND m.valid_to IS NULL")
        params = [match]
        if type is not None:
            sql += " AND m.type = ?"
            params.append(type)
        # secondary sort by recency: bm25 ties must not be broken by SQLite's
        # arbitrary row-scan order (that artificial tiebreak is a full RRF
        # rank apart, which swamps the gentle recency weight applied later)
        sql += " ORDER BY rank, m.updated_at DESC LIMIT ?"
        params.append(limit)
        return [r[0] for r in self._conn.execute(sql, params).fetchall()]

    def _vector_ids(self, query, type=None, limit=200):
        """Ids ranked by cosine similarity to the query, or [] when semantic
        recall is unavailable (no embedder / no numpy / no stored vectors).
        Never raises - any failure degrades to keyword-only recall."""
        if self._embedder is None or not query.strip():
            return []
        try:
            import numpy as np

            q = np.asarray(self._embedder.embed(query), dtype=np.float32)
            # a zero-magnitude query vector carries no directional signal;
            # ranking every row by it would just return the store in arbitrary
            # order, so semantic search contributes nothing here.
            if not np.any(q):
                return []
            sql = ("SELECT id, embedding FROM memories "
                   "WHERE embedding IS NOT NULL AND valid_to IS NULL")
            params = []
            if type is not None:
                sql += " AND type = ?"
                params.append(type)
            rows = self._conn.execute(sql, params).fetchall()
            if not rows:
                return []
            ids = [r[0] for r in rows]
            mat = np.frombuffer(b"".join(r[1] for r in rows),
                                dtype="<f4").reshape(len(ids), -1)
            # stored vectors and q are unit-normalized, so dot == cosine
            sims = mat @ q
            # #15: only genuinely-similar rows seed the walk. Rows below the
            # floor are left for the graph to reach by edge, not seeded as
            # near-tied noise. (floor 0 -> pre-#15 behavior: keep the whole store.)
            order = [i for i in np.argsort(-sims)[:limit]
                     if sims[i] >= self._seed_floor]
            return [ids[i] for i in order]
        except Exception:
            return []

    def _hydrate(self, ids) -> list:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, type, title, body, tags, updated_at FROM memories "
            f"WHERE id IN ({placeholders}) AND valid_to IS NULL", ids).fetchall()
        by_id = {r[0]: {"id": r[0], "type": r[1], "title": r[2],
                        "body": r[3], "tags": r[4], "updated_at": r[5]}
                 for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def _seed_scores(self, query, type) -> dict:
        """The v0.2 hybrid recall scoring (FTS5 + semantic RRF, gentle recency,
        optional decay) as a {id: score} map - the seeds an associative walk
        starts from."""
        fts_ids = self._fts_ids(query, type)
        vec_ids = self._vector_ids(query, type)
        lists = [fts_ids] + ([vec_ids] if vec_ids else [])
        scores = _rrf_scores(lists)
        if not scores:
            return {}
        # gentle recency signal: breaks ties, never overrides a strong match
        recency = _rrf_scores([self._recency_order(list(scores))])
        for mid, s in recency.items():
            scores[mid] += _RECENCY_WEIGHT * s
        # optional exponential time-decay
        if self._decay_half_life_days:
            now = _now()
            updated = self._updated_at_of(list(scores))
            for mid in list(scores):
                scores[mid] *= _decay_factor(
                    _age_days(updated[mid], now), self._decay_half_life_days)
        return scores

    def _tags_of_many(self, ids) -> dict:
        if not ids:
            return {}
        ph = ",".join("?" for _ in ids)
        return {r[0]: r[1] for r in self._conn.execute(
            f"SELECT id, tags FROM memories WHERE id IN ({ph})", ids).fetchall()}

    def _recall_by_tags(self, type, tag_list, limit) -> list:
        """Exact-match tag retrieval, bypassing ranked search and the
        associative graph entirely: every current memory whose tags are a
        superset of `tag_list`, newest first within `limit` - deterministic,
        not subject to FTS/semantic ranking dropping a real match (#50)."""
        sql = "SELECT id, tags FROM memories WHERE valid_to IS NULL"
        params = []
        if type is not None:
            sql += " AND type = ?"
            params.append(type)
        sql += " ORDER BY updated_at DESC, id DESC"
        rows = self._conn.execute(sql, params).fetchall()
        ids = [mid for mid, tags_s in rows if _tags_match(tags_s, tag_list)]
        return self._hydrate(ids[:limit])

    def recall(self, query, type=None, limit=20, budget=None, session=None,
               tags=None) -> list:
        tag_list = _parse_tags(tags)
        if not query or not query.strip():
            if not tag_list:
                return []
            return self._recall_by_tags(type, tag_list, limit)
        seeds = self._seed_scores(query, type)
        if tag_list:
            tags_by_id = self._tags_of_many(list(seeds))
            seeds = {mid: s for mid, s in seeds.items()
                     if _tags_match(tags_by_id.get(mid, ""), tag_list)}
        if not self._graph.enabled:
            if not seeds:
                return []
            order = [mid for mid, _ in sorted(
                seeds.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]
            return self._hydrate(order)          # v0.2 shape, no `via`
        # associative path: seed -> prime -> spread -> two-tier rank -> learn -> receipts
        if budget is None:
            budget = self._recall_budget
        sid = self._graph.resolve_session(session, self._meta_get, self._meta_set)
        # `seeds` stays the immutable v0.2 result (the protected tier); prime a copy
        # so priming/spread never reorder the seed tier.
        activated = dict(seeds)
        for mid, a in self._graph.session_activation(sid).items():
            activated[mid] = activated.get(mid, 0.0) + _PRIMING_WEIGHT * a
        # gate on `seeds`, not the union with primed `activated` - a query with
        # no real hits must not surface a session's primed context (#46).
        if not seeds:
            return []
        activation, receipts = self._graph.spread(activated, budget, type)
        # protect-head / re-rank-tail. The #15 seed floor bounds `seeds` to
        # genuinely-relevant hits, so the head is the real direct matches. Lock
        # that head in exact v0.2 order (a direct hit can't be demoted -> no #25
        # regression), then re-rank everything below it by spread activation,
        # which surfaces connected-but-weakly-matched memories (reached by edge,
        # now below the floor as seeds) into the slots the direct hits didn't
        # claim. HEAD is the protected-prefix size.
        seed_order = [m for m, _ in sorted(
            seeds.items(), key=lambda kv: (-kv[1], kv[0]))]
        head = seed_order[:self._protect_head]
        head_set = set(head)
        tail = sorted((m for m in activation if m not in head_set),
                      key=lambda m: (-activation[m], m))
        if tag_list:
            # a tag filter must hold for the whole result, not just the seed
            # tier - otherwise associative spread could hand back a hit the
            # filter was supposed to exclude.
            tail_tags = self._tags_of_many(tail)
            tail = [m for m in tail if _tags_match(tail_tags.get(m, ""), tag_list)]
        order = (head + tail)[:limit]
        # B1: learn from what the query was ABOUT (the direct-hit head), not
        # from everything the recall returned. Spread- and priming-surfaced
        # tail members consume session activation but never produce it —
        # otherwise any member that once enters the session is re-surfaced by
        # priming into the next result list, re-bumped, and re-wired: a
        # feedback loop that wires spurious cross-task cliques at cap weight
        # (measured on the bench corpus: 80 spurious vs 36 true edges).
        # HEBBIAN_LEARN_FROM_HEAD is a knob (default True == the above): False
        # reverts to learning from the full returned order (pre-B1 behavior).
        # Read as a module attribute (not imported by value) so a test can
        # flip it at runtime via monkeypatch.
        learn_ids = head if graph.HEBBIAN_LEARN_FROM_HEAD else order
        self._graph.touch_session(sid, learn_ids)
        self._conn.commit()
        hits = self._hydrate(order)
        for h in hits:
            r = receipts.get(h["id"])
            if r is not None and h["id"] not in seeds:
                h["via"] = {"path": [{"from": r["from"], "kind": r["kind"], "w": r["w"]}],
                            "hops": r["hops"]}
            else:
                h["via"] = {"seed": True}
        return hits

    def _recency_order(self, ids):
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        return [r[0] for r in self._conn.execute(
            f"SELECT id FROM memories WHERE id IN ({ph}) "
            f"ORDER BY updated_at DESC, id DESC", ids).fetchall()]

    def _updated_at_of(self, ids):
        if not ids:
            return {}
        ph = ",".join("?" for _ in ids)
        return {r[0]: r[1] for r in self._conn.execute(
            f"SELECT id, updated_at FROM memories WHERE id IN ({ph})", ids).fetchall()}

    def _links_of(self, mid) -> list:
        row = self._conn.execute("SELECT links FROM memories WHERE id=?", (mid,)).fetchone()
        if row is None:
            raise ValueError(f"no memory with id {mid}")
        return json.loads(row[0])

    def link(self, id_a, id_b) -> dict:
        a = self._links_of(id_a)
        b = self._links_of(id_b)
        if id_b not in a:
            a.append(id_b)
        if id_a not in b:
            b.append(id_a)
        now = _now()
        self._conn.execute("UPDATE memories SET links=?, updated_at=? WHERE id=?",
                           (json.dumps(a), now, id_a))
        self._conn.execute("UPDATE memories SET links=?, updated_at=? WHERE id=?",
                           (json.dumps(b), now, id_b))
        self._graph.on_link(id_a, id_b)
        self._conn.commit()
        self._sync_now()
        return {"linked": [id_a, id_b]}

    def dismiss_cluster(self, id_a, id_b) -> dict:
        self._graph.dismiss_peak(id_a, id_b)
        return {"dismissed": [id_a, id_b]}

    def forget(self, id) -> dict:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (id,))
        self._graph.on_forget(id)
        self._conn.commit()
        self._sync_now()
        return {"forgotten": id, "existed": cur.rowcount > 0}

    def boot_index(self) -> str:
        rows = self._conn.execute(
            "SELECT id, type, title, updated_at FROM memories WHERE valid_to IS NULL "
            "ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        if not rows:
            return "(no memories yet)"
        full = "\n".join(f"[{t}] #{i} {title}" for i, t, title, _ in rows)
        if not self._graph.enabled or len(rows) <= self._boot_index_cap:
            return full                                  # today's behavior
        try:
            return self._curated_index(rows, self._graph.degree_map(),
                                       self._boot_index_cap)
        except Exception:
            return full                                  # curation failure -> full list

    def _curated_index(self, rows, deg, cap) -> str:
        # rows: [(id, type, title, updated_at)] newest-first
        meta = {r[0]: (r[1], r[2]) for r in rows}        # id -> (type, title)
        upd = {r[0]: r[3] for r in rows}                 # id -> updated_at (tie-break)
        newest = [r[0] for r in rows]
        reserve = max(1, cap // 4)
        recent = newest[:reserve]
        recent_set = set(recent)
        hubs = sorted(
            (mid for mid in deg if deg[mid] > 0 and mid not in recent_set),
            key=lambda mid: (deg[mid], upd[mid], mid), reverse=True,
        )[: cap - reserve]                               # degree desc, then updated_at desc, then id desc
        chosen = set(hubs) | recent_set
        for mid in newest:                               # fill remaining budget with recency
            if len(chosen) >= cap:
                break
            if mid not in chosen:
                recent.append(mid)
                chosen.add(mid)

        def line(mid):
            t, title = meta[mid]
            return f"[{t}] #{mid} {title}"

        if not hubs:
            return "\n".join(line(mid) for mid in recent)
        parts = ["# Load-bearing"] + [line(mid) for mid in hubs]
        parts += ["# Recent"] + [line(mid) for mid in recent]
        return "\n".join(parts)

    def crystallization_candidates(self) -> list:
        """Read-time derived view of principle candidates. [] when disabled.
        Process-memoized on a cheap graph signature (adjustment C) so repeated
        reads in one reflection pass don't recompute. Never raises.

        The signature MUST include the dismissed-set count: dismiss_cluster writes
        to crystallize_dismissed, NOT edges, so an edges-only signature would keep
        serving a dismissed candidate from cache for the life of the process
        (dismissal silently no-ops). Naming a principle self-invalidates because it
        adds crystallized edges."""
        if not self._crystallize:
            return []
        try:
            from . import crystallize
            sig = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM edges), "
                "(SELECT COALESCE(MAX(updated_at), '') FROM edges), "
                "(SELECT COUNT(*) FROM crystallize_dismissed)").fetchone()
            if sig != self._cryst_sig:
                self._cryst_cache = crystallize.candidates(self._conn, self._embedder)
                self._cryst_sig = sig
            return self._cryst_cache
        except Exception:
            return []
