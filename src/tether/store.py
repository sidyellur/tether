"""store.py - the memory store. Owns ALL SQL.

One table (`memories`) plus an external-content FTS5 index kept in sync by
triggers. The four verbs and the boot index are the only public surface;
nothing outside this module speaks SQL.
"""

import bisect
import json
import re
import struct
import sys
import time
import warnings
from datetime import datetime, timezone

from . import graph, sync
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


# How long a read-path sync is allowed to block (#62). Deliberately much
# shorter than the 2.0s write-path default: recall is the latency-visible hot
# path, and sync_now runs the pull on a background thread that keeps going
# after we stop waiting. So a slow backend costs at most this much, and its
# data lands for the NEXT read instead - the freshening is self-healing rather
# than something recall has to block on.
_READ_SYNC_TIMEOUT = 1.0

# Default excerpt width (#30). recall used to return every hit's FULL body, so
# one 55KB memory made unrelated queries cost 77KB of serialize-and-pipe while
# the retrieval itself took under a millisecond - the response was fat, not the
# engine. Search engines solved this long ago: return a ranked index of
# pointers with enough text to judge relevance, and let the caller fetch the
# one document it actually wants. 500 chars is the issue's own estimate for
# taking those 77KB payloads to ~1-2KB.
_EXCERPT_CHARS = 500
# How far to hunt for a word boundary before giving up and cutting mid-word.
_EXCERPT_SNAP = 40


def _excerpt(body: str, query, max_chars: int):
    """(text, truncated) - a relevance-centered window of `body`.

    Centers on the first query term that appears in the body, so the caller
    sees WHY the memory matched rather than just its opening lines. Falls back
    to the head of the body when nothing matches (a semantic-only hit, or a
    tag lookup with no query at all). Ellipses mark where text was cut.
    """
    if max_chars <= 0 or not body or len(body) <= max_chars:
        return body, False
    low = body.lower()
    pos = -1
    for tok in re.split(r"\W+", (query or "").lower()):
        if len(tok) < 3:
            continue                    # too short to locate anything useful
        found = low.find(tok)
        if found != -1 and (pos == -1 or found < pos):
            pos = found
    if pos == -1:
        cut = body[:max_chars].rstrip()
        return cut + "…", True
    start = max(0, pos - max_chars // 2)
    end = min(len(body), start + max_chars)
    start = max(0, end - max_chars)     # re-anchor if we hit the tail
    if start > 0:                       # snap forward to a word boundary
        space = body.find(" ", start, start + _EXCERPT_SNAP)
        if space != -1:
            start = space + 1
    if end < len(body):                 # snap back to a word boundary
        space = body.rfind(" ", max(start, end - _EXCERPT_SNAP), end)
        if space != -1:
            end = space
    out = body[start:end].strip()
    if start > 0:
        out = "…" + out
    if end < len(body):
        out = out + "…"
    return out, True


class Store:
    def __init__(self, conn, device_id: str, sync_now, embedder=None,
                 author="", consolidate=False, dedup_threshold=0.92,
                 decay_half_life_days=None, assoc=False, recall_budget=8,
                 protect_head=_PROTECT_HEAD, seed_floor=_SEED_FLOOR,
                 crystallize=False,
                 boot_index_cap=50, forget=False, forget_age_days=90,
                 forget_interval=20, forget_max_per_sweep=10,
                 session_sweep_interval=50, sync_read_interval=30,
                 excerpt_chars=_EXCERPT_CHARS,
                 db_path=None, on_degrade=None):
        self._conn = conn
        self._device_id = device_id
        self._sync_now = sync_now
        self._embedder = embedder
        # Set only when `conn` is a sync replica (server.py); lets a failed
        # replica write degrade to a fresh local-only connection instead of
        # raising (#44). None (the local-only default) means there is
        # nothing to degrade to, so failures propagate as before.
        self._db_path = db_path
        # Optional: notified once when a degrade actually happens, so a
        # caller (server.py's tether://status resource) can stop reporting a
        # stale "replica" sync mode. Never lets a notification failure break
        # the degrade itself.
        self._on_degrade = on_degrade
        self._degraded = False
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
        self._session_sweep_interval = session_sweep_interval
        # #62: sync_now only ran after writes, so a device that merely READS
        # never pulled other devices' updates - it saw the store as of its own
        # startup probe until it happened to write something. Reads now pull
        # too, debounced to at most once per this many seconds (0/None = off).
        self._sync_read_interval = sync_read_interval
        self._last_sync_at = None
        self._excerpt_chars = excerpt_chars
        # #61: (ids, matrix, types) for every current embedding, built once and
        # reused. Before this, _vector_ids, _find_near_duplicate and
        # graph.on_remember each re-read and re-deserialized every embedding
        # blob in the store on every call - a single remember() paid for two
        # full scans, and recall paid for one.
        #
        # #81: the cache is now maintained INCREMENTALLY by the row-level
        # writes (_cache_put / _cache_drop) instead of being dropped on every
        # write. Dropping it made each remember() re-read every blob in the
        # store to rebuild it for kNN wiring - O(N) per write, 77% of a
        # remember at 8k memories - and the next recall paid the same rebuild
        # again. `ids` is kept sorted ascending so the incremental matrix is
        # byte-identical to a fresh `ORDER BY id` scan (ranking ties break on
        # row order). Full rebuilds remain only where the rows change
        # wholesale (backfill, degrade) or behind our back (see
        # _emb_cache_version).
        self._emb_cache = None
        # Backing storage for the cached matrix: `mat` is a row-prefix view of
        # this buffer, which is over-allocated (doubling) so appending a new
        # memory is O(dims) rather than a copy of the whole matrix.
        self._emb_buf = None
        # The `PRAGMA data_version` observed when the cache was built. It
        # changes only when ANOTHER connection commits to the file (our own
        # writes never bump it), so it is a near-free way to notice a CLI
        # purge, a second server process, or a replica pull that landed rows
        # the incremental path never saw - and rebuild instead of serving a
        # stale matrix. None when the pragma is unavailable (-> no check).
        self._emb_cache_version = None

    def _degrade_to_local(self) -> bool:
        """A replica write just failed (e.g. the network dropped mid-session).
        Swap to a fresh local-only connection for the remainder of the
        process so the caller's write can be retried instead of raised.
        Returns False (nothing to degrade to, or already degraded) when the
        original error should propagate instead."""
        if self._degraded or self._db_path is None:
            return False
        try:
            conn, sync_now, _mode = sync._local(self._db_path)
        except Exception:
            return False
        self._conn = conn
        self._graph._conn = conn
        self._sync_now = sync_now
        self._invalidate_embedding_cache()   # different DB, different rows
        self._degraded = True
        if self._on_degrade is not None:
            try:
                self._on_degrade()
            except Exception:
                pass
        sys.stderr.write(
            "tether: replica write failed; degrading to local-only for the "
            "remainder of the process\n")
        return True

    def _invalidate_embedding_cache(self) -> None:
        """Drop the whole cache so the next read rebuilds it from SQL. The
        blunt path (#61), now reserved for writes that change rows wholesale
        - backfill_embeddings, _degrade_to_local - or any incremental update
        that can't be applied cleanly. Row-level writes use _cache_put /
        _cache_drop instead (#81)."""
        self._emb_cache = None
        self._emb_buf = None
        self._emb_cache_version = None

    def _data_version(self):
        """SQLite's per-connection change counter for OTHER connections'
        commits, or None if the pragma is unavailable (never raises)."""
        try:
            return self._conn.execute("PRAGMA data_version").fetchone()[0]
        except Exception:
            return None

    def _cache_put(self, mid, emb, type_) -> None:
        """Reflect one row's current embedding in the cache (#81): replace it
        if the row is already cached, otherwise insert it at its sorted
        position. `emb is None` means the row has no vector any more, so it
        is dropped. A no-op when there is no cache to maintain (the next read
        rebuilds from SQL, which already includes this row). Any surprise -
        a dimension mismatch, a numpy failure - falls back to a full rebuild
        rather than risking a subtly wrong matrix.

        Rows are edited in the backing buffer in place, and a new highest id
        (the AUTOINCREMENT common case) is appended into spare capacity, so
        the per-write cost is O(dims), not a copy of the matrix. Nothing
        holds a matrix across a put: remember() fetches its shared copy
        AFTER the upsert has patched the cache.
        """
        cache = self._emb_cache
        if cache is None:
            return
        if emb is None:
            self._cache_drop(mid)
            return
        try:
            import numpy as np

            ids, mat, types = cache
            vec = np.frombuffer(emb, dtype="<f4")
            if mat is not None and vec.shape[0] != mat.shape[1]:
                self._invalidate_embedding_cache()
                return
            pos = bisect.bisect_left(ids, mid)
            n = len(ids)
            if pos < n and ids[pos] == mid:
                mat[pos] = vec
                types = list(types)
                types[pos] = type_
                self._emb_cache = (ids, mat, types)
                return
            buf = self._emb_buf
            if (pos == n and mat is not None and buf is not None
                    and mat.base is buf and buf.shape[0] > n):
                buf[n] = vec                          # append, no copy
            else:
                # Grow (or insert mid-way, e.g. a restore of an old id):
                # one copy into a buffer with doubling headroom.
                buf = np.empty((max(16, 2 * (n + 1)), vec.shape[0]), dtype=np.float32)
                if mat is not None:
                    buf[:pos] = mat[:pos]
                    buf[pos + 1:n + 1] = mat[pos:]
                buf[pos] = vec
                self._emb_buf = buf
            self._emb_cache = ([*ids[:pos], mid, *ids[pos:]], buf[:n + 1],
                               [*types[:pos], type_, *types[pos:]])
        except Exception:
            self._invalidate_embedding_cache()

    def _cache_drop(self, mid) -> None:
        """Remove one row from the cache (#81): forgotten, superseded, swept,
        purged, or re-written without a vector. No-op if uncached. Shifts
        the rows above it down in place (numpy handles the overlap)."""
        cache = self._emb_cache
        if cache is None:
            return
        try:
            ids, mat, types = cache
            pos = bisect.bisect_left(ids, mid)
            n = len(ids)
            if pos >= n or ids[pos] != mid:
                return
            buf = self._emb_buf
            if n == 1:
                mat = None
            elif buf is not None and mat.base is buf:
                buf[pos:n - 1] = buf[pos + 1:n]
                mat = buf[:n - 1]
            else:
                import numpy as np
                mat = np.delete(mat, pos, axis=0)
            self._emb_cache = ([*ids[:pos], *ids[pos + 1:]], mat,
                               [*types[:pos], *types[pos + 1:]])
        except Exception:
            self._invalidate_embedding_cache()

    def _embedding_matrix(self):
        """(ids, matrix, types) over every CURRENT embedding, or (None, ...) if
        semantic support is unavailable. One scan, then kept current by the
        row-level writes (#81) - rebuilt only if another connection has
        committed to the file since. Never raises - callers degrade to
        keyword-only."""
        if self._emb_cache is not None:
            version = self._data_version()
            if version is None or version == self._emb_cache_version:
                return self._emb_cache
            self._invalidate_embedding_cache()
        try:
            import numpy as np

            # Read the version BEFORE the scan so a commit that lands between
            # the two is caught on the next read rather than missed forever.
            version = self._data_version()
            rows = self._conn.execute(
                "SELECT id, embedding, type FROM memories "
                "WHERE embedding IS NOT NULL AND valid_to IS NULL "
                "ORDER BY id").fetchall()
            if not rows:
                self._emb_cache = ([], None, [])
            else:
                ids = [r[0] for r in rows]
                # A writable, exactly-sized buffer (frombuffer's view of the
                # joined blobs is read-only); the first append grows it.
                buf = np.array(np.frombuffer(b"".join(r[1] for r in rows),
                                             dtype="<f4").reshape(len(ids), -1))
                self._emb_buf = buf
                self._emb_cache = (ids, buf[:len(ids)], [r[2] for r in rows])
            self._emb_cache_version = version
            return self._emb_cache
        except Exception:
            return ([], None, [])

    def _sync(self, timeout=2.0) -> None:
        """Every sync goes through here so the read-path debounce (#62) sees
        writes too - a chatty writer shouldn't also pull on every read."""
        self._last_sync_at = time.monotonic()
        self._sync_now(timeout)

    def _maybe_sync_for_read(self) -> None:
        """Pull before a read, at most once per _sync_read_interval seconds.

        Bounded by _READ_SYNC_TIMEOUT rather than the write default, and never
        allowed to raise: a read must still serve local data when the backend
        is unreachable - exactly what it would have served before this existed.
        """
        if not self._sync_read_interval:
            return
        now = time.monotonic()
        if (self._last_sync_at is not None
                and now - self._last_sync_at < self._sync_read_interval):
            return
        try:
            self._sync(timeout=_READ_SYNC_TIMEOUT)
        except Exception:
            self._last_sync_at = now    # don't retry-storm a broken backend

    def _write_with_replica_fallback(self, fn):
        """Run a write; on failure, degrade to local once and retry."""
        try:
            return fn()
        except Exception:
            if not self._degrade_to_local():
                raise
            return fn()

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

    def _embedding_meta_key(self, name: str) -> str:
        """Per-device meta key for embedding-model tracking (#45). `meta` rows
        sync like any other data, so a global key made two devices running
        different TETHER_EMBEDDING_MODEL values fight over it forever - each
        boot/sync seeing the other's model name as a "mismatch" and re-wiping.
        Scoping by author/device id means a device only ever compares against
        the value IT wrote, so it converges after its own first backfill."""
        scope = self._author or self._device_id
        return f"{name}:{scope}" if scope else name

    def backfill_embeddings(self, batch=200) -> int:
        """Embed rows lacking a vector. If the active model/dims differ from
        what produced the stored vectors (per this device - see
        _embedding_meta_key), clear them all first so the store never mixes
        incompatible embeddings. Returns rows embedded. No-op (returns 0)
        without an embedder; never raises."""
        if self._embedder is None:
            return 0
        try:
            model_key = self._embedding_meta_key("embedding_model")
            dims_key = self._embedding_meta_key("embedding_dims")
            prev_model = self._meta_get(model_key)
            prev_dims = self._meta_get(dims_key)
            if (prev_model != self._embedder.name
                    or prev_dims != str(self._embedder.dims)):
                self._conn.execute("UPDATE memories SET embedding=NULL")
                self._meta_set(model_key, self._embedder.name)
                self._meta_set(dims_key, self._embedder.dims)
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
                        self._invalidate_embedding_cache()
                        self._conn.commit()
                        return done
                    self._conn.execute(
                        "UPDATE memories SET embedding=? WHERE id=?", (blob, mid))
                    done += 1
                self._conn.commit()
            # Vectors just changed wholesale (and the model-change branch above
            # may have NULLed every one of them), so anything cached is stale.
            self._invalidate_embedding_cache()
            self._graph.backfill_semantic(matrix=self._embedding_matrix())
            return done
        except Exception:
            return 0

    def remember(self, type, title, body, tags=None, links=None,
                 crystallizes=None) -> dict:
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got {type!r}")
        return self._write_with_replica_fallback(
            lambda: self._remember_impl(type, title, body, tags, links, crystallizes))

    def _remember_impl(self, type, title, body, tags, links, crystallizes) -> dict:
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

        # Share one matrix across the whole write (#61). The upsert above
        # already patched this row into the cache (#81), so this is a cache
        # hit that includes `mid` - on_remember excludes `mid` itself, so the
        # row costs nothing but also never triggers a rescan. Before #81 the
        # cache was dropped here and every remember re-read the whole store.
        shared = (self._embedding_matrix()
                  if emb is not None and self._graph.enabled else None)
        self._graph.on_remember(mid, emb, matrix=shared)
        if self._crystallize and crystallizes:
            self._graph.on_crystallize(mid, crystallizes)
        self._conn.commit()
        self._sync()
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
        the probe SELECT): the DB resolves the real conflict, not us.

        The links merge itself must happen inside this same statement rather
        than in Python from `existing` - `existing` can be stale under a
        genuine race (two connections both probe a brand-new title, both see
        no row, both pass different `links`). Merging from that stale
        snapshot would make the losing side's DO UPDATE branch *replace*
        rather than merge the winner's links, reintroducing #47's clobber
        inside the exact race #41 exists to close. SQLite evaluates
        `memories.links` in the SET clause against the row's true current
        value at conflict time, so unioning it with `excluded.links` here is
        correct regardless of who actually won the race.
        """
        links_s = json.dumps(incoming_links)
        superseded = (self._find_near_duplicate(type, emb)
                      if (existing is None and self._consolidate) else None)
        # Snapshot last_insert_rowid() so we can ask the DB which branch of the
        # upsert actually ran (see the action= line below). Read immediately
        # before the INSERT - everything above is SELECT-only, so nothing can
        # have moved it since.
        rowid_before = self._conn.execute(
            "SELECT last_insert_rowid()").fetchone()[0]
        self._conn.execute(
            "INSERT INTO memories(type, title, title_norm, body, tags, links, "
            "created_at, updated_at, device_id, embedding, author, valid_from) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(type, title_norm) WHERE valid_to IS NULL DO UPDATE SET "
            "title=excluded.title, body=excluded.body, tags=excluded.tags, "
            "links=(SELECT json_group_array(value) FROM ("
            "    SELECT value FROM json_each(memories.links) "
            "    UNION SELECT value FROM json_each(excluded.links))), "
            "updated_at=excluded.updated_at, device_id=excluded.device_id, "
            "author=excluded.author, embedding=excluded.embedding",
            (type, title, norm, body, tags_s, links_s, now, now,
             self._device_id, emb, self._author, now))
        # Re-resolve the row: lastrowid doesn't advance on the DO UPDATE branch,
        # so it can't identify the row on its own.
        mid, = self._conn.execute(
            "SELECT id FROM memories "
            "WHERE type=? AND title_norm=? AND valid_to IS NULL",
            (type, norm)).fetchone()
        # Which branch ran? This used to be `created_at == now`, which quietly
        # assumed the clock ticks between two writes. It doesn't always:
        # datetime.now() has ~15.6ms resolution on Windows before Python 3.13,
        # so two remembers inside one tick share a timestamp and an UPDATE
        # reported itself as "created" (caught by the Windows CI job added in
        # #68; reproducible anywhere by freezing the clock). The data was right,
        # but `action` is an agent-facing signal - it's how a caller tells "I
        # made a new memory" from "I refined an existing one".
        #
        # last_insert_rowid() only advances when a row is really inserted, so
        # asking the DB what it did beats inferring it from wall-clock time.
        # AUTOINCREMENT never reissues an id, so a genuine insert can't collide
        # with the previous value.
        rowid_after = self._conn.execute(
            "SELECT last_insert_rowid()").fetchone()[0]
        action = ("created" if rowid_after != rowid_before and rowid_after == mid
                  else "updated")
        # Either branch leaves `mid` current with embedding=emb (#81).
        self._cache_put(mid, emb, type)
        if action == "created" and superseded is not None:
            self._conn.execute(
                "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                (now, mid, superseded))
            self._graph.unprime(superseded)
            self._cache_drop(superseded)
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
            self._cache_put(mid, emb, type)
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
        self._cache_put(mid, emb, type)
        if superseded is not None:
            self._conn.execute(
                "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                (now, mid, superseded))
            self._graph.unprime(superseded)
            self._cache_drop(superseded)
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

    def _maybe_sweep_sessions(self) -> None:
        """Amortized trigger: every session_sweep_interval recalls, sweep
        session_members for sessions abandoned longer than the sweep horizon
        (#48). Independent of any specific session being active. No-op when
        the graph is disabled."""
        if not self._graph.enabled:
            return
        try:
            n = int(self._meta_get("session_sweep_counter") or 0) + 1
            if n >= self._session_sweep_interval:
                self._meta_set("session_sweep_counter", 0)
                self._conn.commit()
                self._graph.sweep_stale_session_members()
                self._conn.commit()
            else:
                self._meta_set("session_sweep_counter", n)
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
                self._cache_drop(mid)
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

            ids, mat, types = self._embedding_matrix()
            if mat is None:
                return None
            q = np.frombuffer(emb, dtype="<f4")
            sims = mat @ q                       # both unit-norm, so dot == cosine
            best_id, best_sim = None, -1.0
            for i, mid in enumerate(ids):
                if types[i] != type:
                    continue
                sim = float(sims[i])
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
            all_ids, mat, all_types = self._embedding_matrix()
            if mat is None:
                return []
            if type is not None:
                keep = [i for i, t in enumerate(all_types) if t == type]
                if not keep:
                    return []
                ids = [all_ids[i] for i in keep]
                mat = mat[keep]
            else:
                ids = all_ids
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

    def get(self, id) -> dict:
        """One memory, whole - the fetch half of snippet-plus-fetch (#30).
        Returns None when the id doesn't exist or is no longer current."""
        self._maybe_sync_for_read()
        hits = self._hydrate([id])
        return hits[0] if hits else None

    def _excerpt_hits(self, hits, query, full) -> list:
        """Replace each hit's body with a relevance-centered excerpt (#30).

        Keeps the `body` key rather than renaming it to `excerpt`: a caller
        doing hit["body"] keeps working, it just gets the part that matters
        instead of tens of KB. `truncated` and `body_chars` are added only when
        text was actually cut, so a hit that fits is byte-identical to before
        and the agent can tell when there is more to fetch.
        """
        if full or not self._excerpt_chars:
            return hits
        for h in hits:
            text, cut = _excerpt(h["body"], query, self._excerpt_chars)
            if cut:
                h["body_chars"] = len(h["body"])
                h["truncated"] = True
            h["body"] = text
        return hits

    def recall(self, query, type=None, limit=20, budget=None, session=None,
               tags=None, full=False) -> list:
        self._maybe_sync_for_read()      # #62: a read-only device pulls too
        tag_list = _parse_tags(tags)
        if not query or not query.strip():
            if not tag_list:
                return []
            return self._excerpt_hits(
                self._recall_by_tags(type, tag_list, limit), query, full)
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
            return self._excerpt_hits(
                self._hydrate(order), query, full)   # v0.2 shape, no `via`
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
        self._maybe_sweep_sessions()
        hits = self._hydrate(order)
        for h in hits:
            r = receipts.get(h["id"])
            if r is not None and h["id"] not in seeds:
                h["via"] = {"path": [{"from": r["from"], "kind": r["kind"], "w": r["w"]}],
                            "hops": r["hops"]}
            else:
                h["via"] = {"seed": True}
        return self._excerpt_hits(hits, query, full)

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

    # #38: union the new id into links inside the statement itself, rather than
    # SELECT-ing the list into Python, appending, and writing it back. The old
    # read-modify-write ran its SELECT outside any transaction (sqlite3 only
    # opens one on the first DML), so two concurrent link() calls touching the
    # same memory could both read the same "before" list and the later UPDATE
    # would clobber the earlier one's addition - the reproduction in #38 left
    # every node with 1 of its 3 expected links. Computing the union in SQL
    # makes each UPDATE self-contained: it reads the row's true current value
    # at write time, under the row lock, so there is no window to lose.
    _LINK_UNION_SQL = (
        "UPDATE memories SET links = ("
        "    SELECT json_group_array(value) FROM ("
        "        SELECT value FROM json_each(memories.links)"
        "        UNION SELECT ?))"
        ", updated_at = ? WHERE id = ?")

    def link(self, id_a, id_b) -> dict:
        # Validate ids outside the retry wrapper: a bad id raises ValueError
        # here, which must surface as-is rather than be mistaken for a replica
        # write failure and trigger a needless degrade.
        self._links_of(id_a)
        self._links_of(id_b)
        return self._write_with_replica_fallback(
            lambda: self._link_impl(id_a, id_b))

    def _link_impl(self, id_a, id_b) -> dict:
        now = _now()
        self._conn.execute(self._LINK_UNION_SQL, (id_b, now, id_a))
        self._conn.execute(self._LINK_UNION_SQL, (id_a, now, id_b))
        self._graph.on_link(id_a, id_b)
        self._conn.commit()
        self._sync()
        return {"linked": [id_a, id_b]}

    def dismiss_cluster(self, id_a, id_b) -> dict:
        """Reflection control, not a memory operation. Refuses when
        crystallization is off (#65): dismissals are persistent rows in
        crystallize_dismissed, so a stray call against a store that isn't
        crystallizing would silently suppress a candidate later, whenever the
        feature does get enabled. Nothing else in tether consumes the table,
        so writing it while disabled is pure future damage."""
        if not self._crystallize:
            raise ValueError(
                "crystallization is not enabled (set TETHER_CRYSTALLIZE=1); "
                "dismiss_cluster has nothing to dismiss")
        self._graph.dismiss_peak(id_a, id_b)
        return {"dismissed": [id_a, id_b]}

    def forget(self, id) -> dict:
        return self._write_with_replica_fallback(lambda: self._forget_impl(id))

    def _forget_impl(self, id) -> dict:
        """Soft-delete: mark the memory no longer current via the same
        valid_to machinery consolidation and forgetting-by-disconnection
        already use. Reversible (clear valid_to to restore) and, like the
        forgetting sweep, keeps edges intact rather than tearing down the
        usage graph. Excluded from recall/boot_index like any other
        no-longer-current row. Use purge() for a real, non-reversible delete."""
        now = _now()
        cur = self._conn.execute(
            "UPDATE memories SET valid_to=? WHERE id=? AND valid_to IS NULL",
            (now, id))
        if cur.rowcount > 0:
            self._graph.unprime(id)          # #42: don't let it linger as primed context
            self._cache_drop(id)
        self._conn.commit()
        self._sync()
        return {"forgotten": id, "existed": cur.rowcount > 0}

    def restore(self, id) -> dict:
        """Un-forget: clear valid_to so a soft-deleted memory is current again.

        forget(), consolidation and the forgetting sweep all promise to be
        reversible, but nothing exposed the reversal (#64) - so in practice a
        soft-deleted memory was only recoverable by hand-editing the DB. This
        is that reversal, kept CLI-only for the same reason purge is: it is an
        operator action, not something an agent should reach for.
        """
        return self._write_with_replica_fallback(lambda: self._restore_impl(id))

    def _restore_impl(self, id) -> dict:
        row = self._conn.execute(
            "SELECT type, title_norm, valid_to FROM memories WHERE id=?",
            (id,)).fetchone()
        if row is None:
            return {"restored": id, "existed": False, "action": "missing"}
        type_, norm, valid_to = row
        if valid_to is None:
            return {"restored": id, "existed": True, "action": "already-current"}
        # A newer memory may have taken over this (type, title) while the row
        # was archived. Restoring would then put two current rows on one dedup
        # key - which the #41 partial unique index rejects anyway, but with an
        # opaque IntegrityError. Check first so the operator gets a message
        # that names the blocking id and what to do about it.
        clash = self._conn.execute(
            "SELECT id FROM memories WHERE type=? AND title_norm=? "
            "AND valid_to IS NULL AND id != ?", (type_, norm, id)).fetchone()
        if clash is not None:
            raise ValueError(
                f"cannot restore #{id}: memory #{clash[0]} is already the "
                f"current {type_!r} titled {norm!r}. forget #{clash[0]} first, "
                f"or edit one of the titles.")
        self._conn.execute(
            "UPDATE memories SET valid_to=NULL, superseded_by=NULL WHERE id=?",
            (id,))
        emb, = self._conn.execute(
            "SELECT embedding FROM memories WHERE id=?", (id,)).fetchone()
        self._cache_put(id, emb, type_)      # back in the current set (#81)
        self._conn.commit()
        self._sync()
        return {"restored": id, "existed": True, "action": "restored"}

    def import_records(self, records) -> dict:
        """Replay exported records through the normal write path.

        Deliberately NOT raw INSERTs: going through remember()/link() means
        upsert-on-(type,title), consolidation, embedding and graph wiring all
        apply, so importing into a non-empty store MERGES rather than
        duplicating or clobbering. The consequence is that ids are not
        preserved - an id in the file may map to a different id here - so
        links are remapped through the (type, title) identity in a second
        pass, and any link pointing outside the file is dropped rather than
        silently pointed at the wrong memory.
        """
        created = updated = skipped = 0
        id_map = {}
        for rec in records:
            try:
                type_ = rec["type"]
                title = rec["title"]
                body = rec.get("body", "")
            except (TypeError, KeyError):
                skipped += 1
                continue
            if type_ not in VALID_TYPES or not str(title).strip():
                skipped += 1
                continue
            res = self.remember(type_, title, body, tags=rec.get("tags", ""))
            if rec.get("id") is not None:
                id_map[rec["id"]] = res["id"]
            if res["action"] == "created":
                created += 1
            else:
                updated += 1

        linked = dropped_links = 0
        seen_pairs = set()
        for rec in records:
            src = id_map.get(rec.get("id")) if isinstance(rec, dict) else None
            if src is None:
                continue
            for old_dst in (rec.get("links") or []):
                new_dst = id_map.get(old_dst)
                if new_dst is None:
                    dropped_links += 1      # points outside the file: unresolvable
                    continue
                pair = (src, new_dst) if src < new_dst else (new_dst, src)
                if pair in seen_pairs or src == new_dst:
                    continue
                seen_pairs.add(pair)
                try:
                    self.link(src, new_dst)
                    linked += 1
                except Exception:
                    dropped_links += 1
        return {"created": created, "updated": updated, "skipped": skipped,
                "linked": linked, "dropped_links": dropped_links}

    def purge(self, id) -> dict:
        """Permanent, non-reversible delete - bypasses valid_to entirely.
        Deliberately NOT exposed as a default MCP tool argument (#49); reserved
        for an explicit admin path (the CLI) so an agent can't trigger it by
        hallucinating a flag."""
        return self._write_with_replica_fallback(lambda: self._purge_impl(id))

    def _purge_impl(self, id) -> dict:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (id,))
        self._graph.on_forget(id)
        self._cache_drop(id)
        self._conn.commit()
        self._sync()
        return {"purged": id, "existed": cur.rowcount > 0}

    def export_all(self) -> list:
        """Dump every CURRENT memory as a plain dict - a backup/export path
        independent of the DB file itself (#49). Excludes the embedding
        vector: it's derived data, cheaply recomputed by backfill_embeddings."""
        rows = self._conn.execute(
            "SELECT id, type, title, body, tags, links, created_at, updated_at, "
            "device_id, author FROM memories WHERE valid_to IS NULL ORDER BY id ASC"
        ).fetchall()
        cols = ("id", "type", "title", "body", "tags", "links", "created_at",
                "updated_at", "device_id", "author")
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            d["links"] = json.loads(d["links"] or "[]")
            out.append(d)
        return out

    def boot_index(self) -> str:
        self._maybe_sync_for_read()      # #62: the session-start read pulls too
        rows = self._conn.execute(
            "SELECT id, type, title, updated_at FROM memories WHERE valid_to IS NULL "
            "ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        if not rows:
            return "(no memories yet)"
        if len(rows) <= self._boot_index_cap:
            return "\n".join(f"[{t}] #{i} {title}" for i, t, title, _ in rows)
        if self._graph.enabled:
            try:
                return self._curated_index(rows, self._graph.degree_map(),
                                           self._boot_index_cap)
            except Exception:
                pass                      # curation failure -> capped fallback below
        # No graph (or curation failed): still apply the cap, just without
        # hub-curation - a size cap must never depend on graph state (#52).
        return self._recency_capped_index(rows, self._boot_index_cap)

    def _recency_capped_index(self, rows, cap) -> str:
        return "\n".join(f"[{t}] #{i} {title}" for i, t, title, _ in rows[:cap])

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
        )[: cap - reserve]               # degree desc, then updated_at desc, then id desc
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
