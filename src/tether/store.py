"""store.py - the memory store. Owns ALL SQL.

One table (`memories`) plus an external-content FTS5 index kept in sync by
triggers. The four verbs and the boot index are the only public surface;
nothing outside this module speaks SQL.
"""

import json
import re
import struct
from datetime import datetime, timezone

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
CREATE INDEX IF NOT EXISTS idx_memories_dedup ON memories(type, title_norm);
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


def _rrf_fuse(ranked_lists, k=60):
    """Reciprocal Rank Fusion: merge several ranked id-lists into one order
    without needing comparable scores across lists. Deterministic - ties break
    by ascending id."""
    scores = {}
    for lst in ranked_lists:
        for rank, mid in enumerate(lst):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    return [mid for mid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


class Store:
    def __init__(self, conn, device_id: str, sync_now, embedder=None):
        self._conn = conn
        self._device_id = device_id
        self._sync_now = sync_now
        self._embedder = embedder

    def migrate(self) -> None:
        fts_existed = self._table_exists("memories_fts")
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_META_SCHEMA)
        self._ensure_embedding_column()
        if not fts_existed:
            # FTS5 external-content tables don't auto-index pre-existing rows
            # in the content table; rebuild so a DB that predates the FTS5
            # table (or embedding column) isn't left with a stale/empty index.
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
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
            return done
        except Exception:
            return 0

    def remember(self, type, title, body, tags=None, links=None) -> dict:
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got {type!r}")
        now = _now()
        norm = _norm(title)
        tags_s = _tags_to_str(tags)
        links_s = json.dumps(links or [])
        emb = self._embed_or_none(title, body)
        existing = self._conn.execute(
            "SELECT id FROM memories WHERE type=? AND title_norm=?", (type, norm)
        ).fetchone()
        if existing:
            mid = existing[0]
            self._conn.execute(
                "UPDATE memories SET title=?, body=?, tags=?, links=?, "
                "updated_at=?, device_id=?, embedding=? WHERE id=?",
                (title, body, tags_s, links_s, now, self._device_id, emb, mid))
            action = "updated"
        else:
            cur = self._conn.execute(
                "INSERT INTO memories(type, title, title_norm, body, tags, links, "
                "created_at, updated_at, device_id, embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (type, title, norm, body, tags_s, links_s, now, now,
                 self._device_id, emb))
            mid = cur.lastrowid
            action = "created"
        self._conn.commit()
        self._sync_now()
        return {"id": mid, "action": action}

    def _fts_ids(self, query, type=None, limit=200):
        match = _fts_query(query)
        if match is None:
            return []
        sql = ("SELECT m.id FROM memories_fts f JOIN memories m ON m.id = f.rowid "
               "WHERE memories_fts MATCH ?")
        params = [match]
        if type is not None:
            sql += " AND m.type = ?"
            params.append(type)
        sql += " ORDER BY rank LIMIT ?"
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
            sql = "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
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
            order = np.argsort(-sims)[:limit]
            return [ids[i] for i in order]
        except Exception:
            return []

    def _hydrate(self, ids) -> list:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, type, title, body, tags, updated_at FROM memories "
            f"WHERE id IN ({placeholders})", ids).fetchall()
        by_id = {r[0]: {"id": r[0], "type": r[1], "title": r[2],
                        "body": r[3], "tags": r[4], "updated_at": r[5]}
                 for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def recall(self, query, type=None, limit=20) -> list:
        if not query or not query.strip():
            return []
        fts_ids = self._fts_ids(query, type)
        vec_ids = self._vector_ids(query, type)
        if vec_ids:
            order = _rrf_fuse([fts_ids, vec_ids])[:limit]
        else:
            order = fts_ids[:limit]
        return self._hydrate(order)

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
        self._conn.commit()
        self._sync_now()
        return {"linked": [id_a, id_b]}

    def forget(self, id) -> dict:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (id,))
        self._conn.commit()
        self._sync_now()
        return {"forgotten": id, "existed": cur.rowcount > 0}

    def boot_index(self) -> str:
        rows = self._conn.execute(
            "SELECT id, type, title FROM memories ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        if not rows:
            return "(no memories yet)"
        return "\n".join(f"[{t}] #{i} {title}" for i, t, title in rows)
