"""store.py - the memory store. Owns ALL SQL.

One table (`memories`) plus an external-content FTS5 index kept in sync by
triggers. The four verbs and the boot index are the only public surface;
nothing outside this module speaks SQL.
"""

import json
import re
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


class Store:
    def __init__(self, conn, device_id: str, sync_now):
        self._conn = conn
        self._device_id = device_id
        self._sync_now = sync_now

    def migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def remember(self, type, title, body, tags=None, links=None) -> dict:
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got {type!r}")
        now = _now()
        norm = _norm(title)
        tags_s = _tags_to_str(tags)
        links_s = json.dumps(links or [])
        existing = self._conn.execute(
            "SELECT id FROM memories WHERE type=? AND title_norm=?", (type, norm)
        ).fetchone()
        if existing:
            mid = existing[0]
            self._conn.execute(
                "UPDATE memories SET title=?, body=?, tags=?, links=?, "
                "updated_at=?, device_id=? WHERE id=?",
                (title, body, tags_s, links_s, now, self._device_id, mid))
            action = "updated"
        else:
            cur = self._conn.execute(
                "INSERT INTO memories(type, title, title_norm, body, tags, links, "
                "created_at, updated_at, device_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (type, title, norm, body, tags_s, links_s, now, now, self._device_id))
            mid = cur.lastrowid
            action = "created"
        self._conn.commit()
        self._sync_now()
        return {"id": mid, "action": action}

    def recall(self, query, type=None, limit=20) -> list:
        match = _fts_query(query)
        if match is None:
            return []
        sql = ("SELECT m.id, m.type, m.title, m.body, m.tags, m.updated_at "
               "FROM memories_fts f JOIN memories m ON m.id = f.rowid "
               "WHERE memories_fts MATCH ?")
        params = [match]
        if type is not None:
            sql += " AND m.type = ?"
            params.append(type)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {"id": r[0], "type": r[1], "title": r[2],
             "body": r[3], "tags": r[4], "updated_at": r[5]}
            for r in rows
        ]

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
