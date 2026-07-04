"""graph.py - the association layer.

A usage graph over memories: edges from embedding geometry (semantic kNN),
explicit links, and learned co-recall (Hebbian), plus an ephemeral session
working set. Owns the `edges` and `session_members` SQL. Spreading activation
(`spread`) turns a set of seed memories into their connected neighborhood.

100% local and deterministic; no LLM, no network. Every operation degrades to
a no-op rather than raising, so the memory layer never breaks the agent.
"""

from datetime import datetime, timezone

KNN_K = 8
HOP_DECAY = 0.4
EPSILON = 1e-4
KIND_W = {"semantic": 1.0, "explicit": 1.2, "hebbian": 1.0}
SESSION_DECAY = 0.5
SESSION_GAP_SECONDS = 1800
SESSION_TTL_ACTIVATION = 0.05
HEBBIAN_INCREMENT = 0.5
HEBBIAN_CAP = 5.0
HEBBIAN_TOP_M = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    src        INTEGER NOT NULL,
    dst        INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS session_members (
    session_id  TEXT NOT NULL,
    memory_id   INTEGER NOT NULL,
    activation  REAL NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, memory_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Graph:
    def __init__(self, conn, enabled=True):
        self._conn = conn
        self.enabled = enabled

    def migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _upsert_edge(self, a, b, kind, weight, now, mode="max") -> None:
        if a == b:
            return
        src, dst = (a, b) if a < b else (b, a)
        if mode == "add":
            expr = "weight = min(?, edges.weight + ?)"
            params = (src, dst, kind, weight, now, HEBBIAN_CAP, weight)
        else:  # max
            expr = "weight = max(edges.weight, excluded.weight)"
            params = (src, dst, kind, weight, now)
        self._conn.execute(
            f"INSERT INTO edges(src, dst, kind, weight, updated_at) "
            f"VALUES (?,?,?,?,?) "
            f"ON CONFLICT(src, dst, kind) DO UPDATE SET {expr}, updated_at=excluded.updated_at",
            params)

    def on_forget(self, mid) -> None:
        try:
            self._conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (mid, mid))
            self._conn.execute("DELETE FROM session_members WHERE memory_id=?", (mid,))
        except Exception:
            pass

    def on_remember(self, mid, emb_blob) -> None:
        if not self.enabled or emb_blob is None:
            return
        try:
            import numpy as np

            q = np.frombuffer(emb_blob, dtype="<f4")
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories "
                "WHERE embedding IS NOT NULL AND valid_to IS NULL AND id != ?",
                (mid,)).fetchall()
            if not rows:
                return
            ids = [r[0] for r in rows]
            mat = np.frombuffer(b"".join(r[1] for r in rows),
                                dtype="<f4").reshape(len(ids), -1)
            sims = mat @ q
            k = min(KNN_K, len(ids))
            top = np.argsort(-sims)[:k]
            now = _now()
            for i in top:
                w = float(sims[i])
                if w <= 0:
                    continue
                self._upsert_edge(mid, ids[int(i)], "semantic", w, now, mode="max")
        except Exception:
            return

    def on_link(self, id_a, id_b) -> None:
        if not self.enabled:
            return
        try:
            self._upsert_edge(id_a, id_b, "explicit", 1.0, _now(), mode="max")
        except Exception:
            return

    def backfill_semantic(self) -> None:
        if not self.enabled:
            return
        try:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories "
                "WHERE embedding IS NOT NULL AND valid_to IS NULL").fetchall()
            for mid, blob in rows:
                self.on_remember(mid, blob)
        except Exception:
            return

    def backfill_explicit(self, pairs) -> None:
        if not self.enabled:
            return
        try:
            now = _now()
            for a, b in pairs:
                self._upsert_edge(a, b, "explicit", 1.0, now, mode="max")
        except Exception:
            return
