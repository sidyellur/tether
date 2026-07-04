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
KIND_W = {"semantic": 1.0, "explicit": 1.2, "hebbian": 1.0, "crystallized": 1.2}
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

    def on_crystallize(self, principle_id, source_ids) -> None:
        """Directional crystallized edges principle->source (the one kind NOT
        canonicalized, so provenance is recoverable for dedup). Read paths
        (_neighbors/spread/degree_map) read both endpoints, so activation stays
        bidirectional. No-op when disabled; never raises."""
        if not self.enabled:
            return
        try:
            now = _now()
            for src in source_ids:
                if src == principle_id:
                    continue
                self._conn.execute(
                    "INSERT INTO edges(src, dst, kind, weight, updated_at) "
                    "VALUES (?,?,'crystallized',1.0,?) "
                    "ON CONFLICT(src, dst, kind) DO UPDATE SET updated_at=excluded.updated_at",
                    (principle_id, src, now))
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

    def degree_map(self, kinds=("explicit", "hebbian", "crystallized")) -> dict:
        """Behavioral weighted degree for every current memory (semantic edges
        excluded by default). Emits explicit 0.0 for isolated current nodes -
        forgetting needs the degree-0 set, which a pure edge-scan can't produce.
        Only edges between two current nodes count. Never raises."""
        try:
            current = {r[0] for r in self._conn.execute(
                "SELECT id FROM memories WHERE valid_to IS NULL").fetchall()}
            deg = {mid: 0.0 for mid in current}
            if not kinds:
                return deg
            ph = ",".join("?" for _ in kinds)
            rows = self._conn.execute(
                f"SELECT src, dst, weight FROM edges WHERE kind IN ({ph})",
                tuple(kinds)).fetchall()
            for src, dst, w in rows:
                if src in current and dst in current:
                    deg[src] += w
                    deg[dst] += w
            return deg
        except Exception:
            return {}

    def _neighbors(self, node, type=None):
        rows = self._conn.execute(
            "SELECT CASE WHEN src=? THEN dst ELSE src END, kind, weight "
            "FROM edges WHERE src=? OR dst=?", (node, node, node)).fetchall()
        if not rows:
            return []
        agg = {}
        for nbr, kind, w in rows:
            agg.setdefault(nbr, {})[kind] = w
        ids = list(agg)
        ph = ",".join("?" for _ in ids)
        params = list(ids)
        sql = f"SELECT id, type FROM memories WHERE id IN ({ph}) AND valid_to IS NULL"
        if type is not None:
            sql += " AND type = ?"
            params.append(type)
        valid = {r[0]: r[1] for r in self._conn.execute(sql, params).fetchall()}
        out = []
        for nbr in sorted(agg):
            if nbr not in valid:
                continue
            kinds = agg[nbr]
            blended = sum(KIND_W.get(k, 0.0) * w for k, w in kinds.items())
            if blended <= 0:
                continue
            dominant = max(kinds, key=lambda k: KIND_W.get(k, 0.0) * kinds[k])
            out.append((nbr, blended, dominant))
        return out

    def spread(self, seed_activation, budget, type=None):
        activation = dict(seed_activation)
        if not self.enabled or budget <= 0 or not activation:
            return activation, {}
        receipts = {}
        depth = {mid: 0 for mid in seed_activation}
        fired = set()
        expansions = 0
        while expansions < budget:
            candidates = [(a, mid) for mid, a in activation.items()
                          if mid not in fired and a >= EPSILON]
            if not candidates:
                break
            candidates.sort(key=lambda x: (-x[0], x[1]))
            a, node = candidates[0]
            fired.add(node)
            expansions += 1
            for nbr, w, kind in self._neighbors(node, type):
                transmit = a * w * HOP_DECAY
                if transmit < EPSILON:
                    continue
                activation[nbr] = activation.get(nbr, 0.0) + transmit
                d = depth.get(node, 0) + 1
                if nbr not in depth or d < depth[nbr]:
                    depth[nbr] = d
                if nbr not in seed_activation:
                    prev = receipts.get(nbr)
                    if prev is None or transmit > prev["_t"]:
                        receipts[nbr] = {"from": node, "kind": kind,
                                         "w": round(w, 3), "hops": depth[nbr], "_t": transmit}
        for r in receipts.values():
            r.pop("_t", None)
        return activation, receipts

    def resolve_session(self, session, meta_get, meta_set) -> str:
        now_iso = _now()
        if session:
            sid = str(session)
        else:
            last = meta_get("assoc_last_activity")
            cur = meta_get("assoc_session")
            gap = None
            if last is not None:
                try:
                    gap = (datetime.fromisoformat(now_iso)
                           - datetime.fromisoformat(last)).total_seconds()
                except Exception:
                    gap = None
            if cur and gap is not None and gap <= SESSION_GAP_SECONDS:
                sid = cur
            else:
                sid = now_iso
        meta_set("assoc_session", sid)
        meta_set("assoc_last_activity", now_iso)
        return sid

    def session_activation(self, session_id) -> dict:
        try:
            return {r[0]: r[1] for r in self._conn.execute(
                "SELECT memory_id, activation FROM session_members WHERE session_id=?",
                (session_id,)).fetchall()}
        except Exception:
            return {}

    def touch_session(self, session_id, ordered_ids) -> None:
        try:
            now = _now()
            self._conn.execute(
                "UPDATE session_members SET activation = activation * ?, updated_at=? "
                "WHERE session_id=?", (SESSION_DECAY, now, session_id))
            for mid in ordered_ids:
                self._conn.execute(
                    "INSERT INTO session_members(session_id, memory_id, activation, updated_at) "
                    "VALUES (?,?,?,?) ON CONFLICT(session_id, memory_id) "
                    "DO UPDATE SET activation = session_members.activation + 1.0, updated_at=?",
                    (session_id, mid, 1.0, now, now))
            active = [r[0] for r in self._conn.execute(
                "SELECT memory_id FROM session_members WHERE session_id=? "
                "ORDER BY activation DESC, memory_id LIMIT ?",
                (session_id, HEBBIAN_TOP_M)).fetchall()]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    self._upsert_edge(active[i], active[j], "hebbian",
                                      HEBBIAN_INCREMENT, now, mode="add")
            self._conn.execute(
                "DELETE FROM session_members WHERE activation < ?",
                (SESSION_TTL_ACTIVATION,))
        except Exception:
            return
