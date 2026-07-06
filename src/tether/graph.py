"""graph.py - the association layer.

A usage graph over memories: edges from embedding geometry (semantic kNN),
explicit links, and learned co-recall (Hebbian), plus an ephemeral session
working set. Owns the `edges` and `session_members` SQL. Spreading activation
(`spread`) turns a set of seed memories into their connected neighborhood.

100% local and deterministic; no LLM, no network. Every operation degrades to
a no-op rather than raising, so the memory layer never breaks the agent.
"""

import uuid
from datetime import datetime, timedelta, timezone

KNN_K = 8
HOP_DECAY = 0.4
EPSILON = 1e-4
KIND_W = {"semantic": 1.0, "explicit": 1.2, "hebbian": 1.0, "crystallized": 1.2}
SESSION_DECAY = 0.5
SESSION_GAP_SECONDS = 1800
SESSION_TTL_ACTIVATION = 0.05
# Horizon for the periodic session_members sweep (#48): well above
# SESSION_GAP_SECONDS so an infrequently-touched but still-live explicit
# session isn't swept out from under its caller, while still bounding
# long-term growth from sessions that are genuinely never touched again.
SESSION_SWEEP_HORIZON_SECONDS = 24 * 60 * 60
HEBBIAN_INCREMENT = 0.5
HEBBIAN_CAP = 5.0
HEBBIAN_TOP_M = 8
# Rank-weighted session bump (B1): the id at rank r of a recall's result list
# is bumped by HEBBIAN_BUMP_DECAY**r instead of a uniform +1.0. A recall's
# working-set contribution then reflects what the recall was ABOUT (its top
# hits), not everything it happened to return — a returned list is mostly
# padding, and bumping all of it uniformly (a) makes activation carry no
# information (mass ties, broken by memory_id, so low-id memories permanently
# squat the Hebbian top-M) and (b) wires dense spurious cliques at cap.
# HEBBIAN_BUMP_DECAY=1.0 with HEBBIAN_WIRE_FLOOR=0.0 restores the old rule
# exactly. HEBBIAN_WIRE_FLOOR gates pair-wiring to members that are genuinely
# active (recently a top-3 subject), so straggler noise in a sparse session
# does not get wired merely for surviving the TTL.
HEBBIAN_BUMP_DECAY = 0.5
HEBBIAN_WIRE_FLOOR = 0.25
# True = learn from the protected direct-hit head (B1 default); False = learn
# from the full returned order (pre-B1 behavior).
HEBBIAN_LEARN_FROM_HEAD = True

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
CREATE INDEX IF NOT EXISTS idx_session_members_updated ON session_members(updated_at);
CREATE TABLE IF NOT EXISTS crystallize_dismissed (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    PRIMARY KEY (src, dst)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Graph:
    def __init__(self, conn, enabled=True):
        self._conn = conn
        self.enabled = enabled
        # Established once per Graph (i.e. once per server process/startup, see
        # server.py's lazily-built singleton store); scopes the implicit
        # session bucket per-process so concurrent unrelated callers sharing
        # the same underlying DB don't get bucketed into one session (#53).
        self._process_id = uuid.uuid4().hex[:12]

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

    def unprime(self, mid) -> None:
        """Drop `mid` from every session's primed working set. Edges are left
        alone (a soft-archived node just stops matching valid_to IS NULL in
        the reads that matter); only the primed-session-context path needs
        scrubbing, mirroring what on_forget does for hard deletes."""
        try:
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

    def dismiss_peak(self, a, b) -> None:
        try:
            src, dst = (a, b) if a < b else (b, a)
            self._conn.execute(
                "INSERT OR IGNORE INTO crystallize_dismissed(src, dst) VALUES(?,?)",
                (src, dst))
            self._conn.commit()
        except Exception:
            return

    def dismissed_peaks(self) -> set:
        try:
            return {(r[0], r[1]) for r in self._conn.execute(
                "SELECT src, dst FROM crystallize_dismissed").fetchall()}
        except Exception:
            return set()

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
        # Per-process meta keys: two Graph instances (e.g. two server
        # processes sharing one DB) each track their own gap/continuation
        # state instead of racing over one shared global bucket (#53).
        session_key = f"assoc_session:{self._process_id}"
        activity_key = f"assoc_last_activity:{self._process_id}"
        if session:
            sid = str(session)
        else:
            last = meta_get(activity_key)
            cur = meta_get(session_key)
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
                sid = f"{now_iso}:{self._process_id}"
        meta_set(session_key, sid)
        meta_set(activity_key, now_iso)
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
            for rank, mid in enumerate(ordered_ids):
                bump = HEBBIAN_BUMP_DECAY ** rank
                if bump < SESSION_TTL_ACTIVATION:
                    break  # geometric: every later rank is smaller still
                self._conn.execute(
                    "INSERT INTO session_members(session_id, memory_id, activation, updated_at) "
                    "VALUES (?,?,?,?) ON CONFLICT(session_id, memory_id) "
                    "DO UPDATE SET activation = session_members.activation + ?, updated_at=?",
                    (session_id, mid, bump, now, bump, now))
            active = [r[0] for r in self._conn.execute(
                "SELECT memory_id FROM session_members WHERE session_id=? "
                "AND activation >= ? ORDER BY activation DESC, memory_id LIMIT ?",
                (session_id, HEBBIAN_WIRE_FLOOR, HEBBIAN_TOP_M)).fetchall()]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    self._upsert_edge(active[i], active[j], "hebbian",
                                      HEBBIAN_INCREMENT, now, mode="add")
            self._conn.execute(
                "DELETE FROM session_members WHERE activation < ?",
                (SESSION_TTL_ACTIVATION,))
        except Exception:
            return

    def sweep_stale_session_members(self, horizon_seconds=SESSION_SWEEP_HORIZON_SECONDS) -> int:
        """Periodic maintenance sweep, independent of any specific session
        being touched. touch_session's decay UPDATE refreshes updated_at for
        every row of the session id it's given, so a row's updated_at is that
        session's last-activity time; a session that's simply never touched
        again (abandoned) leaves its rows' updated_at frozen in the past.
        Removes rows whose session has been idle longer than horizon_seconds,
        regardless of their current activation (#48). Returns rows removed;
        never raises."""
        try:
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(seconds=horizon_seconds)).isoformat()
            cur = self._conn.execute(
                "DELETE FROM session_members WHERE updated_at < ?", (cutoff,))
            return cur.rowcount
        except Exception:
            return 0
