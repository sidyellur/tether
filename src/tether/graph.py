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
CREATE INDEX IF NOT EXISTS idx_edges_updated ON edges(updated_at);
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
        # #112: degree_map() reads every explicit/hebbian/crystallized edge
        # plus every current memory id - called from boot_index() (once the
        # store is above boot_index_cap) and from every forgetting sweep.
        # Memoized on a cheap signature keyed by `kinds` too, since callers
        # pass different kind-tuples and each needs its own cached answer.
        self._degree_sig = None
        self._degree_cache = None
        # #112 follow-up: bumped by Store._commit() on every commit, via
        # _touch(). Replaces MAX(updated_at) for detecting this process's own
        # writes in _degree_signature - a wall-clock string comparison can tie
        # (see Store._store_signature's docstring for the #68-shaped bug this
        # caused on Windows), a monotonic counter can't.
        self._write_seq = 0
        # The implicit recall session for this process (see resolve_session):
        # its id and the time of the last recall.
        self._session_id = None
        self._last_activity = None

    def _touch(self) -> None:
        self._write_seq += 1

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

    # Rows per multi-row INSERT in _upsert_edges: 5 bound values each, kept
    # well under SQLite's oldest default limit of 999 parameters.
    _EDGE_BATCH = 150

    def _upsert_edges(self, rows, now, mode="max") -> None:
        """Upsert many edges in one statement per _EDGE_BATCH rows.

        `rows` is an iterable of (a, b, kind, weight); each is canonicalized
        and merged exactly as _upsert_edge would, in order (a pair repeated
        within `rows` conflicts with its own earlier row, so "add" adds twice
        and "max" keeps the larger, same as two separate calls).

        One statement instead of one per edge matters on a sync replica,
        where each statement inside a write transaction is its own trip to
        the primary: a recall wires up to 28 Hebbian pairs and a remember 8
        semantic neighbours.
        """
        values = []
        for a, b, kind, weight in rows:
            if a == b:
                continue
            src, dst = (a, b) if a < b else (b, a)
            values.append((src, dst, kind, weight, now))
        if mode == "add":
            expr = "weight = min(?, edges.weight + excluded.weight)"
            tail = (HEBBIAN_CAP,)
        else:  # max
            expr = "weight = max(edges.weight, excluded.weight)"
            tail = ()
        for i in range(0, len(values), self._EDGE_BATCH):
            chunk = values[i:i + self._EDGE_BATCH]
            self._conn.execute(
                "INSERT INTO edges(src, dst, kind, weight, updated_at) VALUES "
                + ",".join("(?,?,?,?,?)" for _ in chunk)
                + f" ON CONFLICT(src, dst, kind) DO UPDATE SET {expr}, "
                "updated_at=excluded.updated_at",
                tuple(v for row in chunk for v in row) + tail)

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

    def on_remember(self, mid, emb_blob, matrix=None, replace=False) -> None:
        """Wire semantic kNN edges for `mid`.

        `matrix` is an optional (ids, mat, types) triple the caller has already
        built - the Store passes its cached one (#61) so a single remember()
        doesn't deserialize every embedding in the store twice (once for the
        near-duplicate probe, once here). Falls back to its own scan when the
        caller has nothing to share, so Graph stays usable standalone.
        """
        if not self.enabled:
            return
        if emb_blob is None:
            return
        try:
            import numpy as np

            q = np.frombuffer(emb_blob, dtype="<f4")
            if matrix is not None and matrix[1] is not None:
                ids, mat = matrix[0], matrix[1]
            else:
                rows = self._conn.execute(
                    "SELECT id, embedding FROM memories "
                    "WHERE embedding IS NOT NULL AND valid_to IS NULL").fetchall()
                if not rows:
                    return
                ids = [r[0] for r in rows]
                mat = np.frombuffer(b"".join(r[1] for r in rows),
                                    dtype="<f4").reshape(len(ids), -1)
            sims = mat @ q
            # `mid` itself is in the shared matrix (it was just written), and
            # self-similarity is 1.0 - so it would win every kNN. _upsert_edge
            # already refuses a self-edge, but excluding it here keeps it from
            # consuming one of the k slots.
            #
            # Only the top KNN_K matter, so select them rather than sorting
            # (and then walking, in Python) the whole store: argpartition
            # pulls the KNN_K + 1 best in O(n) - one spare because `mid` may
            # be among them - and only those few are ordered. A full argsort
            # plus a per-row list comprehension here was ~70% of a remember()
            # at 8k memories and made a backfill quadratic in Python work.
            # Ties order by row index (ascending id), so the result is
            # deterministic.
            k = min(KNN_K + 1, len(ids))
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.lexsort((top, -sims[top]))]
            order = [int(i) for i in top if ids[int(i)] != mid]
            neighbours = [(mid, ids[i], "semantic", float(sims[i]))
                          for i in order[:KNN_K] if float(sims[i]) > 0]
            if replace:
                # `mid` was rewritten: its semantic edges were weighted for
                # text that is gone. Upserting on top kept every old edge at
                # full weight forever, so a memory rewritten from cars to
                # pizza still surfaced for "automobile". Start its edges
                # over - but an edge is shared with the memory at the other
                # end, which may have chosen `mid` as ITS neighbour, so a
                # former neighbour is kept, at its new similarity, when it is
                # still at least as close as the weakest of the new top-k.
                former = {r[0] for r in self._conn.execute(
                    "SELECT CASE WHEN src=? THEN dst ELSE src END FROM edges "
                    "WHERE kind='semantic' AND (src=? OR dst=?)",
                    (mid, mid, mid)).fetchall()}
                chosen = {n[1] for n in neighbours}
                floor = min((n[3] for n in neighbours), default=0.0)
                index = {ids[i]: i for i in range(len(ids))} if former - chosen else {}
                for other in sorted(former - chosen):
                    i = index.get(other)
                    if i is not None and float(sims[i]) > 0 and float(sims[i]) >= floor:
                        neighbours.append((mid, other, "semantic", float(sims[i])))
                self._conn.execute(
                    "DELETE FROM edges WHERE kind='semantic' AND (src=? OR dst=?)",
                    (mid, mid))
            self._upsert_edges(neighbours, _now(), mode="max")
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

    def backfill_semantic(self, matrix=None) -> None:
        """Wire semantic edges for the whole store.

        Builds the embedding matrix ONCE and hands the same one to every
        on_remember (#61). Previously each of the n calls re-read and
        re-deserialized all n embeddings, making a backfill quadratic in I/O
        on top of the unavoidable quadratic in similarity math - the dominant
        cost of first boot on a large store.
        """
        if not self.enabled:
            return
        try:
            import numpy as np

            if matrix is not None and matrix[1] is not None:
                ids, mat = matrix[0], matrix[1]
            else:
                rows = self._conn.execute(
                    "SELECT id, embedding FROM memories "
                    "WHERE embedding IS NOT NULL AND valid_to IS NULL "
                    "ORDER BY id").fetchall()
                if not rows:
                    return
                ids = [r[0] for r in rows]
                mat = np.frombuffer(b"".join(r[1] for r in rows),
                                    dtype="<f4").reshape(len(ids), -1)
            shared = (ids, mat, None)
            # A wholesale re-wire starts clean: every vector is being placed
            # afresh, so edges from the previous vectors are stale.
            self._conn.execute("DELETE FROM edges WHERE kind='semantic'")
            for i, mid in enumerate(ids):
                self.on_remember(mid, mat[i].tobytes(), matrix=shared)
        except Exception:
            return

    def backfill_explicit(self, pairs) -> None:
        if not self.enabled:
            return
        try:
            self._upsert_edges(((a, b, "explicit", 1.0) for a, b in pairs),
                               _now(), mode="max")
        except Exception:
            return

    def _degree_signature(self):
        """Cheap fingerprint of "could degree_map()'s answer have changed?":
        `_write_seq` (this process's own on_link/on_remember/touch_session/
        unprime writes, bumped by Store._commit()'s call to _touch() on the
        very next call) paired with `PRAGMA data_version` (any OTHER
        connection's commit - see Store._data_version, same pragma, same
        connection).

        #128: previously used an edge-count/current-memory-count SELECT
        instead of data_version for the foreign-connection half - but those
        counts don't change on an UPDATE (a foreign connection bumping an
        existing edge's weight, or archiving a memory via valid_to without
        also touching `edges`), so that write went undetected and
        degree_map() kept serving a stale answer. data_version catches any
        commit from another connection regardless of statement shape.

        Used MAX(updated_at) instead of _write_seq originally, but that's a
        wall-clock comparison that can tie within one clock tick (see
        Store._store_signature's docstring) - swapped for the same reason."""
        try:
            data_version = self._conn.execute("PRAGMA data_version").fetchone()[0]
        except Exception:
            data_version = None
        return (self._write_seq, data_version)

    def degree_map(self, kinds=("explicit", "hebbian", "crystallized")) -> dict:
        """Behavioral weighted degree for every current memory (semantic edges
        excluded by default). Emits explicit 0.0 for isolated current nodes -
        forgetting needs the degree-0 set, which a pure edge-scan can't produce.
        Only edges between two current nodes count. Never raises.

        Memoized (#112) on a cheap change signature; `kinds` is part of the
        cache key since callers (boot_index's curation vs. the forgetting
        sweep) may pass different kind-tuples."""
        try:
            sig = (self._degree_signature(), kinds)
            if self._degree_cache is not None and sig == self._degree_sig:
                return self._degree_cache
            deg = self._degree_map_impl(kinds)
            self._degree_sig = sig
            self._degree_cache = deg
            return deg
        except Exception:
            return {}

    def _degree_map_impl(self, kinds) -> dict:
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
        valid = {r[0]: r[1] for r in self._conn.execute(sql, tuple(params)).fetchall()}
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

    def resolve_session(self, session) -> str:
        """The session id a recall belongs to: the caller's explicit one, or
        this process's implicit bucket, which continues until
        SESSION_GAP_SECONDS pass without a recall.

        The bucket lives in memory. It used to be two `meta` rows keyed by
        this Graph's random `_process_id` (#53), which no other process could
        ever read - so persisting them bought nothing, cost two writes at the
        start of every recall, and left two rows behind per server start.
        Worse on a sync replica: that first write opened a remote
        transaction, after which every read in the recall (the session
        lookup, each spread expansion) was forwarded to the primary too.
        Only ever called under the Store's lock, so plain attributes do.
        """
        now_iso = _now()
        if session:
            sid = str(session)
        else:
            gap = None
            if self._last_activity is not None:
                try:
                    gap = (datetime.fromisoformat(now_iso)
                           - datetime.fromisoformat(self._last_activity)).total_seconds()
                except Exception:
                    gap = None
            if (self._session_id and gap is not None
                    and gap <= SESSION_GAP_SECONDS):
                sid = self._session_id
            else:
                sid = f"{now_iso}:{self._process_id}"
        self._session_id = sid
        self._last_activity = now_iso
        return sid

    def session_activation(self, session_id) -> dict:
        try:
            return {r[0]: r[1] for r in self._conn.execute(
                "SELECT memory_id, activation FROM session_members WHERE session_id=?",
                (session_id,)).fetchall()}
        except Exception:
            return {}

    def touch_session(self, session_id, ordered_ids) -> None:
        """Record a recall in its session: decay the working set, bump the
        recalled ids by rank, wire the most active members to each other
        (Hebbian), and drop members that have decayed below the TTL.

        The session is read ONCE, up front, and the resulting activations
        are worked out here rather than re-read mid-transaction; every write
        is then a single statement. The arithmetic is the same IEEE double
        multiply and add SQLite performs in the UPDATE/upsert below, so the
        values (and therefore which members clear HEBBIAN_WIRE_FLOOR, in
        what order) match what a re-read would return. That keeps a recall's
        learning to at most four statements, none of them a read - on a sync
        replica each statement inside the write transaction is a trip to the
        primary, and this used to be up to 36 of them.
        """
        try:
            now = _now()
            activation = {r[0]: r[1] * SESSION_DECAY for r in self._conn.execute(
                "SELECT memory_id, activation FROM session_members "
                "WHERE session_id=?", (session_id,)).fetchall()}
            had_members = bool(activation)
            bumps = []
            for rank, mid in enumerate(ordered_ids):
                bump = HEBBIAN_BUMP_DECAY ** rank
                if bump < SESSION_TTL_ACTIVATION:
                    break  # geometric: every later rank is smaller still
                bumps.append((session_id, mid, bump, now))
                activation[mid] = activation.get(mid, 0.0) + bump
            if had_members:
                self._conn.execute(
                    "UPDATE session_members SET activation = activation * ?, updated_at=? "
                    "WHERE session_id=?", (SESSION_DECAY, now, session_id))
            if bumps:
                self._conn.execute(
                    "INSERT INTO session_members(session_id, memory_id, activation, updated_at) "
                    "VALUES " + ",".join("(?,?,?,?)" for _ in bumps)
                    + " ON CONFLICT(session_id, memory_id) DO UPDATE SET "
                    "activation = session_members.activation + excluded.activation, "
                    "updated_at = excluded.updated_at",
                    tuple(v for row in bumps for v in row))
            active = sorted((m for m, a in activation.items() if a >= HEBBIAN_WIRE_FLOOR),
                            key=lambda m: (-activation[m], m))[:HEBBIAN_TOP_M]
            self._upsert_edges(
                ((active[i], active[j], "hebbian", HEBBIAN_INCREMENT)
                 for i in range(len(active)) for j in range(i + 1, len(active))),
                now, mode="add")
            if any(a < SESSION_TTL_ACTIVATION for a in activation.values()):
                self._conn.execute(
                    "DELETE FROM session_members WHERE session_id=? AND activation < ?",
                    (session_id, SESSION_TTL_ACTIVATION))
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
            # Look before deleting: the usual answer is "nothing stale", and
            # an indexed read costs nothing, whereas a DELETE opens a write
            # transaction (a remote one on a sync replica) even when it
            # matches no rows.
            if self._conn.execute(
                    "SELECT 1 FROM session_members WHERE updated_at < ? LIMIT 1",
                    (cutoff,)).fetchone() is None:
                return 0
            cur = self._conn.execute(
                "DELETE FROM session_members WHERE updated_at < ?", (cutoff,))
            return cur.rowcount
        except Exception:
            return 0
