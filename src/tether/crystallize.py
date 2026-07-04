"""crystallize.py - Tier B2 candidate detection (read-time derived view).

Seed-from-peak (explicit + hebbian density) + semantic-expand (membership),
NOT global community detection. Deterministic, local, degrade-never: any
failure returns [] rather than raising. No LLM; the calling agent names the
clusters. See the B2 design doc.
"""

# Tuning constants (module-level, like graph.py's KNN_K/HEBBIAN_*; eval-tuned).
PEAK_KINDS = ("explicit", "hebbian")      # crystallized & semantic excluded from seeding
PEAK_W = {"explicit": 1.0, "hebbian": 0.0}  # W_b = 0 at launch (behavioral parked)
PEAK_FLOOR = 0.5                          # boundness a peak edge must clear to seed
EXPAND_COS = 0.5                          # semantic edge weight (== cosine) to admit a member
EXPAND_HOPS = 1                           # membership hop cap (precision boundary)
MIN_CLUSTER = 3                           # min members to surface
DEDUP_OVERLAP = 0.6                        # basis-recovery fraction to suppress
FALLBACK_Z = 2.0                          # outlier threshold: mean + Z*std of node semantic weight
FALLBACK_MIN_DEG = 2                      # a fallback seed needs at least this many tight neighbors


def _current_ids(conn):
    return {r[0] for r in conn.execute(
        "SELECT id FROM memories WHERE valid_to IS NULL")}


def _peak_edges(conn, current):
    ph = ",".join("?" for _ in PEAK_KINDS)
    rows = conn.execute(
        f"SELECT src, dst, kind, weight FROM edges WHERE kind IN ({ph})",
        PEAK_KINDS).fetchall()
    out = []
    for src, dst, kind, w in rows:
        if src in current and dst in current:
            if PEAK_W.get(kind, 0.0) * w >= PEAK_FLOOR:
                a, b = (src, dst) if src < dst else (dst, src)
                out.append((a, b))
    return sorted(set(out))


def _components(peaks):
    """Union-find over peak edges. Deterministic: smaller id is always the root.
    Returns (root_of, members_by_root)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in peaks:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return find, groups


def _expand(members, conn, current):
    frontier = set(members)
    members = set(members)
    for _ in range(EXPAND_HOPS):
        if not frontier:
            break
        ph = ",".join("?" for _ in frontier)
        rows = conn.execute(
            f"SELECT src, dst FROM edges WHERE kind='semantic' AND weight >= ? "
            f"AND (src IN ({ph}) OR dst IN ({ph}))",
            (EXPAND_COS, *frontier, *frontier)).fetchall()
        new = set()
        for src, dst in rows:
            for a, b in ((src, dst), (dst, src)):
                if a in frontier and b in current and b not in members:
                    new.add(b)
        members |= new
        frontier = new
    return members


def _descriptor(conn, member_ids):
    rows = conn.execute(
        "SELECT title, tags FROM memories WHERE id IN (%s)"
        % ",".join("?" for _ in member_ids), member_ids).fetchall()
    titles = "; ".join(r[0] for r in rows if r[0])
    tags = sorted({t for r in rows for t in (r[1] or "").split(",") if t})
    return titles + (f" [tags: {', '.join(tags)}]" if tags else "")


def _principle_bases(conn, current):
    """{principle_id: frozenset(source_ids)} from directional crystallized edges,
    restricted to still-current memories. on_forget deletes a node's edges, but a
    soft-forgotten (valid_to set, edges retained) principle or source must not
    suppress a live candidate via stale basis-recovery overlap."""
    bases = {}
    for src, dst in conn.execute(
            "SELECT src, dst FROM edges WHERE kind='crystallized'").fetchall():
        if src in current and dst in current:
            bases.setdefault(src, set()).add(dst)
    return {p: frozenset(s) for p, s in bases.items()}


def _covered(member_ids, bases):
    members = set(member_ids)
    for sources in bases.values():
        if sources and len(members & sources) / len(sources) >= DEDUP_OVERLAP:
            return True
    return False


def _dismissed(conn):
    try:
        return {(r[0], r[1]) for r in conn.execute(
            "SELECT src, dst FROM crystallize_dismissed").fetchall()}
    except Exception:
        return set()


def _semantic_outlier_seeds(conn, current):
    """Nodes whose summed semantic neighborhood weight is an outlier vs the
    store's own baseline. Computed only over materialized semantic edges
    (never all-pairs)."""
    node_w = {}
    for src, dst, w in conn.execute(
            "SELECT src, dst, weight FROM edges WHERE kind='semantic'").fetchall():
        if src in current and dst in current:
            node_w.setdefault(src, []).append((dst, w))
            node_w.setdefault(dst, []).append((src, w))
    if len(node_w) < 3:
        return []
    strengths = [sum(w for _, w in nbrs) for nbrs in node_w.values()]
    mean = sum(strengths) / len(strengths)
    var = sum((s - mean) ** 2 for s in strengths) / len(strengths)
    std = var ** 0.5
    threshold = mean + FALLBACK_Z * std
    seeds = []
    for node, nbrs in sorted(node_w.items()):
        strong = [d for d, w in nbrs if w >= EXPAND_COS]
        if sum(w for _, w in nbrs) >= threshold and len(strong) >= FALLBACK_MIN_DEG:
            seeds.append((node, {node, *strong}))
    return seeds


def candidates(conn, embedder=None, fallback=False):
    """Candidate clusters (peak-seeded, semantic-expanded; dedup + dismissed suppression; optional gated cold-start fallback)."""
    try:
        current = _current_ids(conn)
        if not current:
            return []
        peaks = _peak_edges(conn, current)
        bases = _principle_bases(conn, current)
        dismissed = _dismissed(conn)
        out = []
        if peaks:
            find, groups = _components(peaks)
            peaks_by_root = {}
            for a, b in peaks:
                peaks_by_root.setdefault(find(a), []).append((a, b))
            for root, seed in sorted(groups.items()):
                members = _expand(seed, conn, current)
                if len(members) < MIN_CLUSTER:
                    continue
                member_ids = sorted(members)
                root_peaks = sorted(peaks_by_root.get(root, []))
                if root_peaks[0] in dismissed:           # dismissed nucleus
                    continue
                if _covered(member_ids, bases):          # basis already crystallized
                    continue
                out.append({
                    "peak_key": root_peaks[0],
                    "member_ids": member_ids,
                    "member_titles": [r[0] for r in conn.execute(
                        "SELECT title FROM memories WHERE id IN (%s) ORDER BY id"
                        % ",".join("?" for _ in member_ids), member_ids).fetchall()],
                    "why": [(a, b, "peak") for a, b in root_peaks],
                    "descriptor": _descriptor(conn, member_ids),
                })
        if fallback:
            seen = {m for c in out for m in c["member_ids"]}
            for node, members in _semantic_outlier_seeds(conn, current):
                members = _expand(members, conn, current)
                member_ids = sorted(members)
                if len(member_ids) < MIN_CLUSTER:
                    continue
                if any(m in seen for m in member_ids):     # don't duplicate peak clusters
                    continue
                if _covered(member_ids, bases):
                    continue
                out.append({
                    "peak_key": (node, node),              # semantic seed: self-key
                    "member_ids": member_ids,
                    "member_titles": [r[0] for r in conn.execute(
                        "SELECT title FROM memories WHERE id IN (%s) ORDER BY id"
                        % ",".join("?" for _ in member_ids), member_ids).fetchall()],
                    "why": [("semantic-density", node, "fallback")],
                    "descriptor": _descriptor(conn, member_ids),
                })
                seen.update(member_ids)
        return out
    except Exception:
        return []
