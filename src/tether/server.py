#!/usr/bin/env python3
"""server.py - the MCP server. The agent-facing edge.

Four verbs over a persistent SQLite-backed memory store, plus an auto-loaded
boot index exposed as an MCP resource. The store is built lazily on first use
so importing the module (and listing tools) never touches the filesystem.

Run it as an MCP stdio server:

    tether-memory               # installed entry point
    python -m tether.server     # or as a module
"""

import json
import threading

try:
    # mcp >= 2.0 (released 2026-07-28) renamed FastMCP -> MCPServer and moved
    # it to mcp.server.mcpserver, with no back-compat alias (#69). The surface
    # tether uses - constructor, @tool(), @resource(uri), run() - is identical
    # across both, so a plain import shim covers both majors.
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - exercised by whichever mcp is installed
    from mcp.server.fastmcp import FastMCP as MCPServer

from . import config
from .store import Store
from .sync import open_connection

mcp = MCPServer("tether")

_store = None
_sync_mode = None
# #83: tool calls run on worker threads, so two first calls could otherwise
# both build a store (two connections, two migrations, two model loads).
_store_lock = threading.Lock()


def _get_store() -> Store:
    global _store, _sync_mode
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        path = config.db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn, sync_now, _sync_mode = open_connection(path, config.sync_config())
        # Only an actual replica connection has somewhere to degrade to on a
        # write failure (#44 - a mid-session network drop should degrade
        # gracefully instead of raising out of remember/link/forget); local
        # and already-degraded connections have nothing further to fall back
        # to. on_degrade updates the status resource's sync_mode (#51) so a
        # later mid-session degrade doesn't leave it reporting stale "replica".
        degrade_db_path = path if _sync_mode == "replica" else None

        def _mark_degraded():
            global _sync_mode
            _sync_mode = "degraded"

        embedder = None
        if config.semantic_enabled():
            from . import embed
            embedder = embed.get_embedder(config.embedding_model())
        store = Store(conn, device_id=config.device_id(), sync_now=sync_now,
                      embedder=embedder, author=config.author(),
                      db_path=degrade_db_path, on_degrade=_mark_degraded,
                      consolidate=config.consolidate_enabled(),
                      dedup_threshold=config.dedup_threshold(),
                      decay_half_life_days=config.decay_half_life_days(),
                      assoc=config.assoc_enabled(),
                      recall_budget=config.recall_budget(),
                      protect_head=config.protect_head(),
                      seed_floor=config.seed_floor(),
                      crystallize=config.crystallize_enabled(),
                      boot_index_cap=config.boot_index_cap(),
                      forget=config.forget_enabled(),
                      forget_age_days=config.forget_age_days(),
                      forget_interval=config.forget_interval(),
                      forget_max_per_sweep=config.forget_max_per_sweep(),
                      sync_read_interval=config.sync_read_interval(),
                      excerpt_chars=config.excerpt_chars(),
                      project=config.project())
        store.migrate()
        if embedder is not None:
            store.backfill_embeddings()
        _store = store
    return _store


@mcp.tool()
def remember(type: str, title: str, body: str,
             tags: str = "", links: list | None = None,
             crystallizes: list | None = None) -> dict:
    """Save a durable memory. UPSERTS: a memory of the same `type` with the same
    (whitespace/case-normalized) `title` is updated in place instead of
    duplicated, so re-remembering a fact refines it rather than cluttering.

    Worth remembering: decisions and their reasons, conventions, gotchas that
    cost time, the user's preferences and how they like to work, facts about
    the environment (paths, commands, accounts) you had to discover. Not worth
    it: anything derivable from the code, or a transcript of what you did.
    Remember it when you learn it, not at the end of the session.

    Memories of type project/feedback/reference are tagged with the current
    project automatically (`proj:<name>`, from CLAUDE_PROJECT_DIR) unless you
    pass a `proj:` tag yourself; `user` memories are about the person and
    stay global.

    Args:
        type: one of "user", "feedback", "project", "reference".
        title: a short label; also the dedup key within a type.
        body: the fact. For feedback/project, a "Why:" / "How to apply:" line helps.
        tags: optional comma-separated tags.
        links: optional list of related memory ids. Merged (union) into any
            links already on the memory, never replaces them - omitting this
            on a refine call preserves links set earlier.
        crystallizes: optional list of source memory ids this memory abstracts;
            links it over them as a crystallized principle (needs TETHER_CRYSTALLIZE).

    Returns {"id", "action"} where action is "created", "updated", or (with
    TETHER_CONSOLIDATE on) "consolidated" - a near-duplicate was superseded.
    """
    try:
        return _get_store().remember(type, title, body, tags=tags, links=links,
                                     crystallizes=crystallizes)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def recall(query: str = "", type: str | None = None, limit: int = 20,
           budget: int | None = None, session: str | None = None,
           tags: str | None = None, id: int | None = None,
           full: bool = False) -> dict:
    """Search memories by keyword and semantic similarity, then follow the
    usage graph to related memories, most relevant first.

    Ask in plain language, the way you would ask a colleague ("how do we run
    the integration tests?", "what did the user decide about the auth
    library?") - a memory matching some of the words is a hit, and the best
    match ranks first. Recall BEFORE starting a task, not only when stuck:
    the index you were given at session start is titles only. Memories from
    the current project rank slightly ahead of equally-good ones from
    elsewhere.

    Each hit carries {id, type, title, body, tags, updated_at} and a `via`
    receipt explaining why it surfaced (a direct match, or the edge it was
    reached through). Use `updated_at` to judge staleness (an old fact may no
    longer hold; verify before relying on it) and `id` to cite what you update
    via remember/link.

    `body` is an EXCERPT centered on your query, not the whole memory. When a
    memory was longer than the excerpt, the hit also carries `truncated: true`
    and `body_chars` (the full length). To read one in full, call
    recall(id=N) - that returns just that memory, whole.

    Args:
        query: free text; punctuation is safe. May be omitted if `tags` is given.
        type: optional filter ("user"/"feedback"/"project"/"reference").
        limit: max results (default 20).
        budget: how far to follow associations (0 = direct matches only).
        session: optional id grouping related recalls so they prime each other.
        tags: optional comma-separated tags; exact-match filter (a memory must
            carry every listed tag). Combine with `query` to filter its ranked
            hits, or use alone (query omitted) to list every current memory
            with those tags, newest first, deterministic rather than
            ranked - raise `limit` to fetch beyond the default page size.
        id: fetch this one memory in full instead of searching. Use it after a
            search returns a `truncated` hit you want to read completely.
        full: return complete bodies for every hit instead of excerpts. Costs
            the whole payload - prefer id= for the one memory you actually need.
    """
    try:
        store = _get_store()
        if id is not None:
            hit = store.get(id)
            return {"results": [hit] if hit else []}
        return {"results": store.recall(
            query, type=type, limit=limit, budget=budget, session=session,
            tags=tags, full=full)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def link(id_a: int, id_b: int) -> dict:
    """Create a bidirectional link between two memories by id."""
    try:
        return _get_store().link(id_a, id_b)
    except Exception as e:
        return {"error": str(e)}


def dismiss_cluster(id_a: int, id_b: int) -> dict:
    """Reflection control: dismiss the crystallization candidate nucleated by the
    peak edge (id_a, id_b) so it is not re-surfaced. Not a memory operation.

    Registered as an MCP tool only when TETHER_CRYSTALLIZE is on - see the
    _register_crystallization_surface() call at the bottom of this module.
    """
    try:
        return _get_store().dismiss_cluster(id_a, id_b)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def forget(id: int) -> dict:
    """Soft-delete a memory by id: marks it no longer current (excluded from
    recall/the boot index) but keeps the row, reversibly, like consolidation
    and the forgetting sweep already do. Returns {"forgotten", "existed"}.
    (Permanent purge is an admin-only CLI operation, not available here.)
    """
    try:
        return _get_store().forget(id)
    except Exception as e:
        return {"error": str(e)}


@mcp.resource("tether://memory-index")
def memory_index() -> str:
    """A compact index of memories - one line per memory as `[type] #id title`.
    When the server knows which project it is serving (CLAUDE_PROJECT_DIR or
    TETHER_PROJECT), a `# This project` section comes first, then
    `# Everything else`; a large store is curated to the most load-bearing and
    most recent memories. Auto-loaded each session so memory helps even
    without an explicit recall; these are titles only - call recall() with a
    question, or recall(id=N), to read the ones that matter for the task.
    """
    try:
        return _get_store().boot_index()
    except Exception as e:
        return f"(memory index unavailable: {e})"


@mcp.resource("tether://status")
def status() -> str:
    """Read-only runtime status (#51): what's actually active right now, since
    several features (semantic recall, sync) degrade silently by design.
    Pull-only like tether://crystallization - not auto-loaded.
    """
    try:
        store = _get_store()
        conn = store._conn
        memory_count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE valid_to IS NULL").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return json.dumps({
            "semantic_enabled": store._embedder is not None,
            "embedding_model": getattr(store._embedder, "name", None),
            "sync_mode": _sync_mode,
            "project": store._project,
            "memory_count": memory_count,
            "edge_count": edge_count,
            "db_path": str(config.db_path()),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def crystallization() -> str:
    """Pull-only reflection view: candidate clusters that may want a name. Read
    it during a reflection pass (NOT auto-loaded). For each cluster, name it via
    remember(..., crystallizes=member_ids) or drop it via dismiss_cluster(peak).

    Registered as an MCP resource only when TETHER_CRYSTALLIZE is on.
    """
    try:
        return json.dumps({"candidates": _get_store().crystallization_candidates()})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _register_crystallization_surface() -> bool:
    """Register the crystallization tool + resource only when the feature is on
    (#65). With it off - the default - tether's MCP surface is exactly the four
    memory verbs the README promises, instead of a fifth tool an agent can only
    misuse and a resource that can only return an empty list.

    Both functions are defined unconditionally above and merely *registered*
    here, so importing this module gives the same Python surface either way -
    only what the agent is offered changes. Reading the env at import time is
    consistent with the module's promise that import never touches the
    filesystem: config here is a pure environment read.
    """
    if not config.crystallize_enabled():
        return False
    mcp.tool()(dismiss_cluster)
    mcp.resource("tether://crystallization")(crystallization)
    return True


_crystallization_registered = _register_crystallization_surface()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
