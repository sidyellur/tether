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
import sqlite3

from mcp.server.fastmcp import FastMCP

from . import config
from .store import Store
from .sync import open_connection

mcp = FastMCP("tether")

_store = None


def _get_store() -> Store:
    global _store
    if _store is None:
        path = config.db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn, sync_now = open_connection(path, config.sync_config())
        # A plain sqlite3.Connection is already local-only; only a sync
        # replica connection needs a db_path to degrade to on write failure
        # (#44 - a mid-session network drop should degrade gracefully
        # instead of raising out of remember/link/forget).
        degrade_db_path = None if isinstance(conn, sqlite3.Connection) else path
        embedder = None
        if config.semantic_enabled():
            from . import embed
            embedder = embed.get_embedder(config.embedding_model())
        store = Store(conn, device_id=config.device_id(), sync_now=sync_now,
                      embedder=embedder, author=config.author(),
                      db_path=degrade_db_path,
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
                      forget_max_per_sweep=config.forget_max_per_sweep())
        store.migrate()
        if embedder is not None:
            store.backfill_embeddings()
        _store = store
    return _store


@mcp.tool()
def remember(type: str, title: str, body: str,
             tags: str = "", links: list = None,
             crystallizes: list = None) -> dict:
    """Save a durable memory. UPSERTS: a memory of the same `type` with the same
    (whitespace/case-normalized) `title` is updated in place instead of
    duplicated, so re-remembering a fact refines it rather than cluttering.

    Args:
        type: one of "user", "feedback", "project", "reference".
        title: a short label; also the dedup key within a type.
        body: the fact. For feedback/project, a "Why:" / "How to apply:" line helps.
        tags: optional comma-separated tags.
        links: optional list of related memory ids.
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
def recall(query: str, type: str = None, limit: int = 20,
           budget: int = None, session: str = None) -> dict:
    """Search memories by keyword and semantic similarity, then follow the
    usage graph to related memories, most relevant first.

    Each hit carries {id, type, title, body, tags, updated_at} and a `via`
    receipt explaining why it surfaced (a direct match, or the edge it was
    reached through). Use `updated_at` to judge staleness (an old fact may no
    longer hold; verify before relying on it) and `id` to cite what you update
    via remember/link.

    Args:
        query: free text; punctuation is safe.
        type: optional filter ("user"/"feedback"/"project"/"reference").
        limit: max results (default 20).
        budget: how far to follow associations (0 = direct matches only).
        session: optional id grouping related recalls so they prime each other.
    """
    try:
        return {"results": _get_store().recall(
            query, type=type, limit=limit, budget=budget, session=session)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def link(id_a: int, id_b: int) -> dict:
    """Create a bidirectional link between two memories by id."""
    try:
        return _get_store().link(id_a, id_b)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def dismiss_cluster(id_a: int, id_b: int) -> dict:
    """Reflection control: dismiss the crystallization candidate nucleated by the
    peak edge (id_a, id_b) so it is not re-surfaced. Not a memory operation."""
    try:
        return _get_store().dismiss_cluster(id_a, id_b)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def forget(id: int) -> dict:
    """Permanently delete a memory by id. Returns {"forgotten", "existed"}."""
    try:
        return _get_store().forget(id)
    except Exception as e:
        return {"error": str(e)}


@mcp.resource("tether://memory-index")
def memory_index() -> str:
    """A compact index of ALL memories - one line per memory as `[type] #id
    title`, newest first. Auto-loaded each session so memory helps even without
    an explicit recall; pull full bodies with recall() using the id.
    """
    try:
        return _get_store().boot_index()
    except Exception as e:
        return f"(memory index unavailable: {e})"


@mcp.resource("tether://crystallization")
def crystallization() -> str:
    """Pull-only reflection view: candidate clusters that may want a name. Read
    it during a reflection pass (NOT auto-loaded). For each cluster, name it via
    remember(..., crystallizes=member_ids) or drop it via dismiss_cluster(peak).
    """
    try:
        return json.dumps({"candidates": _get_store().crystallization_candidates()})
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    mcp.run()


if __name__ == "__main__":
    main()
