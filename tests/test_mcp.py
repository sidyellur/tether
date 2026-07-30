"""End-to-end test over the real MCP stdio transport (what an agent uses)."""

import asyncio
import json
import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {"remember", "recall", "link", "forget"}


def _text(result):
    return "".join(c.text for c in result.content
                   if getattr(c, "type", "") == "text")


def test_server_imports_against_installed_mcp_major():
    """#69: mcp 2.0 renamed FastMCP -> MCPServer and moved it to
    mcp.server.mcpserver with no back-compat alias, which broke every fresh
    install (`mcp>=1.0` is unbounded). server.py carries an import shim; this
    asserts it resolved against whatever mcp major is actually installed and
    produced a usable server object, rather than the module merely importing.
    """
    from tether import server

    assert server.mcp is not None
    # the decorators must have bound to a real server: constructor + @tool()
    # + @resource() + run() are the whole surface tether depends on.
    for attr in ("tool", "resource", "run"):
        assert callable(getattr(server.mcp, attr)), f"mcp.{attr} missing"
    for verb in EXPECTED_TOOLS:
        assert callable(getattr(server, verb)), f"{verb} not defined"


def test_optional_tool_params_advertise_null_in_schema():
    """A param defaulting to None must say so in its JSON schema.

    Before the implicit-Optional cleanup these were annotated `type: str = None`,
    which made the SDK emit {"type": "string", "default": null} — a schema whose
    own default violates the type it declares. A strict client validating
    defaults can reject that. Each such param should now be a null-accepting
    union instead.
    """
    from tether import server

    async def get_schemas():
        tools = await server.mcp.list_tools()
        return {t.name: (getattr(t, "input_schema", None)
                         or getattr(t, "inputSchema")) for t in tools}

    schemas = asyncio.run(get_schemas())
    optional_params = [("recall", "type"), ("recall", "budget"),
                       ("recall", "session"), ("recall", "tags"),
                       ("remember", "links"), ("remember", "crystallizes")]
    for tool, param in optional_params:
        prop = schemas[tool]["properties"][param]
        assert prop.get("default", "missing") is None, f"{tool}.{param} default"
        variants = prop.get("anyOf") or prop.get("oneOf") or [prop]
        assert any(v.get("type") == "null" for v in variants), (
            f"{tool}.{param} defaults to null but its schema does not "
            f"accept null: {prop}")


def test_mcp_stdio_roundtrip(tmp_path):
    env = dict(os.environ, TETHER_DB=str(tmp_path / "mem.db"),
               TETHER_DEVICE_ID="ci")
    env.pop("TETHER_SYNC_URL", None)
    env.pop("TETHER_SYNC_TOKEN", None)
    env["TETHER_SEMANTIC"] = "0"  # keyword-only: no model download in CI
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tether.server"], env=env)

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = {t.name for t in (await session.list_tools()).tools}
                assert EXPECTED_TOOLS <= tools

                created = json.loads(_text(await session.call_tool(
                    "remember",
                    {"type": "user", "title": "Prefers TDD",
                     "body": "Tests first, evidence before done.",
                     "tags": "blog-journal,proj:tether"})))
                assert created["action"] == "created"

                recall_result = json.loads(_text(await session.call_tool(
                    "recall", {"query": "tests"})))
                hits = recall_result.get("results", [])
                assert any(h["title"] == "Prefers TDD" for h in hits)

                # #50: exact tag filter, standalone (no query)
                tag_result = json.loads(_text(await session.call_tool(
                    "recall", {"tags": "blog-journal,proj:tether"})))
                tag_hits = tag_result.get("results", [])
                assert any(h["title"] == "Prefers TDD" for h in tag_hits)

                # a superstring tag must not match ("proj:tether" != "proj:tether2")
                no_match = json.loads(_text(await session.call_tool(
                    "recall", {"tags": "proj:tether2"})))
                assert no_match.get("results", []) == []

                # boot index resource reflects the write
                res = await session.read_resource("tether://memory-index")
                text = "".join(getattr(c, "text", "") for c in res.contents)
                assert "Prefers TDD" in text

    asyncio.run(run())


def test_server_wires_semantic_recall(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    from tether import server

    class Fake:
        name = "fake-3d"
        dims = 3
        _AXES = [("car", "automobile", "drive"),
                 ("pizza", "food"), ("python", "test")]

        def embed(self, text):
            import math
            t = text.lower()
            v = [float(sum(w in t for w in ax)) for ax in self._AXES]
            n = math.sqrt(sum(x * x for x in v))
            return [x / n for x in v] if n else v

    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("TETHER_SEMANTIC", raising=False)
    monkeypatch.delenv("TETHER_SYNC_URL", raising=False)
    monkeypatch.delenv("TETHER_SYNC_TOKEN", raising=False)
    monkeypatch.setattr("tether.embed.get_embedder", lambda *a, **k: Fake())

    server._store = None  # reset the lazy singleton
    try:
        store = server._get_store()
        store.remember("user", "Commute", "I love my car")
        hits = store.recall("automobile")     # synonym, keyword would miss
        assert hits and hits[0]["title"] == "Commute"
    finally:
        server._store = None


def test_server_consolidation_defaults_off(monkeypatch, tmp_path):
    from tether import server
    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("TETHER_CONSOLIDATE", raising=False)
    monkeypatch.setenv("TETHER_SEMANTIC", "0")  # keep it hermetic
    monkeypatch.delenv("TETHER_SYNC_URL", raising=False)
    monkeypatch.delenv("TETHER_SYNC_TOKEN", raising=False)

    server._store = None
    try:
        store = server._get_store()
        assert store._consolidate is False
        assert store._decay_half_life_days is None
        assert store._author  # falls back to device id, never empty
    finally:
        server._store = None


@pytest.mark.skipif(os.environ.get("TETHER_TEST_REAL_MODEL") != "1",
                    reason="set TETHER_TEST_REAL_MODEL=1 to run the real model over stdio")
def test_mcp_roundtrip_with_real_semantic(tmp_path):
    """Opt-in: exercise the REAL Model2Vec embedder through the full MCP stdio
    transport (the seam the hermetic tests skip). Off by default so CI stays
    fast and offline."""
    pytest.importorskip("model2vec")
    pytest.importorskip("numpy")
    env = dict(os.environ, TETHER_DB=str(tmp_path / "mem.db"),
               TETHER_DEVICE_ID="ci", TETHER_SEMANTIC="1")
    env.pop("TETHER_SYNC_URL", None)
    env.pop("TETHER_SYNC_TOKEN", None)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tether.server"], env=env)

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(
                    "remember",
                    {"type": "user", "title": "Commute",
                     "body": "I love my car and driving to work"})
                recall_result = json.loads(_text(await session.call_tool(
                    "recall", {"query": "automobile"})))  # synonym keyword misses
                hits = recall_result.get("results", [])
                assert any(h["title"] == "Commute" for h in hits)

    asyncio.run(run())


def test_server_recall_budget_session(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    from tether import server

    class Fake:
        name = "fake-3d"
        dims = 3
        _AXES = [("car", "automobile", "drive"), ("pizza", "food"), ("python", "test")]

        def embed(self, text):
            import math
            t = text.lower()
            v = [float(sum(w in t for w in ax)) for ax in self._AXES]
            n = math.sqrt(sum(x * x for x in v))
            return [x / n for x in v] if n else v

    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("TETHER_ASSOC", raising=False)
    monkeypatch.delenv("TETHER_SEMANTIC", raising=False)
    monkeypatch.delenv("TETHER_SYNC_URL", raising=False)
    monkeypatch.delenv("TETHER_SYNC_TOKEN", raising=False)
    monkeypatch.setattr("tether.embed.get_embedder", lambda *a, **k: Fake())
    server._store = None
    try:
        s = server._get_store()
        assert s._graph.enabled is True
        a = s.remember("user", "Auth", "we switched to JWT tokens")["id"]
        b = s.remember("project", "Why", "rationale")["id"]
        s.link(a, b)
        hits = s.recall("JWT tokens", budget=8, session="sess-1")
        assert any(h["id"] == b for h in hits)     # reached via the explicit edge
    finally:
        server._store = None


def test_mcp_forget_is_soft_delete(tmp_path):
    env = dict(os.environ, TETHER_DB=str(tmp_path / "mem.db"),
               TETHER_DEVICE_ID="ci")
    env.pop("TETHER_SYNC_URL", None)
    env.pop("TETHER_SYNC_TOKEN", None)
    env["TETHER_SEMANTIC"] = "0"
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tether.server"], env=env)

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                created = json.loads(_text(await session.call_tool(
                    "remember", {"type": "user", "title": "A", "body": "b"})))
                mid = created["id"]

                forgotten = json.loads(_text(await session.call_tool(
                    "forget", {"id": mid})))
                assert forgotten == {"forgotten": mid, "existed": True}

                recall_result = json.loads(_text(await session.call_tool(
                    "recall", {"query": "A"})))
                assert recall_result.get("results", []) == []

                res = await session.read_resource("tether://status")
                status = json.loads(
                    "".join(getattr(c, "text", "") for c in res.contents))
                # soft-deleted, not gone: memory_count excludes it (valid_to set)
                assert status["memory_count"] == 0

    asyncio.run(run())


def test_server_wires_forget_config(monkeypatch, tmp_path):
    from tether import server
    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("TETHER_SEMANTIC", "0")     # no embedder needed
    monkeypatch.setenv("TETHER_FORGET", "1")
    monkeypatch.setenv("TETHER_BOOT_INDEX_CAP", "7")
    server._store = None
    try:
        s = server._get_store()
        assert s._forget is True
        assert s._boot_index_cap == 7
    finally:
        server._store = None


def test_mcp_status_resource(tmp_path):
    env = dict(os.environ, TETHER_DB=str(tmp_path / "mem.db"),
               TETHER_DEVICE_ID="ci")
    env.pop("TETHER_SYNC_URL", None)
    env.pop("TETHER_SYNC_TOKEN", None)
    env["TETHER_SEMANTIC"] = "0"  # keyword-only: no model download in CI
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tether.server"], env=env)

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(
                    "remember", {"type": "user", "title": "A", "body": "b"})

                res = await session.read_resource("tether://status")
                text = "".join(getattr(c, "text", "") for c in res.contents)
                data = json.loads(text)
                assert data["semantic_enabled"] is False
                assert data["embedding_model"] is None
                assert data["sync_mode"] == "local"
                assert data["memory_count"] == 1
                assert data["edge_count"] == 0
                assert data["db_path"] == str(tmp_path / "mem.db")

    asyncio.run(run())


def test_status_resource_reports_semantic_and_sync_config(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    from tether import server

    class Fake:
        name = "fake-3d"
        dims = 3

        def embed(self, text):
            return [1.0, 0.0, 0.0]

    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("TETHER_SEMANTIC", raising=False)
    monkeypatch.delenv("TETHER_SYNC_URL", raising=False)
    monkeypatch.delenv("TETHER_SYNC_TOKEN", raising=False)
    monkeypatch.setattr("tether.embed.get_embedder", lambda *a, **k: Fake())

    server._store = None
    server._sync_mode = None
    try:
        data = json.loads(server.status())
        assert data["semantic_enabled"] is True
        assert data["embedding_model"] == "fake-3d"
        assert data["sync_mode"] == "local"
        assert data["memory_count"] == 0
        assert data["db_path"] == str(tmp_path / "m.db")
    finally:
        server._store = None
        server._sync_mode = None


def test_status_resource_reports_degraded_sync(monkeypatch, tmp_path):
    from tether import server, sync

    def boom(*a, **k):
        raise RuntimeError("cannot reach turso")
    monkeypatch.setattr(sync, "_open_replica", boom)

    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("TETHER_SEMANTIC", "0")
    monkeypatch.setenv("TETHER_SYNC_URL", "libsql://x.turso.io")
    monkeypatch.setenv("TETHER_SYNC_TOKEN", "tok")

    server._store = None
    server._sync_mode = None
    try:
        data = json.loads(server.status())
        assert data["sync_mode"] == "degraded"
    finally:
        server._store = None
        server._sync_mode = None


def _surface(tmp_path, **extra_env):
    """Tool names + resource URIs an agent is actually offered, over real stdio."""
    env = dict(os.environ, TETHER_DB=str(tmp_path / "mem.db"),
               TETHER_DEVICE_ID="ci", TETHER_SEMANTIC="0", **extra_env)
    env.pop("TETHER_SYNC_URL", None)
    env.pop("TETHER_SYNC_TOKEN", None)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tether.server"], env=env)

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                resources = {str(r.uri)
                             for r in (await session.list_resources()).resources}
                return tools, resources

    return asyncio.run(run())


def test_crystallization_surface_hidden_by_default(tmp_path):
    """#65: with TETHER_CRYSTALLIZE off (the default), the agent must be offered
    exactly the four memory verbs - not a fifth tool it can only misuse and a
    resource that can only return an empty list."""
    tools, resources = _surface(tmp_path)

    assert tools == EXPECTED_TOOLS, f"expected only the four verbs, got {tools}"
    assert "dismiss_cluster" not in tools
    assert not any("crystallization" in u for u in resources), resources


def test_crystallization_surface_appears_when_enabled(tmp_path):
    """...and the same surface DOES appear once the feature is on, so the gate
    hides the tool rather than removing it."""
    tools, resources = _surface(tmp_path, TETHER_CRYSTALLIZE="1")

    assert "dismiss_cluster" in tools
    assert EXPECTED_TOOLS <= tools
    assert any("crystallization" in u for u in resources), resources


def test_dismiss_cluster_refuses_when_crystallization_off(tmp_path):
    """Defense in depth: hiding the tool stops an agent being *offered* it, but
    the store must also refuse the write. crystallize_dismissed rows are
    persistent and nothing else consumes them, so a stray dismissal against a
    non-crystallizing store would silently suppress a candidate later, whenever
    the feature does get enabled."""
    import sqlite3

    from tether.store import Store

    conn = sqlite3.connect(":memory:")
    s = Store(conn, "d", lambda *a, **k: None, assoc=True)   # crystallize off
    s.migrate()
    with pytest.raises(ValueError, match="crystallization is not enabled"):
        s.dismiss_cluster(1, 2)
    n = conn.execute("SELECT COUNT(*) FROM crystallize_dismissed").fetchone()[0]
    assert n == 0, "a refused dismissal must not leave a row behind"


def test_mcp_crystallization_resource(tmp_path):
    """Test that the tether://crystallization resource is wired and returns valid JSON."""
    env = dict(os.environ, TETHER_DB=str(tmp_path / "mem.db"),
               TETHER_DEVICE_ID="ci", TETHER_CRYSTALLIZE="1")
    env.pop("TETHER_SYNC_URL", None)
    env.pop("TETHER_SYNC_TOKEN", None)
    env["TETHER_SEMANTIC"] = "0"  # keyword-only: no model download in CI
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tether.server"], env=env)

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # read the crystallization resource
                res = await session.read_resource("tether://crystallization")
                text = "".join(getattr(c, "text", "") for c in res.contents)
                # parse as JSON and assert it has the expected shape
                data = json.loads(text)
                assert isinstance(data, dict)
                assert "candidates" in data
                assert isinstance(data["candidates"], list)

    asyncio.run(run())


def test_mcp_recall_excerpts_then_fetches_whole(tmp_path):
    """#30 end-to-end over stdio: a search returns a marked excerpt, and
    recall(id=N) - an overload rather than a fifth verb - returns that memory
    whole. This is the contract an agent actually sees."""
    long_body = ("preamble. " + "filler. " * 300
                 + " the DISTINCTIVE detail sits here. " + "tail. " * 300)
    env = dict(os.environ, TETHER_DB=str(tmp_path / "mem.db"),
               TETHER_DEVICE_ID="ci", TETHER_SEMANTIC="0")
    env.pop("TETHER_SYNC_URL", None)
    env.pop("TETHER_SYNC_TOKEN", None)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tether.server"], env=env)

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("remember", {
                    "type": "project", "title": "Big", "body": long_body})

                found = json.loads(_text(await session.call_tool(
                    "recall", {"query": "distinctive"})))["results"]
                assert found, "excerpted recall lost the hit entirely"
                hit = found[0]
                assert hit["truncated"] is True
                assert hit["body_chars"] == len(long_body)
                assert len(hit["body"]) < len(long_body) / 5
                assert "DISTINCTIVE" in hit["body"]

                whole = json.loads(_text(await session.call_tool(
                    "recall", {"id": hit["id"]})))["results"]
                assert whole[0]["body"] == long_body
                assert "truncated" not in whole[0]

    asyncio.run(run())
