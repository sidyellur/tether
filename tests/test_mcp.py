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
