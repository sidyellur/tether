"""End-to-end test over the real MCP stdio transport (what an agent uses)."""

import asyncio
import json
import os
import sys

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
                     "body": "Tests first, evidence before done."})))
                assert created["action"] == "created"

                recall_result = json.loads(_text(await session.call_tool(
                    "recall", {"query": "tests"})))
                hits = recall_result.get("results", [])
                assert any(h["title"] == "Prefers TDD" for h in hits)

                # boot index resource reflects the write
                res = await session.read_resource("tether://memory-index")
                text = "".join(getattr(c, "text", "") for c in res.contents)
                assert "Prefers TDD" in text

    asyncio.run(run())
