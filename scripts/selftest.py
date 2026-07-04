#!/usr/bin/env python3
"""Manual end-to-end smoke test against a temp DB (no network, no MCP).

Run: python scripts/selftest.py
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tether.store import Store  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(str(Path(d) / "m.db"))
        s = Store(conn, device_id="selftest", sync_now=lambda *a, **k: None)
        s.migrate()

        a = s.remember("user", "Prefers TDD", "Tests first.")
        assert a["action"] == "created", a
        again = s.remember("user", "prefers tdd", "Tests first, evidence before done.")
        assert again["action"] == "updated" and again["id"] == a["id"], again

        b = s.remember("project", "tether", "Shared agent memory across devices.")
        hits = s.recall("memory")
        assert any(h["id"] == b["id"] for h in hits), hits

        s.link(a["id"], b["id"])
        assert b["id"] in s._links_of(a["id"])

        idx = s.boot_index()
        assert "prefers tdd" in idx and "tether" in idx, idx

        gone = s.forget(b["id"])
        assert gone == {"forgotten": b["id"], "existed": True}, gone
        assert s.recall("tether across devices") == [] or all(
            h["id"] != b["id"] for h in s.recall("tether"))

    print("SELF-TEST: ALL PASS")


if __name__ == "__main__":
    main()
