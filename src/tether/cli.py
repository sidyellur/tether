#!/usr/bin/env python3
"""cli.py - the admin command line: export and hard-purge.

Not the agent-facing surface (that's server.py's MCP tools). This is the
operator escape hatch tether's non-destructive design otherwise lacks (#49):
a plain backup independent of the DB file, and a real permanent delete for
when forget()'s soft-delete genuinely isn't enough.
"""

import argparse
import json
import sys

from . import config
from .store import Store
from .sync import open_connection


def _build_store() -> Store:
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn, sync_now, _mode = open_connection(path, config.sync_config())
    store = Store(conn, device_id=config.device_id(), sync_now=sync_now,
                  author=config.author())
    store.migrate()
    return store


def cmd_export(args) -> int:
    data = _build_store().export_all()
    text = json.dumps(data, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


def cmd_purge(args) -> int:
    if not args.yes:
        sys.stderr.write(
            f"refusing to permanently purge memory #{args.id} without --yes\n")
        return 1
    result = _build_store().purge(args.id)
    print(json.dumps(result))
    return 0 if result["existed"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tether", description="tether admin CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="dump all current memories to JSON")
    p_export.add_argument("-o", "--output", help="write to this file instead of stdout")
    p_export.set_defaults(func=cmd_export)

    p_purge = sub.add_parser(
        "purge", help="permanently delete a memory (bypasses forget's soft-delete)")
    p_purge.add_argument("id", type=int)
    p_purge.add_argument("--yes", action="store_true",
                          help="confirm the permanent, non-reversible delete")
    p_purge.set_defaults(func=cmd_purge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
