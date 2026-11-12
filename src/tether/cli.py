#!/usr/bin/env python3
"""cli.py - the admin command line: export, import, restore, history, revert and hard-purge.

Not the agent-facing surface (that's server.py's MCP tools). This is the
operator escape hatch tether's non-destructive design otherwise lacks (#49):
a plain backup independent of the DB file, and a real permanent delete for
when forget()'s soft-delete genuinely isn't enough.

`import` and `restore` close the two asymmetries in that story (#64): a
backup you can take but not restore isn't a backup, and a soft-delete
documented as reversible needs somewhere to actually reverse it.
"""

import argparse
import json
import os
import sys

from . import config
from .store import Store
from .sync import open_connection


def _build_store() -> Store:
    path = config.db_path()
    conn, sync_now, _mode = open_connection(path, config.sync_config())
    # Same tokenizer setting as the server, so an admin command never flips
    # the FTS index back and forth against what the server built (#90).
    store = Store(conn, device_id=config.device_id(), sync_now=sync_now,
                  author=config.author(), stemming=config.fts_stemming())
    store.migrate()
    return store


def cmd_export(args) -> int:
    data = _build_store().export_all()
    text = json.dumps(data, indent=2)
    if args.output:
        # A backup holds every memory: create it readable by its owner only.
        # (An existing file keeps whatever mode it already has.)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


def cmd_import(args) -> int:
    try:
        with open(args.file) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"could not read {args.file}: {e}\n")
        return 1
    if not isinstance(data, list):
        sys.stderr.write(
            f"{args.file} is not a tether export (expected a JSON list)\n")
        return 1
    result = _build_store().import_records(data)
    print(json.dumps(result))
    return 0


def cmd_restore(args) -> int:
    try:
        result = _build_store().restore(args.id)
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    print(json.dumps(result))
    return 0 if result["existed"] else 1


def cmd_history(args) -> int:
    versions = _build_store().history(args.id)
    print(json.dumps(versions, indent=2))
    return 0


def cmd_revert(args) -> int:
    try:
        result = _build_store().revert(args.id, args.version)
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    print(json.dumps(result))
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

    p_import = sub.add_parser(
        "import", help="merge a `tether export` JSON file back into the store")
    p_import.add_argument("file", help="path to a JSON file written by `tether export`")
    p_import.set_defaults(func=cmd_import)

    p_restore = sub.add_parser(
        "restore", help="un-forget a soft-deleted memory (clears valid_to)")
    p_restore.add_argument("id", type=int)
    p_restore.set_defaults(func=cmd_restore)

    p_history = sub.add_parser(
        "history", help="list the earlier versions of a memory that updates replaced")
    p_history.add_argument("id", type=int)
    p_history.set_defaults(func=cmd_history)

    p_revert = sub.add_parser(
        "revert", help="put an earlier version of a memory back (see `history`)")
    p_revert.add_argument("id", type=int)
    p_revert.add_argument("version", type=int)
    p_revert.set_defaults(func=cmd_revert)

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
