"""config.py - resolve DB path, sync credentials, and device id from the env.

Pure environment reads, no side effects. The zero-config default (no env vars)
yields a local-only DB under XDG_DATA_HOME and no sync.
"""

import os
import socket
from collections import namedtuple
from pathlib import Path

SyncConfig = namedtuple("SyncConfig", ["url", "token"])


def db_path() -> Path:
    override = os.environ.get("TETHER_DB")
    if override:
        return Path(override)
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "tether" / "memory.db"


def sync_config():
    url = os.environ.get("TETHER_SYNC_URL")
    token = os.environ.get("TETHER_SYNC_TOKEN")
    if url and token:
        return SyncConfig(url, token)
    return None


def device_id() -> str:
    return os.environ.get("TETHER_DEVICE_ID") or socket.gethostname()
