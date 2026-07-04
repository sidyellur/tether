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


_SEMANTIC_OFF = {"0", "false", "no", "off"}
_DEFAULT_EMBEDDING_MODEL = "minishlab/potion-base-8M"


def semantic_enabled() -> bool:
    """Semantic recall is on by default; any of 0/false/no/off disables it.

    Disabling forces keyword-only recall without needing the [semantic] extra.
    """
    val = os.environ.get("TETHER_SEMANTIC")
    if val is None:
        return True
    return val.strip().lower() not in _SEMANTIC_OFF


def embedding_model() -> str:
    return os.environ.get("TETHER_EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL
