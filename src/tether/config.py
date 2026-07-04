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


_CONSOLIDATE_ON = {"1", "true", "yes", "on"}
_DEFAULT_DEDUP_THRESHOLD = 0.92


def author() -> str:
    return os.environ.get("TETHER_AUTHOR") or device_id()


def consolidate_enabled() -> bool:
    val = os.environ.get("TETHER_CONSOLIDATE")
    if val is None:
        return False
    return val.strip().lower() in _CONSOLIDATE_ON


def dedup_threshold() -> float:
    raw = os.environ.get("TETHER_DEDUP_THRESHOLD")
    if not raw:
        return _DEFAULT_DEDUP_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_DEDUP_THRESHOLD


def decay_half_life_days():
    raw = os.environ.get("TETHER_DECAY_HALF_LIFE_DAYS")
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None


_ASSOC_OFF = {"0", "false", "no", "off"}
_DEFAULT_RECALL_BUDGET = 24


def assoc_enabled() -> bool:
    """Associative (spreading-activation) recall is on by default; any of
    0/false/no/off forces plain v0.2 hybrid recall."""
    val = os.environ.get("TETHER_ASSOC")
    if val is None:
        return True
    return val.strip().lower() not in _ASSOC_OFF


def recall_budget() -> int:
    """Default spreading budget (max node-expansions). 0 = spreading off."""
    raw = os.environ.get("TETHER_RECALL_BUDGET")
    if not raw:
        return _DEFAULT_RECALL_BUDGET
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_RECALL_BUDGET
    return val if val >= 0 else _DEFAULT_RECALL_BUDGET
