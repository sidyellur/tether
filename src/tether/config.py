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


_FORGET_ON = {"1", "true", "yes", "on"}
_DEFAULT_BOOT_INDEX_CAP = 50
_DEFAULT_FORGET_AGE_DAYS = 90
_DEFAULT_FORGET_INTERVAL = 20
_DEFAULT_FORGET_MAX_PER_SWEEP = 10


def _pos_int(env: str, default: int) -> int:
    """A positive integer from the env, or the default (also on <1/unparseable)."""
    raw = os.environ.get(env)
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val >= 1 else default


def boot_index_cap() -> int:
    """Boot-index size above which hub-curation kicks in (needs a graph)."""
    return _pos_int("TETHER_BOOT_INDEX_CAP", _DEFAULT_BOOT_INDEX_CAP)


def forget_enabled() -> bool:
    """Forgetting sweep is opt-in, off by default."""
    val = os.environ.get("TETHER_FORGET")
    if val is None:
        return False
    return val.strip().lower() in _FORGET_ON


def forget_age_days() -> int:
    return _pos_int("TETHER_FORGET_AGE_DAYS", _DEFAULT_FORGET_AGE_DAYS)


def forget_interval() -> int:
    return _pos_int("TETHER_FORGET_INTERVAL", _DEFAULT_FORGET_INTERVAL)


def forget_max_per_sweep() -> int:
    return _pos_int("TETHER_FORGET_MAX_PER_SWEEP", _DEFAULT_FORGET_MAX_PER_SWEEP)
