"""config.py - resolve DB path, sync credentials, and device id from the env.

Pure environment reads, no side effects. The zero-config default (no env vars)
yields a local-only DB under XDG_DATA_HOME and no sync.
"""

import os
import socket
from collections import namedtuple
from pathlib import Path

SyncConfig = namedtuple("SyncConfig", ["url", "token"])


def _is_windows() -> bool:
    """Indirection so tests can exercise the Windows branch on any host -
    monkeypatching os.name directly would also change what pathlib
    instantiates, which fails outright on a POSIX box."""
    return os.name == "nt"


def db_path() -> Path:
    """Where memory.db lives. TETHER_DB wins; then XDG_DATA_HOME (honored on
    every platform, since someone who sets it means it); then the platform's
    own convention - %LOCALAPPDATA% on Windows (#68), ~/.local/share
    elsewhere."""
    override = os.environ.get("TETHER_DB")
    if override:
        return Path(override)
    base = os.environ.get("XDG_DATA_HOME")
    if not base:
        if _is_windows():
            base = (os.environ.get("LOCALAPPDATA")
                    or str(Path.home() / "AppData" / "Local"))
        else:
            base = str(Path.home() / ".local" / "share")
    return Path(base) / "tether" / "memory.db"


def sync_config():
    url = os.environ.get("TETHER_SYNC_URL")
    token = os.environ.get("TETHER_SYNC_TOKEN")
    if url and token:
        return SyncConfig(url, token)
    return None


def device_id() -> str:
    return os.environ.get("TETHER_DEVICE_ID") or socket.gethostname()


_PROJECT_OFF = {"0", "false", "no", "off"}


def project():
    """The project this server is serving, or None (#92).

    TETHER_PROJECT names it explicitly (any of 0/false/no/off disables project
    awareness). Otherwise it is the basename of CLAUDE_PROJECT_DIR, which
    Claude Code sets in every stdio MCP server's environment to the stable
    project root. Nothing falls back to the working directory: that is
    wherever the client happened to launch from, and a wrong project is
    worse than none. Commas are replaced (they separate tags) and whitespace
    trimmed so the value is safe as a `proj:<name>` tag."""
    raw = os.environ.get("TETHER_PROJECT")
    if raw is not None and raw.strip():
        if raw.strip().lower() in _PROJECT_OFF:
            return None
        name = raw
    else:
        root = os.environ.get("CLAUDE_PROJECT_DIR", "")
        name = os.path.basename(os.path.normpath(root)) if root.strip() else ""
    name = name.strip().replace(",", "-")
    return name or None


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
_DEFAULT_RECALL_BUDGET = 8
_DEFAULT_PROTECT_HEAD = 8
_DEFAULT_SEED_FLOOR = 0.35


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


def protect_head() -> int:
    """Number of top v0.2 hits locked in place before spread re-ranks the tail
    (seed-dominance guard; larger = more protection, less associative upside)."""
    raw = os.environ.get("TETHER_PROTECT_HEAD")
    if not raw:
        return _DEFAULT_PROTECT_HEAD
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_PROTECT_HEAD
    return val if val >= 0 else _DEFAULT_PROTECT_HEAD


def seed_floor() -> float:
    """Minimum cosine a vector hit needs to seed an associative walk (#15).
    Below it a memory is reachable only by edge, not seeded as near-tied noise.
    Out-of-range or unparseable values fall back to the default; 0 disables."""
    raw = os.environ.get("TETHER_SEED_FLOOR")
    if not raw:
        return _DEFAULT_SEED_FLOOR
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_SEED_FLOOR
    return val if 0.0 <= val <= 1.0 else _DEFAULT_SEED_FLOOR


_CRYSTALLIZE_ON = {"1", "true", "yes", "on"}


def crystallize_enabled() -> bool:
    """Crystallization (agent-in-the-loop principle detection) is opt-in, off by
    default."""
    val = os.environ.get("TETHER_CRYSTALLIZE")
    if val is None:
        return False
    return val.strip().lower() in _CRYSTALLIZE_ON


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


_DEFAULT_EXCERPT_CHARS = 500


def excerpt_chars() -> int:
    """Width of the relevance-centered excerpt recall returns per hit (#30).
    0 restores full-body recall. Unparseable or negative falls back to the
    default."""
    raw = os.environ.get("TETHER_EXCERPT_CHARS")
    if raw is None or not raw.strip():
        return _DEFAULT_EXCERPT_CHARS
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_EXCERPT_CHARS
    return val if val >= 0 else _DEFAULT_EXCERPT_CHARS


_STEMMING_OFF = {"0", "false", "no", "off"}


def fts_stemming() -> bool:
    """Porter stemming in the keyword index (#90) is on by default; any of
    0/false/no/off turns it off (the stemmer is English-only). migrate()
    rebuilds the index whenever this differs from what the DB was built with."""
    val = os.environ.get("TETHER_FTS_STEMMING")
    if val is None:
        return True
    return val.strip().lower() not in _STEMMING_OFF


_DEFAULT_SYNC_READ_INTERVAL = 30


def sync_read_interval() -> int:
    """Seconds between read-path sync pulls (#62). Reads debounce to at most
    one pull per interval; 0 disables read-path syncing entirely (restoring
    the write-only-sync behavior). Unparseable or negative falls back to the
    default."""
    raw = os.environ.get("TETHER_SYNC_READ_INTERVAL")
    if raw is None or not raw.strip():
        return _DEFAULT_SYNC_READ_INTERVAL
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_SYNC_READ_INTERVAL
    return val if val >= 0 else _DEFAULT_SYNC_READ_INTERVAL
