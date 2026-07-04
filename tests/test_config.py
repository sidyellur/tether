import socket
from pathlib import Path

from tether import config


def test_db_path_prefers_explicit_override(monkeypatch):
    monkeypatch.setenv("TETHER_DB", "/tmp/custom/mem.db")
    assert config.db_path() == Path("/tmp/custom/mem.db")


def test_db_path_falls_back_to_xdg(monkeypatch):
    monkeypatch.delenv("TETHER_DB", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg")
    assert config.db_path() == Path("/tmp/xdg/tether/memory.db")


def test_sync_config_needs_both_vars(monkeypatch):
    monkeypatch.delenv("TETHER_SYNC_URL", raising=False)
    monkeypatch.delenv("TETHER_SYNC_TOKEN", raising=False)
    assert config.sync_config() is None
    monkeypatch.setenv("TETHER_SYNC_URL", "libsql://x.turso.io")
    assert config.sync_config() is None  # token still missing
    monkeypatch.setenv("TETHER_SYNC_TOKEN", "tok")
    cfg = config.sync_config()
    assert cfg.url == "libsql://x.turso.io" and cfg.token == "tok"


def test_device_id_defaults_to_hostname(monkeypatch):
    monkeypatch.delenv("TETHER_DEVICE_ID", raising=False)
    assert config.device_id() == socket.gethostname()
    monkeypatch.setenv("TETHER_DEVICE_ID", "laptop")
    assert config.device_id() == "laptop"


def test_semantic_enabled_default_true(monkeypatch):
    monkeypatch.delenv("TETHER_SEMANTIC", raising=False)
    assert config.semantic_enabled() is True


def test_semantic_disabled_by_env(monkeypatch):
    for v in ("0", "false", "off", "NO"):
        monkeypatch.setenv("TETHER_SEMANTIC", v)
        assert config.semantic_enabled() is False


def test_embedding_model_default_and_override(monkeypatch):
    monkeypatch.delenv("TETHER_EMBEDDING_MODEL", raising=False)
    assert config.embedding_model() == "minishlab/potion-base-8M"
    monkeypatch.setenv("TETHER_EMBEDDING_MODEL", "some/other-model")
    assert config.embedding_model() == "some/other-model"


def test_author_defaults_to_device_id(monkeypatch):
    monkeypatch.delenv("TETHER_AUTHOR", raising=False)
    monkeypatch.setenv("TETHER_DEVICE_ID", "laptop")
    assert config.author() == "laptop"
    monkeypatch.setenv("TETHER_AUTHOR", "sid")
    assert config.author() == "sid"


def test_consolidate_off_by_default(monkeypatch):
    monkeypatch.delenv("TETHER_CONSOLIDATE", raising=False)
    assert config.consolidate_enabled() is False
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv("TETHER_CONSOLIDATE", v)
        assert config.consolidate_enabled() is True


def test_dedup_threshold_default_and_bad_value(monkeypatch):
    monkeypatch.delenv("TETHER_DEDUP_THRESHOLD", raising=False)
    assert config.dedup_threshold() == 0.92
    monkeypatch.setenv("TETHER_DEDUP_THRESHOLD", "0.8")
    assert config.dedup_threshold() == 0.8
    monkeypatch.setenv("TETHER_DEDUP_THRESHOLD", "not-a-number")
    assert config.dedup_threshold() == 0.92


def test_decay_half_life_off_by_default_and_positive_only(monkeypatch):
    monkeypatch.delenv("TETHER_DECAY_HALF_LIFE_DAYS", raising=False)
    assert config.decay_half_life_days() is None
    monkeypatch.setenv("TETHER_DECAY_HALF_LIFE_DAYS", "30")
    assert config.decay_half_life_days() == 30.0
    monkeypatch.setenv("TETHER_DECAY_HALF_LIFE_DAYS", "-5")
    assert config.decay_half_life_days() is None
    monkeypatch.setenv("TETHER_DECAY_HALF_LIFE_DAYS", "junk")
    assert config.decay_half_life_days() is None


def test_assoc_enabled_default_true(monkeypatch):
    monkeypatch.delenv("TETHER_ASSOC", raising=False)
    assert config.assoc_enabled() is True
    for v in ("0", "false", "off", "NO"):
        monkeypatch.setenv("TETHER_ASSOC", v)
        assert config.assoc_enabled() is False


def test_recall_budget_default_and_parsing(monkeypatch):
    monkeypatch.delenv("TETHER_RECALL_BUDGET", raising=False)
    assert config.recall_budget() == 24
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "8")
    assert config.recall_budget() == 8
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "0")
    assert config.recall_budget() == 0
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "-5")
    assert config.recall_budget() == 24
    monkeypatch.setenv("TETHER_RECALL_BUDGET", "junk")
    assert config.recall_budget() == 24
