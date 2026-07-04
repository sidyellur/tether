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
