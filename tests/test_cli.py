import json

from tether import cli
from tether.store import Store


def _seed(tmp_path, monkeypatch):
    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("TETHER_SEMANTIC", "0")
    monkeypatch.delenv("TETHER_SYNC_URL", raising=False)
    monkeypatch.delenv("TETHER_SYNC_TOKEN", raising=False)
    store = cli._build_store()
    a = store.remember("user", "A", "body a")["id"]
    b = store.remember("project", "B", "body b")["id"]
    return a, b


def test_export_writes_current_memories_to_stdout(tmp_path, monkeypatch, capsys):
    a, b = _seed(tmp_path, monkeypatch)
    assert cli.main(["export"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {m["id"] for m in out} == {a, b}


def test_export_excludes_forgotten_memories(tmp_path, monkeypatch, capsys):
    a, b = _seed(tmp_path, monkeypatch)
    store = cli._build_store()
    store.forget(b)
    capsys.readouterr()  # discard the store-build's own noise, if any
    assert cli.main(["export"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {m["id"] for m in out} == {a}


def test_export_writes_to_file(tmp_path, monkeypatch):
    a, b = _seed(tmp_path, monkeypatch)
    out_path = tmp_path / "backup.json"
    assert cli.main(["export", "-o", str(out_path)]) == 0
    data = json.loads(out_path.read_text())
    assert {m["id"] for m in data} == {a, b}


def test_purge_without_yes_refuses(tmp_path, monkeypatch, capsys):
    a, _b = _seed(tmp_path, monkeypatch)
    assert cli.main(["purge", str(a)]) == 1
    err = capsys.readouterr().err
    assert "--yes" in err
    store = cli._build_store()
    assert store._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id=?", (a,)).fetchone()[0] == 1


def test_purge_with_yes_hard_deletes(tmp_path, monkeypatch, capsys):
    a, _b = _seed(tmp_path, monkeypatch)
    assert cli.main(["purge", str(a), "--yes"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"purged": a, "existed": True}
    store = cli._build_store()
    assert store._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id=?", (a,)).fetchone()[0] == 0


def test_purge_nonexistent_id_reports_false_and_fails(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    assert cli.main(["purge", "9999", "--yes"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result == {"purged": 9999, "existed": False}


def test_build_store_does_not_require_mcp_or_embed(tmp_path, monkeypatch):
    # The CLI is an admin path independent of the MCP server; it must not
    # need an embedder to export/purge, even with semantic search enabled.
    monkeypatch.setenv("TETHER_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("TETHER_SEMANTIC", raising=False)
    monkeypatch.delenv("TETHER_SYNC_URL", raising=False)
    monkeypatch.delenv("TETHER_SYNC_TOKEN", raising=False)
    store = cli._build_store()
    assert isinstance(store, Store)
    assert store._embedder is None
