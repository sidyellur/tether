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


def test_restore_brings_back_a_forgotten_memory(tmp_path, monkeypatch, capsys):
    """#64: forget() is documented as reversible, so there must be somewhere to
    reverse it. Before this, a soft-deleted memory was only recoverable by
    hand-editing the DB."""
    a, b = _seed(tmp_path, monkeypatch)
    cli._build_store().forget(b)
    capsys.readouterr()

    assert cli.main(["export"]) == 0
    assert {m["id"] for m in json.loads(capsys.readouterr().out)} == {a}

    assert cli.main(["restore", str(b)]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "restored"

    assert cli.main(["export"]) == 0
    assert {m["id"] for m in json.loads(capsys.readouterr().out)} == {a, b}


def test_restore_refuses_when_the_title_was_taken_over(tmp_path, monkeypatch, capsys):
    """A newer memory can claim the (type, title) while the old one is archived.
    Restoring would put two current rows on one dedup key, which the partial
    unique index rejects — so fail with a message naming the blocker instead of
    an opaque IntegrityError."""
    _seed(tmp_path, monkeypatch)
    store = cli._build_store()
    old = store.remember("user", "Editor", "vim")["id"]
    store.forget(old)
    new = store.remember("user", "Editor", "emacs")["id"]
    assert new != old
    capsys.readouterr()

    assert cli.main(["restore", str(old)]) == 1
    err = capsys.readouterr().err
    assert f"#{new}" in err and "already the current" in err
    # the blocking row is untouched and the archived one stays archived
    store = cli._build_store()
    assert store._conn.execute(
        "SELECT valid_to IS NULL FROM memories WHERE id=?", (old,)).fetchone()[0] == 0


def test_restore_reports_missing_and_already_current(tmp_path, monkeypatch, capsys):
    a, _b = _seed(tmp_path, monkeypatch)
    assert cli.main(["restore", "9999"]) == 1
    assert json.loads(capsys.readouterr().out)["action"] == "missing"
    assert cli.main(["restore", str(a)]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "already-current"


def test_export_import_round_trips_into_an_empty_store(tmp_path, monkeypatch, capsys):
    """The point of #64: a backup you can take but not restore isn't a backup.
    Export a store with links, wipe it, import, and assert the content and the
    link topology both survive."""
    a, b = _seed(tmp_path, monkeypatch)
    store = cli._build_store()
    store.link(a, b)
    backup = tmp_path / "backup.json"
    assert cli.main(["export", "-o", str(backup)]) == 0
    original = {(m["type"], m["title"], m["body"]) for m in
                json.loads(backup.read_text())}

    # fresh, empty store
    monkeypatch.setenv("TETHER_DB", str(tmp_path / "restored.db"))
    assert cli.main(["export"]) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert cli.main(["import", str(backup)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["created"] == 2 and report["skipped"] == 0
    assert report["linked"] == 1 and report["dropped_links"] == 0

    assert cli.main(["export"]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert {(m["type"], m["title"], m["body"]) for m in imported} == original
    # links survived, remapped onto the new ids rather than the file's
    by_title = {m["title"]: m for m in imported}
    assert by_title["B"]["id"] in by_title["A"]["links"]
    assert by_title["A"]["id"] in by_title["B"]["links"]


def test_import_merges_rather_than_duplicating(tmp_path, monkeypatch, capsys):
    """Importing into a NON-empty store must upsert on (type, title), not
    fragment the store into near-duplicates."""
    a, _b = _seed(tmp_path, monkeypatch)
    backup = tmp_path / "backup.json"
    assert cli.main(["export", "-o", str(backup)]) == 0
    capsys.readouterr()

    assert cli.main(["import", str(backup)]) == 0      # same store, again
    report = json.loads(capsys.readouterr().out)
    assert report["created"] == 0 and report["updated"] == 2

    assert cli.main(["export"]) == 0
    after = json.loads(capsys.readouterr().out)
    assert len(after) == 2, "re-importing must not duplicate"
    assert a in {m["id"] for m in after}, "existing ids kept on merge"


def test_import_skips_invalid_records_and_unresolvable_links(tmp_path, monkeypatch,
                                                             capsys):
    _seed(tmp_path, monkeypatch)
    monkeypatch.setenv("TETHER_DB", str(tmp_path / "target.db"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([
        {"id": 1, "type": "user", "title": "Good", "body": "keep me",
         "links": [1, 999]},          # 999 points outside the file
        {"id": 2, "type": "nonsense", "title": "Bad type", "body": "x"},
        {"id": 3, "type": "user", "title": "   ", "body": "blank title"},
        "not even a dict",
    ]))
    capsys.readouterr()
    assert cli.main(["import", str(bad)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["created"] == 1
    assert report["skipped"] == 3
    assert report["dropped_links"] == 1      # 999 unresolvable, self-link ignored
    assert report["linked"] == 0


def test_import_rejects_unreadable_or_wrong_shaped_files(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    assert cli.main(["import", str(tmp_path / "nope.json")]) == 1
    assert "could not read" in capsys.readouterr().err

    notalist = tmp_path / "obj.json"
    notalist.write_text('{"type": "user"}')
    assert cli.main(["import", str(notalist)]) == 1
    assert "not a tether export" in capsys.readouterr().err


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
