"""Hermetic tests for bench/locomo.py: a two-conversation corpus in LoCoMo's
JSON shape, no download, no model. The real run is `python -m bench.locomo`."""
import json

import pytest
from bench import locomo


def _conv(name, sessions, qa):
    c = {}
    for i, (date, turns) in enumerate(sessions, start=1):
        c[f"session_{i}"] = [
            {"dia_id": f"{name}:{i}:{j}", "speaker": spk, "text": text, **extra}
            for j, (spk, text, extra) in enumerate(turns, start=1)]
        c[f"session_{i}_date_time"] = date
    return {"sample_id": name, "conversation": c, "qa": qa}


CORPUS = [
    _conv("A", [
        ("1 May 2023", [
            ("Caroline", "I finally adopted a greyhound named Biscuit!", {}),
            ("Melanie", "That is wonderful, dogs are the best.", {}),
            ("Caroline", "", {"blip_caption": "a dog sleeping on a sofa"}),
        ]),
        ("9 May 2023", [
            ("Melanie", "My pottery class starts on Thursday.", {}),
            ("Caroline", "Biscuit chewed my running shoes yesterday.", {}),
        ]),
    ], [
        {"question": "What is the name of Caroline's greyhound?",
         "evidence": ["A:1:1"], "category": 4},
        {"question": "What did Biscuit chew?", "evidence": ["A:2:2"], "category": 4},
        {"question": "When does Melanie's pottery class start?",
         "evidence": ["A:2:1"], "category": 2},
        {"question": "Did Caroline win the lottery?", "evidence": ["A:1:1"], "category": 5},
        {"question": "no evidence at all", "evidence": [], "category": 4},
    ]),
    _conv("B", [
        ("3 June 2023", [
            ("Nate", "I am training for the Berlin marathon in September.", {}),
            ("Zoe", "I switched jobs; I work at a bakery now.", {}),
        ]),
    ], [
        {"question": "Which marathon is Nate training for?", "evidence": ["B:1:1"],
         "category": 4},
        {"question": "Where does Zoe work now?", "evidence": ["B:1:2"], "category": 1},
    ]),
]


def test_turns_of_includes_captions_and_dates():
    turns = locomo.turns_of(CORPUS[0])
    assert [t[0] for t in turns] == ["A:1:1", "A:1:2", "A:1:3", "A:2:1", "A:2:2"]
    assert turns[2][3] == " [shared an image: a dog sleeping on a sofa]"
    assert turns[3][2] == "9 May 2023"


def test_questions_of_excludes_adversarial_and_evidence_free():
    qs = locomo.questions_of(CORPUS[0])
    assert [q["question"][:4] for q in qs] == ["What", "What", "When"]


def test_keyword_condition_finds_the_evidence():
    report = locomo.evaluate(CORPUS, embedder=None, conditions=("keyword",))
    assert report["n_questions"] == 5
    assert report["skipped"] == []
    kw = report["conditions"]["keyword"]
    assert kw["R@10"] == 1.0                     # every evidence turn in the top 10
    assert kw["MRR"] >= 0.5
    assert 0.0 <= kw["R@5"] <= 1.0 and kw["ms"] > 0
    cats = kw["by_category"]
    assert cats["single-hop"]["n"] == 3 and cats["temporal"]["n"] == 1
    assert cats["multi-hop"]["n"] == 1


def test_embedder_conditions_are_skipped_without_a_model():
    report = locomo.evaluate(CORPUS, embedder=None, conditions=("keyword", "hybrid", "full"))
    assert report["skipped"] == ["full", "hybrid"]
    assert list(report["conditions"]) == ["keyword"]


def test_hybrid_and_full_run_with_a_fake_embedder():
    from tests.test_bench import FakeEmbedder
    report = locomo.evaluate(CORPUS, embedder=FakeEmbedder(),
                             conditions=("hybrid", "full"))
    assert report["skipped"] == []
    for cond in ("hybrid", "full"):
        assert 0.0 <= report["conditions"][cond]["R@10"] <= 1.0


def test_bm25_condition_is_optional():
    pytest.importorskip("rank_bm25")
    report = locomo.evaluate(CORPUS, conditions=("bm25",))
    assert report["conditions"]["bm25"]["R@10"] == 1.0


def test_format_report_is_a_table():
    report = locomo.evaluate(CORPUS, conditions=("keyword",))
    text = locomo.format_report(report)
    assert "5 questions" in text
    assert "keyword" in text and "R@10" in text and "single-hop" in text


def test_load_locomo_reads_a_local_file_without_downloading(tmp_path, monkeypatch):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(CORPUS))
    monkeypatch.setattr(locomo.urllib.request, "urlretrieve",
                        lambda *a, **k: pytest.fail("must not download"))
    assert len(locomo.load_locomo(str(path))) == 2
    monkeypatch.setenv("TETHER_LOCOMO_PATH", str(path))
    assert len(locomo.load_locomo()) == 2


def test_cli_runs_keyword_only(tmp_path, capsys):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(CORPUS))
    report = locomo.main(["--path", str(path), "--conditions", "keyword",
                          "--no-embedder", "--convs", "1"])
    assert report["n_questions"] == 3
    assert "keyword" in capsys.readouterr().out
