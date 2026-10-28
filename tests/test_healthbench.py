"""HealthBench loading, splits and export: deterministic, disjoint, complete."""

import json
from collections import Counter

import pytest

from rubricrl.data import healthbench as hb
from rubricrl.data.schema import Criterion, Example, Judgment

RAW_OK = (hb.RAW / hb.SOURCES["oss_eval"].file).exists()
needs_raw = pytest.mark.skipif(not RAW_OK, reason="run `healthbench download` first")


def _row(pid: str, points: list[int], theme: str = "hedging") -> dict:
    return {
        "prompt_id": pid,
        "prompt": [{"role": "user", "content": f"question {pid}"}],
        "rubrics": [
            {
                "criterion": f"c{i}",
                "points": p,
                "tags": ["level:example", "axis:accuracy"],
            }
            for i, p in enumerate(points)
        ],
        "example_tags": [f"theme:{theme}", "physician_agreed_category:x"],
        "ideal_completions_data": None,
        "canary": "x",
    }


def _example(pid: str, points: list[int], theme: str = "hedging") -> Example:
    return hb.parse_example(_row(pid, points, theme))


def test_parse_example():
    e = _example("p1", [5, -3])
    assert (
        e.id == "p1"
        and e.themes == ("hedging",)
        and e.split in ("train", "val", "test")
    )
    c = e.rubric[1]
    assert (c.id, c.points, c.axis, c.level) == ("p1:1", -3, "accuracy", "example")
    assert str(c) == "[-3] c1"  # what simple-evals puts in the grader prompt


def test_schema_round_trips():
    e = _example("p1", [5, -3, 2])
    assert Example.from_dict(json.loads(json.dumps(e.to_dict()))) == e
    c = e.rubric[0]
    assert Criterion.from_dict(c.to_dict()) == c
    j = Judgment(
        "c:k", "p", "c", e.messages, "resp", "rubric", "k", (True, False), ("a", "b")
    )
    assert Judgment.from_dict(json.loads(json.dumps(j.to_dict()))) == j


def test_conversation_split_is_deterministic_and_seeded():
    ids = [f"id-{i}" for i in range(4000)]
    a = [hb.conversation_split(i) for i in ids]
    assert a == [hb.conversation_split(i) for i in ids]
    counts = Counter(a)
    for name, frac in hb.SPLITS:
        assert abs(counts[name] / len(ids) - frac) < 0.03
    assert a != [hb.conversation_split(i, seed=1) for i in ids]


@pytest.mark.parametrize(
    "points", [[5, 3], [5, 3, 1], [5, -2, 3, -1, 4, 2], [1, 1, 1, 1, -9]]
)
def test_criteria_split_invariants(points):
    e = _example("p", points)
    s = hb.criteria_split(e)
    train, held = {c.id for c in s.train}, {c.id for c in s.heldout}
    pos = {c.id for c in e.rubric if c.points > 0}
    neg = {c.id for c in e.rubric if c.points < 0}
    assert neg <= train and neg <= held  # negatives in both halves
    assert not (train & held & pos)  # positives disjoint
    assert (train | held) == {c.id for c in e.rubric}  # nothing lost
    assert train & pos and held & pos  # at least one positive each
    assert len(train & pos) - len(held & pos) in (0, 1)
    assert [c.id for c in s.train] == [
        c.id for c in e.rubric if c.id in train
    ]  # order kept
    assert hb.criteria_split(e) == s


def test_criteria_split_needs_two_positives():
    assert hb.criteria_split(_example("p", [5, -3])) is None


def test_stratified_subset():
    themes = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
    examples = [_example(f"p{i}", [1, 1], t) for i, t in enumerate(themes)]
    sub = hb.stratified_subset(examples, 20)
    assert len(sub) == 20 and Counter(e.themes[0] for e in sub) == {
        "a": 12,
        "b": 6,
        "c": 2,
    }
    assert sub == hb.stratified_subset(list(reversed(examples)), 20)  # order-free


def test_export_miles(tmp_path):
    examples = [_example(f"p{i}", [3, 2, -1], "hedging") for i in range(300)]
    examples.append(_example("one-positive", [3, -1]))
    ids_path = tmp_path / "eval_ids.txt"
    counts = hb.export_miles(
        examples, tmp_path / "miles", ids_path, n_val=5, n_test=10,
        count_tokens=lambda m: 10,
    )  # fmt: skip
    assert counts["dropped_one_positive"] == 1 and counts["dropped_long_prompt"] == 0
    assert (
        counts["healthbench_test.jsonl"] == 10 and counts["healthbench_val.jsonl"] == 5
    )
    train = [json.loads(x) for x in (tmp_path / "miles/healthbench_train.jsonl").open()]
    test = [json.loads(x) for x in (tmp_path / "miles/healthbench_test.jsonl").open()]
    row = train[0]
    assert row["prompt"] == row["metadata"]["messages"]
    assert row["metadata"]["eval"] is False and test[0]["metadata"]["eval"] is True
    train_ids = {r["label"] for r in train}
    test_ids = {r["label"] for r in test}
    assert not (train_ids & test_ids)
    assert hb.read_ids(ids_path) == [r["label"] for r in test]

    # A second export reuses the frozen ids, and refuses ids that aren't eligible.
    hb.export_miles(
        examples, tmp_path / "m2", ids_path, n_val=5, count_tokens=lambda m: 10
    )
    assert (tmp_path / "m2/healthbench_test.jsonl").read_text() == (
        tmp_path / "miles/healthbench_test.jsonl"
    ).read_text()
    ids_path.write_text("not-a-test-id\n")
    with pytest.raises(ValueError, match="not eligible"):
        hb.export_miles(examples, tmp_path / "m3", ids_path, count_tokens=lambda m: 10)


@needs_raw
def test_real_data():
    examples = hb.load_examples()
    assert len(examples) == 5000 and len({e.id for e in examples}) == 5000
    assert sum(len(e.rubric) for e in examples) == 57237
    assert all(e.messages[-1]["role"] == "user" for e in examples)
    ids = hb.read_ids(hb.EVAL_IDS)
    by_id = {e.id: e for e in examples}
    assert len(ids) == 200 and all(by_id[i].split == "test" for i in ids)
    assert all(hb.criteria_split(by_id[i]) is not None for i in ids)


@needs_raw
def test_rubrics_match_the_source():
    """Every criterion as in the raw file; the halves partition each real rubric."""
    rows = hb._read_jsonl(hb.RAW / hb.SOURCES["oss_eval"].file)
    examples = {e.id: e for e in hb.load_examples()}
    for row in rows:
        e = examples[row["prompt_id"]]
        assert e.messages == row["prompt"]
        assert [(c.text, c.points, list(c.tags)) for c in e.rubric] == [
            (r["criterion"], r["points"], r["tags"]) for r in row["rubrics"]
        ]
        assert all(sum(t.startswith("axis:") for t in c.tags) == 1 for c in e.rubric)
        split = hb.criteria_split(e)
        if split is None:
            continue
        train, heldout = {c.id for c in split.train}, {c.id for c in split.heldout}
        negatives = {c.id for c in e.rubric if c.points < 0}
        assert train | heldout == {c.id for c in e.rubric}
        assert train & heldout == negatives
    # The paper's "48,562 unique criteria": cluster-level criteria repeat.
    assert len({c.text for e in examples.values() for c in e.rubric}) == 48562
