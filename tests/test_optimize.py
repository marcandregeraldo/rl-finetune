"""Judge prompt optimization: the data splits, the GEPA adapter and reflector,
and a whole run with a fake judge and a fake reflection model. No network."""

import json

import pytest

from rubricrl.data.schema import Judgment
from rubricrl.judge import optimize as opt
from rubricrl.judge.client import FakeJudge, Verdict
from rubricrl.judge.prompt import GRADER_TEMPLATE
from rubricrl.llm import FakeLLM

pytest.importorskip("gepa")


def _judgments(n=120, cats=10):
    out = []
    for i in range(n):
        for c in range(cats):
            met = (i + c) % 3 != 0
            labels = (met, met) if i % 5 else (True, False)  # every 5th: a tie
            out.append(
                Judgment(
                    f"{i}:cat{c}",
                    f"p{i}",
                    str(i),
                    [{"role": "user", "content": f"question {i}"}],
                    f"answer {i}",
                    f"rubric {c}",
                    f"cluster:cat{c}",
                    labels,
                    ("x", "y"),
                )  # fmt: skip
            )
    return out


def test_splits_are_balanced_and_disjoint_from_test():
    js = _judgments()
    data = opt.splits(js, n_train=40, n_val=20, test_n=20)
    train, val = data["train"], data["val"]
    assert len(train) == 40 and len(val) == 20
    assert sum(it.majority for it in train) == 20
    assert sum(it.majority for it in val) == 10
    test_prompts = {j.prompt_id for j in data["test"]}
    held = set(data["heldout_categories"])
    for it in train + val:
        assert it.judgment.prompt_id not in test_prompts
        assert it.judgment.category not in held
        assert opt.majority(it.judgment) is not None
    assert not {it.judgment.id for it in train} & {it.judgment.id for it in val}
    assert len(held) == 6
    assert opt.splits(js, n_train=40, n_val=20, test_n=20)["train"] == train


def test_adapter_scores_agreement_and_rejects_broken_templates():
    js = _judgments(10, 1)
    items = [opt.Item(j, m) for j in js if (m := opt.majority(j)) is not None]
    judge = FakeJudge(rule=lambda p: "STRICT" not in p)
    adapter = opt.JudgeAdapter(judge)
    out = adapter.evaluate(items, {opt.COMPONENT: GRADER_TEMPLATE}, capture_traces=True)
    assert out.outputs == [True] * len(items)
    assert out.scores == [1.0 if it.majority else 0.0 for it in items]
    records = adapter.make_reflective_dataset(
        {opt.COMPONENT: GRADER_TEMPLATE}, out, [opt.COMPONENT]
    )[opt.COMPONENT]
    wrong = next(r for r, it in zip(records, items, strict=True) if not it.majority)
    assert "DISAGREES" in wrong["Feedback"] and "0 of 2" in wrong["Feedback"]
    assert wrong["Inputs"]["rubric item"] == "rubric 0"
    broken = adapter.evaluate(items, {opt.COMPONENT: "no placeholders"})
    assert broken.scores == [0.0] * len(items) and len(judge.prompts) == len(items)


def test_feedback_and_agreement():
    j = Judgment("1:a", "p", "1", [], "c", "r", "cluster:a", (True, True, False))
    assert opt.agreement(True, j.physician_labels) == pytest.approx(2 / 3)
    assert opt.agreement(None, j.physician_labels) == 0.0
    assert "2 of 3" in opt.feedback(opt.Item(j, True), Verdict(True))
    assert "no verdict" in opt.feedback(opt.Item(j, True), Verdict(None, error="x"))


def test_run_end_to_end_with_fakes(tmp_path):
    """A strict seed judge; the reflection model proposes a lenient template; the
    optimizer keeps the one that agrees with physicians more."""
    lenient = GRADER_TEMPLATE.replace("# Rubric item", "# Rubric item (LENIENT)")

    def judge_rule(p):  # the seed says "met" for no one; LENIENT follows the truth
        return "LENIENT" in p and "rubric 0" not in p

    def reflect(prompt, schema):
        assert "<<conversation>>" in prompt and "Physicians:" in prompt
        return {"diagnosis": "too strict", "template": lenient}

    out = opt.run(
        name="t", n_train=40, n_val=20, minibatch=4, max_metric_calls=200, test_n=20,
        out=tmp_path, judgments=_judgments(), judge=FakeJudge(rule=judge_rule),
        reflection_llm=_FakeReflection(reflect),
    )  # fmt: skip
    assert (
        out["best_index"] > 0
        and out["best_val"]["val_score"] > out["baseline_val"]["val_score"]
    )
    assert (tmp_path / "t" / "t.txt").read_text() == lenient
    saved = json.loads((tmp_path / "t" / "result.json").read_text())
    assert saved["proposals"][0]["diagnosis"] == "too strict"
    assert saved["candidates"][0]["template"] == GRADER_TEMPLATE


class _FakeReflection(FakeLLM):
    class config:  # noqa: N801
        model = "fake"


def test_json_cache_merge_and_snapshot(tmp_path):
    from rubricrl.llm import JSONCache

    a, b = JSONCache(tmp_path / "a.sqlite"), JSONCache(tmp_path / "b.sqlite")
    a.put("k1", "m", {"x": 1}, (1, 2))
    b.put("k1", "m", {"x": 9}, (1, 2))
    b.put("k2", "m", {"y": 2}, (1, 2))
    assert a.merge(tmp_path / "b.sqlite") == 1
    assert a.get("k1") == {"x": 1} and a.get("k2") == {"y": 2}  # existing kept
    a.snapshot(tmp_path / "c.sqlite")
    assert len(JSONCache(tmp_path / "c.sqlite")) == 2
