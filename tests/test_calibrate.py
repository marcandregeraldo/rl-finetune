"""Calibration metrics, checked by hand and against simple-evals' definitions."""

import asyncio

import pytest

from rubricrl.analysis import calibrate as cal
from rubricrl.data.schema import Judgment
from rubricrl.judge.client import (
    FakeJudge,
    JudgeConfig,
    Verdict,
    VerdictCache,
    cache_key,
)
from rubricrl.judge.prompt import render


def test_pairwise_f1_by_hand():
    # rater, reference: TP=2, FN=1, FP=1, TN=1
    pairs = [(True, True), (True, True), (False, True), (True, False), (False, False)]
    m = cal.pairwise_f1(pairs)
    assert m["precision_pos"] == m["recall_pos"] == pytest.approx(2 / 3)
    assert m["precision_neg"] == m["recall_neg"] == pytest.approx(1 / 2)
    assert m["f1_balanced"] == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_pairwise_f1_one_class_missing():
    m = cal.pairwise_f1([(True, True), (True, True)])
    assert m["f1_pos"] == 1.0 and m["f1_neg"] is None and m["f1_balanced"] is None


def _j(
    i: int, labels: tuple, cat: str = "cluster:a", ids: tuple = ("x", "y")
) -> Judgment:
    return Judgment(f"{i}:{cat}", "p", str(i), [], "c" * i, "r", cat, labels, ids)


def test_judge_metrics_counts_every_physician_pair():
    js = [_j(1, (True, False)), _j(2, (True, True), "cluster:b")]
    m = cal.judge_metrics(js, [True, None])
    assert m["overall"]["n_pairs"] == 2 and "cluster:b" not in m


def test_physician_metrics():
    # Physicians x and y disagree on item 3; each is scored against the other.
    js = [
        _j(1, (True, True)),
        _j(2, (False, False)),
        _j(3, (True, False)),
        _j(4, (True, True)),
    ]
    m = cal.physician_metrics(js)["overall"]
    assert m["physicians"] == 2
    x = cal.pairwise_f1([(True, True), (False, False), (True, False), (True, True)])
    y = cal.pairwise_f1([(True, True), (False, False), (False, True), (True, True)])
    assert m["f1_balanced_mean"] == pytest.approx(
        (x["f1_balanced"] + y["f1_balanced"]) / 2
    )


def test_cohen_kappa():
    a, b = [True, False, True, False], [True, True, False, False]
    assert cal.cohen_kappa(a, a) == 1.0
    assert cal.cohen_kappa(a, b) == 0.0
    assert cal.cohen_kappa([True, True], [True, True]) is None


def test_sample_judgments_is_stratified_and_nested():
    js = [_j(i, (True,), f"cluster:{'ab'[i % 2]}", ("x",)) for i in range(100)]
    small, big = cal.sample_judgments(js, 10), cal.sample_judgments(js, 20)
    assert len(small) == 10 and sum(j.category == "cluster:a" for j in small) == 5
    # Nested: a smaller re-run is all cache hits.
    assert {j.id for j in small} <= {j.id for j in big}


def _cache_all(cache, config, js, retest):
    for repeat, items in ((0, js), (1, js[:retest])):
        for j in items:
            prompt = render(j.messages, j.completion, j.rubric)
            cache.put(
                cache_key(config, prompt, repeat), config.model, Verdict(True), (1, 1)
            )


def test_cache_coverage(tmp_path):
    js = [_j(i, (True, False)) for i in range(1, 6)]
    config, cache = JudgeConfig.named("haiku"), VerdictCache(tmp_path / "c.sqlite")
    assert cal.cache_coverage(config, js, 2, cache) == (0, 7)
    _cache_all(cache, config, js[:3], 2)
    assert cal.cache_coverage(config, js, 2, cache) == (5, 7)


def _run(tmp_path, monkeypatch, js, **kw):
    monkeypatch.setattr(cal.healthbench, "load_judgments", lambda root: js)
    monkeypatch.setattr(
        cal, "ClaudeJudge", lambda config, cache: FakeJudge(lambda p: True)
    )
    return asyncio.run(
        cal.run(
            n=len(js), retest=2, cache=str(tmp_path / "c.sqlite"), out=tmp_path, **kw
        )
    )


def test_require_cached_stops_before_any_call(tmp_path, monkeypatch):
    js = [_j(i, (True, False)) for i in range(1, 5)]
    monkeypatch.setattr(cal, "grade_all", None)  # any grading would fail
    with pytest.raises(SystemExit, match="6 verdicts not cached"):
        _run(tmp_path, monkeypatch, js, require_cached=True)


def test_run_traces_every_call_and_logs_a_summary(tmp_path, monkeypatch):
    from rubricrl import observe

    js = [_j(i, (True, False)) for i in range(1, 5)]
    logged, traced = {}, []
    monkeypatch.setattr(observe, "init", lambda project: logged.update(project=project))
    monkeypatch.setattr(observe, "publish_judgments", lambda subset, seed: "ds")
    monkeypatch.setattr(observe, "publish_prompt", lambda: None)

    def traced_grader(judge, attributes):
        async def grade(judgments, repeat):
            out = []
            for j in judgments:
                traced.append((j.id, repeat, attributes["run"]))
                out.append(
                    await judge.grade(j.messages, j.completion, j.rubric, repeat)
                )
            return out

        return grade

    def summary(result, attributes, dataset):
        logged.update(n=attributes["n"], dataset=dataset)
        return "https://wandb.ai/x"

    monkeypatch.setattr(observe, "traced_grader", traced_grader)
    monkeypatch.setattr(observe, "log_calibration_summary", summary)
    result = _run(tmp_path, monkeypatch, js, weave_project="proj")
    assert result["weave_url"] == "https://wandb.ai/x"
    assert logged == {"project": "proj", "n": 4, "dataset": "ds"}
    assert len(traced) == 4 + 2  # every call, retest pass included
    assert {r for _, r, _ in traced} == {0, 1}
    assert traced[0][2] == "claude-haiku-5-5-low"
    assert result["judge_rate_met"] == 1.0
