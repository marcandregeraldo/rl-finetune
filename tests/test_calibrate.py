"""Calibration metrics, checked by hand and against simple-evals' definitions."""

import pytest

from rubricrl.analysis import calibrate as cal
from rubricrl.data.schema import Judgment


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
