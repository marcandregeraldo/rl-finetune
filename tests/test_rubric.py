"""The rubric score on fixtures computed by hand."""

import pytest

from rubricrl.data.schema import Criterion
from rubricrl.rewards.rubric import axis_scores, clipped_mean, score


def _c(points: int, axis: str = "accuracy") -> Criterion:
    return Criterion(id=f"x:{points}:{axis}", text="t", points=points, axis=axis)


CRITERIA = [_c(10), _c(5, "completeness"), _c(-4), _c(-8, "completeness")]


@pytest.mark.parametrize(
    ("met", "expected"),
    [
        ([True, True, False, False], 1.0),
        ([True, False, False, False], 10 / 15),
        ([False, False, True, False], -4 / 15),  # negative, unclipped
        ([True, True, True, True], 3 / 15),
        ([False, False, True, True], -12 / 15),
        ([None, True, False, None], 1.0),  # failed verdicts drop out of both sums
    ],
)
def test_score(met, expected):
    assert score(CRITERIA, met) == pytest.approx(expected)


def test_score_without_positive_points_is_none():
    assert score([_c(-4)], [True]) is None
    assert score([_c(10), _c(-4)], [None, True]) is None


def test_axis_scores():
    axes = axis_scores(CRITERIA, [True, False, False, True])
    assert axes == {"accuracy": 1.0, "completeness": pytest.approx(-8 / 5)}
    assert axis_scores([_c(-4)], [True]) == {}


def test_clipped_mean_matches_simple_evals():
    assert clipped_mean([-0.5, 0.2]) == 0.0
    assert clipped_mean([0.4, 0.6]) == pytest.approx(0.5)
    assert clipped_mean([1.0, 1.0]) == 1.0
