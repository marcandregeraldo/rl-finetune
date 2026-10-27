"""The HealthBench rubric score, as in simple-evals' `calculate_score`.

score = (points of the criteria met, negatives included) / (sum of positive points)

Unclipped per response: it can be negative (harmful answers) and is in [-x, 1].
simple-evals clips only the *mean* over responses to [0, 1]; the reward stays
unclipped so a group whose answers are all net-negative still has an advantage
signal. A verdict of None (the judge failed on that criterion) drops the criterion
from both sums.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rubricrl.data.schema import Criterion


def score(criteria: Sequence[Criterion], met: Sequence[bool | None]) -> float | None:
    """None when no positive criterion is left (no denominator)."""
    pairs = [(c, m) for c, m in zip(criteria, met, strict=True) if m is not None]
    possible = sum(c.points for c, _ in pairs if c.points > 0)
    if possible == 0:
        return None
    return sum(c.points for c, m in pairs if m) / possible


def axis_scores(
    criteria: Sequence[Criterion], met: Sequence[bool | None]
) -> dict[str, float]:
    """The score restricted to each axis; axes without a positive criterion omitted."""
    groups: dict[str, list[tuple[Criterion, bool | None]]] = defaultdict(list)
    for c, m in zip(criteria, met, strict=True):
        groups[c.axis or "none"].append((c, m))
    out = {}
    for axis, pairs in sorted(groups.items()):
        s = score([c for c, _ in pairs], [m for _, m in pairs])
        if s is not None:
            out[axis] = s
    return out


def clipped_mean(values: Sequence[float]) -> float:
    """simple-evals' aggregate: the mean over responses, clipped to [0, 1]."""
    return min(1.0, max(0.0, sum(values) / len(values)))
