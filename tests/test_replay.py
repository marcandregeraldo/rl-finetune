"""Replaying a run's rewards through another judge."""

import asyncio

import pytest

from rubricrl.analysis import replay
from rubricrl.judge.client import FakeJudge

CRITERIA = [
    {"id": "p:0", "text": "Says A.", "points": 5, "axis": "accuracy"},
    {"id": "p:1", "text": "Says B.", "points": 5, "axis": "accuracy"},
]
META = {"p": {"prompt_id": "p", "messages": [{"role": "user", "content": "q"}],
              "train_criteria": CRITERIA, "heldout_criteria": []}}  # fmt: skip


def _record(t: float, met: tuple, eval_: bool = False) -> dict:
    return {
        "prompt_id": "p", "eval": eval_, "time": t, "response": "A and B",
        "score": sum(met) / 2,
        "criteria": [c | {"met": m} for c, m in zip(CRITERIA, met, strict=True)],
    }  # fmt: skip


def test_one_step_takes_the_first_training_records():
    records = [_record(3, (1, 1)), _record(1, (1, 0), eval_=True), _record(2, (0, 0))]
    step = replay.one_step(records, verdicts=2)
    assert [r["time"] for r in step] == [2]
    assert [r["time"] for r in replay.one_step(records, 3)] == [2, 3]


def test_replay_compares_verdicts_and_scores():
    records = [_record(1, (True, False)), _record(2, (True, True))]
    judge = FakeJudge(lambda prompt: "Says A." in prompt)  # met A, not B
    out = asyncio.run(replay.replay(judge, records, META))
    assert out["verdicts"] == 4 and out["failed"] == 0
    assert out["agreement"] == pytest.approx(3 / 4)
    assert out["rate_met"] == {"run": 3 / 4, "replay": 1 / 2}
    assert out["score_mean"]["replay"] == pytest.approx(0.5)
