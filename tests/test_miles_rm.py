"""The Miles reward: which half is graded, response cleaning, the log; no Miles."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from rubricrl.data import healthbench as hb
from rubricrl.judge.client import FakeJudge
from rubricrl.rewards import miles_rm


def _metadata(eval: bool) -> dict:
    row = {
        "prompt_id": "p",
        "prompt": [{"role": "user", "content": "q"}],
        "rubrics": [
            {"criterion": f"crit{i}", "points": p, "tags": ["axis:accuracy"]}
            for i, p in enumerate([4, 6, -5])
        ],
        "example_tags": ["theme:hedging"],
    }
    e = hb.parse_example(row)
    return hb.miles_row(e, hb.criteria_split(e), eval)["metadata"]


def test_clean_response():
    assert miles_rm.clean_response("<think>\n\n</think>\n\nHello<|im_end|>") == "Hello"
    assert miles_rm.clean_response("Hi <|endoftext|>") == "Hi"
    assert miles_rm.clean_response(None) == ""


@pytest.mark.parametrize("is_eval", [False, True])
def test_grades_the_right_half(is_eval):
    md = _metadata(is_eval)
    half = md["heldout_criteria" if is_eval else "train_criteria"]
    judge = FakeJudge(rule=lambda p: True)  # every criterion met, negatives too
    rec = asyncio.run(miles_rm.grade_sample(judge, md, "answer<|im_end|>"))
    graded = [p.split("# Rubric item\n")[1].split("\n")[0] for p in judge.prompts]
    assert graded == [f"[{c['points']}] {c['text']}" for c in half]
    positives = sum(c["points"] for c in half if c["points"] > 0)
    assert rec["score"] == pytest.approx((positives - 5) / positives)
    assert rec["eval"] is is_eval and rec["response"] == "answer"
    assert [c["id"] for c in rec["criteria"]] == [c["id"] for c in half]
    assert all(c["met"] is True and "explanation" in c for c in rec["criteria"])


def test_reward_entry_point(tmp_path, monkeypatch):
    log = tmp_path / "rewards.jsonl"
    monkeypatch.setenv("RUBRICRL_LOG", str(log))
    monkeypatch.setattr(miles_rm, "_judge", lambda: FakeJudge(rule=lambda p: None))
    sample = SimpleNamespace(metadata=_metadata(False), response="x", response_length=3)
    reward = asyncio.run(miles_rm.healthbench_reward(None, sample))
    assert reward == 0.0  # no usable verdict
    rec = json.loads(log.read_text())
    assert rec["score"] is None
    assert rec["failed"] == len(sample.metadata["train_criteria"])
    assert rec["tokens"] == 3


def test_judge_usage_counts_only_fresh_calls():
    from rubricrl.judge.client import Verdict

    judge = SimpleNamespace(config=SimpleNamespace(model="claude-haiku-5-5"))
    verdicts = [
        Verdict(True, input_tokens=2000, output_tokens=100),
        Verdict(False, cached=True, input_tokens=2000, output_tokens=100),  # free
    ]
    u = miles_rm.judge_usage(judge, verdicts)
    assert (u["calls"], u["cache_hits"], u["input_tokens"]) == (1, 1, 2000)
    assert u["cost_usd"] == pytest.approx((2000 * 0.10 + 100 * 0.50) / 1e6)
