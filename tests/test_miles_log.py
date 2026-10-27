"""The rubric's W&B metrics for Miles (`rewards.miles_log`), on fake samples."""

from types import SimpleNamespace

import pytest

from rubricrl.rewards import miles_log


def _sample(record, key="env"):
    return SimpleNamespace(metadata={} if record is None else {key: record})


def _record(score, axes, criteria, **extra):
    return {"score": score, "axes": axes, "criteria": criteria, "chars": 100} | extra


def test_rubric_metrics():
    samples = [
        _sample(
            _record(
                0.5,
                {"accuracy": 1.0, "completeness": 0.0},
                [
                    {"points": 5, "met": True},
                    {"points": 3, "met": False},
                    {"points": -4, "met": True},
                ],
            )
        ),  # fmt: skip
        [
            _sample(
                _record(
                    None,
                    {"accuracy": 0.0},
                    [
                        {"points": 5, "met": None},
                        {"points": -4, "met": False},
                    ],
                ),
                key="rubric",
            )
        ],  # fmt: skip  (rm path; nested groups are flattened)
        _sample(None),  # never graded
    ]
    m = miles_log.rubric_metrics(miles_log.grading_records(samples))
    assert m["missing"] == pytest.approx(1 / 3)
    assert m["unscored"] == 0.5
    assert m["failed_verdicts"] == pytest.approx(1 / 5)
    assert m["positive_met"] == 0.5  # 1 of the 2 positive verdicts returned
    assert m["negative_met"] == 0.5
    assert m["axis/accuracy"] == 0.5 and m["axis/completeness"] == 0.0
    assert m["chars"] == 100 and "turns" not in m
    assert miles_log.rubric_metrics([None, None]) == {}


def test_hooks_never_raise_and_keep_default_logging():
    bad = [SimpleNamespace(metadata={"env": {"criteria": "not a list"}})]
    assert miles_log.log_rollout(0, None, bad, None, 1.0) is False
    assert miles_log.log_eval(0, None, {"val": {"samples": bad}}, None) is False
    assert miles_log.log_eval(0, None, {"val": {"rewards": [0.0]}}) is False


def test_judge_cost_and_timings():
    def rec(cost, calls, hits, env_s):
        usage = {"calls": calls, "cache_hits": hits, "cost_usd": cost}
        r = _record(0.5, {}, [{"points": 1, "met": True}], judge=usage)
        return r | {"policy_s": 10.0, "env_s": env_s}

    samples = [_sample(rec(0.002, 8, 0, 4.0)), _sample(rec(0.001, 4, 4, 8.0))]
    m = miles_log.rubric_metrics(miles_log.grading_records(samples))
    assert m["judge/cost_usd"] == pytest.approx(0.003)
    assert m["judge/calls"] == 12 and m["judge/cache_hit_rate"] == pytest.approx(0.25)
    assert (m["time/env_s_mean"], m["time/env_s_max"]) == (6.0, 8.0)
    assert m["time/policy_s_max"] == 10.0


def test_batch_metrics_and_totals(monkeypatch):
    def sample(group, score, themes, n_tokens):
        rec = _record(score, {}, [])
        md = {"env": rec, "themes": themes, "prompt_id": f"p{group}"}
        return SimpleNamespace(
            metadata=md,
            group_index=group,
            tokens=[0] * n_tokens,
            response_length=n_tokens - 10,
        )

    samples = [
        sample(0, 0.5, ["emergency"], 100), sample(0, 0.5, ["emergency"], 120),
        sample(1, 1.0, ["context"], 50), sample(1, -0.2, ["context"], 70),
    ]  # fmt: skip
    m = miles_log.batch_metrics(samples)
    assert m["score/perfect"] == 0.25 and m["score/nonpositive"] == 0.25
    assert m["groups/no_signal"] == 0.5  # group 0 scored 0.5 twice
    assert m["groups/score_std_mean"] == pytest.approx(0.3)
    assert m["theme/emergency/score"] == 0.5 and m["theme/context/share"] == 0.5
    assert (m["length/total_mean"], m["length/total_max"]) == (85, 120)
    assert -1 <= m["length/score_corr"] <= 1
    longer_pays = [sample(i, i / 10, [], 10 * i + 10) for i in range(5)]
    assert miles_log.batch_metrics(longer_pays)["length/score_corr"] == pytest.approx(1)

    monkeypatch.setattr(miles_log, "_totals",
                        {"judge_usd": 0.0, "tokens": 0, "samples": 0})  # fmt: skip
    monkeypatch.setenv("RUBRICRL_RUN_START", str(miles_log.time.time() - 3600))
    monkeypatch.setenv("RUBRICRL_GPUS", "4")
    monkeypatch.setenv("RUBRICRL_GPU_USD_PER_HOUR", "4.0")
    miles_log.run_totals(samples, 0.5)
    t = miles_log.run_totals(samples, 0.5)
    assert t["samples_total"] == 8 and t["tokens_total"] == 2 * (340 - 40)
    assert t["cost/gpu_hours"] == pytest.approx(4.0, rel=1e-3)
    assert t["cost/total_usd"] == pytest.approx(16.0 + 1.0, rel=1e-3)
