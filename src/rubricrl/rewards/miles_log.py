"""Rubric metrics in Miles' W&B run, next to its own `rollout/*` and `eval/*`:

    --custom-rollout-log-function-path rubricrl.rewards.miles_log.log_rollout
    --custom-eval-rollout-log-function-path rubricrl.rewards.miles_log.log_eval

Miles logs the reward (`rollout/raw_reward`, `eval/<dataset>`), lengths,
truncation, the losses, grad norm, KL, learning rate and timings itself. These
add what the grading records know: `sample.metadata["env"]` (the environment's
episode record, `--reward env`) or `["rubric"]` (`miles_rm`, `--reward rm`).
Training metrics are `rollout/rubric/*`; eval metrics `eval/<dataset>/rubric/*`.

- `axis/<axis>`: the score on each HealthBench axis (accuracy, completeness,
  ...), averaged over the samples with a positive criterion on that axis.
- `positive_met`: share of positive criteria met. `negative_met`: share of
  negative (penalty) criteria met, e.g. harmful or wrong advice; it should not
  rise while the reward does.
- `failed_verdicts`: share of verdicts the judge failed to return.
  `unscored`: share of samples with no usable score (reward 0.0).
  `missing`: share of samples with no grading record (environment errors).
- `chars`: mean answer length in characters; `turns`: doctor turns (patient mode).
- `judge/cost_usd`: what grading this batch cost (sum); `judge/calls`,
  `judge/cache_hit_rate`.
- `time/policy_s_*`, `time/env_s_*` (`--reward env`): per episode, the policy's
  generation time and the environment's (judge, patient), mean and max: which
  one a slow step waited on. Miles also logs the latter as
  `perf/non_generation_time/*`.

And per batch, from the samples (`rollout/run/*`, `eval/<dataset>/run/*`), the
counterparts of the panels Xiaomi pinned on its public MiMo-V2.6 RL dashboard
(passrate zero/one, batch composition by source, tokens per step, context
length, running cost):

- `score/perfect`, `score/nonpositive`: shares of samples scoring >= 0.999 and
  <= 0 (MiMo's passrate one/zero, for a graded score).
- `groups/no_signal`: share of prompt groups whose samples all scored the same:
  zero advantage, nothing learned from them. `groups/score_std_mean`: the spread
  GRPO learns from.
- `theme/<theme>/score`, `theme/<theme>/share`: per HealthBench theme (the
  batch composition), from the row's `themes`.
- `length/total_mean`, `length/total_max`: prompt + response tokens.
  `length/score_corr`: Pearson correlation of response tokens and score in the
  batch; a rising one says longer answers are what pays (length hacking).
- Training only, since the run started: `tokens_step`, `tokens_total`,
  `samples_total`; `cost/gpu_hours`, `cost/gpu_usd` (wall time x GPUs x
  RUBRICRL_GPU_USD_PER_HOUR, an estimate), `cost/judge_usd`, `cost/total_usd`.
  RUBRICRL_RUN_START and RUBRICRL_GPUS come from `modal/miles_rft.py`.

Both hooks return False, so Miles' default logging still runs; a failure here is
logged and never stops training.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)
_totals = {"judge_usd": 0.0, "tokens": 0, "samples": 0}  # in the rollout process


def _flat(samples: Any) -> list:
    out = []
    for s in samples or []:
        out.extend(_flat(s) if isinstance(s, list) else [s])
    return out


def grading_records(samples: Any) -> list[dict | None]:
    """One grading record per sample (None: the sample was never graded)."""
    out = []
    for s in _flat(samples):
        metadata = getattr(s, "metadata", None) or {}
        out.append(metadata.get("env") or metadata.get("rubric"))
    return out


def rubric_metrics(records: list[dict | None]) -> dict[str, float]:
    graded = [r for r in records if r is not None]
    if not graded:
        return {}
    criteria = [c for r in graded for c in r.get("criteria") or []]
    out = {
        "missing": 1 - len(graded) / len(records),
        "unscored": mean(r.get("score") is None for r in graded),
        "chars": mean(r.get("chars") or 0 for r in graded),
    }
    if criteria:
        out["failed_verdicts"] = mean(c.get("met") is None for c in criteria)
    for name, positive in (("positive_met", True), ("negative_met", False)):
        met = [
            bool(c["met"])
            for c in criteria
            if c.get("met") is not None
            and (c.get("points") or 0) != 0
            and ((c.get("points") or 0) > 0) == positive
        ]
        if met:
            out[name] = mean(met)
    axes: dict[str, list[float]] = defaultdict(list)
    for r in graded:
        for axis, value in (r.get("axes") or {}).items():
            axes[axis].append(value)
    out |= {f"axis/{a}": mean(v) for a, v in sorted(axes.items())}
    if any(r.get("mode") == "patient" for r in graded):
        out["turns"] = mean(r.get("turns") or 0 for r in graded)
    usage = [r["judge"] for r in graded if r.get("judge")]
    if usage:
        calls = sum(u.get("calls", 0) for u in usage)
        hits = sum(u.get("cache_hits", 0) for u in usage)
        out["judge/cost_usd"] = sum(u.get("cost_usd", 0.0) for u in usage)
        out["judge/calls"] = calls
        out["judge/cache_hit_rate"] = hits / (calls + hits) if calls + hits else 0.0
    for key in ("policy_s", "env_s"):
        times = [r[key] for r in graded if r.get(key) is not None]
        if times:
            out[f"time/{key}_mean"], out[f"time/{key}_max"] = mean(times), max(times)
    return {k: float(v) for k, v in out.items()}


def _score(sample, record: dict | None) -> float | None:
    if record is not None:
        return record.get("score")
    reward = getattr(sample, "reward", None)
    return reward if isinstance(reward, (int, float)) else None


def batch_metrics(samples: Any) -> dict[str, float]:
    """Per-batch shares, groups, themes and lengths, from the samples."""
    flat = _flat(samples)
    if not flat:
        return {}
    scored = []  # (sample, score)
    for s, r in zip(flat, grading_records(flat), strict=True):
        if (score := _score(s, r)) is not None:
            scored.append((s, score))
    out: dict[str, float] = {}
    if scored:
        out["score/perfect"] = mean(x >= 0.999 for _, x in scored)
        out["score/nonpositive"] = mean(x <= 0 for _, x in scored)
    groups: dict[Any, list[float]] = defaultdict(list)
    for s, x in scored:
        md = getattr(s, "metadata", None) or {}
        key = getattr(s, "group_index", None)
        groups[md.get("prompt_id") if key is None else key].append(x)
    spreads = [_std(v) for v in groups.values() if len(v) > 1]
    if spreads:
        out["groups/no_signal"] = mean(sd < 1e-6 for sd in spreads)
        out["groups/score_std_mean"] = mean(spreads)
    themes: dict[str, list[float]] = defaultdict(list)
    for s, x in scored:
        for theme in (getattr(s, "metadata", None) or {}).get("themes") or []:
            themes[theme].append(x)
    for theme, xs in sorted(themes.items()):
        out[f"theme/{theme}/score"] = mean(xs)
        out[f"theme/{theme}/share"] = len(xs) / len(flat)
    pairs = [(getattr(s, "response_length", 0) or 0, x) for s, x in scored]
    if (corr := _corr(pairs)) is not None:
        out["length/score_corr"] = corr
    totals = [len(s.tokens) for s in flat if getattr(s, "tokens", None)]
    if totals:
        out["length/total_mean"], out["length/total_max"] = mean(totals), max(totals)
    return {k: float(v) for k, v in out.items()}


def _corr(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    sx, sy = _std(xs), _std(ys)
    if sx == 0 or sy == 0:
        return None
    mx, my = mean(xs), mean(ys)
    return mean((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def _std(xs: list[float]) -> float:
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def run_totals(samples: Any, judge_usd: float) -> dict[str, float]:
    """Training totals since the run started, updated with this batch."""
    flat = _flat(samples)
    tokens = sum(getattr(s, "response_length", 0) or 0 for s in flat)
    _totals["tokens"] += tokens
    _totals["samples"] += len(flat)
    _totals["judge_usd"] += judge_usd
    out = {
        "tokens_step": tokens,
        "tokens_total": _totals["tokens"],
        "samples_total": _totals["samples"],
        "cost/judge_usd": _totals["judge_usd"],
    }
    start = os.environ.get("RUBRICRL_RUN_START")
    if start:
        gpu_hours = (
            (time.time() - float(start))
            / 3600
            * int(os.environ.get("RUBRICRL_GPUS", "1"))
        )
        gpu_usd = gpu_hours * float(os.environ.get("RUBRICRL_GPU_USD_PER_HOUR", "0"))
        out |= {
            "cost/gpu_hours": gpu_hours,
            "cost/gpu_usd": gpu_usd,
            "cost/total_usd": gpu_usd + _totals["judge_usd"],
        }
    return {k: float(v) for k, v in out.items()}


def _log(args, rollout_id: int, metrics: dict[str, float], step_key: str) -> None:
    if not metrics:
        return
    from miles.utils.metric_utils import compute_rollout_step
    from miles.utils.tracking_utils import tracking

    metrics[step_key] = compute_rollout_step(args, rollout_id)
    logger.info(f"rubric {rollout_id}: {metrics}")
    tracking.log(args, metrics, step_key=step_key)


def log_rollout(rollout_id, args, samples, extra_metrics=None, rollout_time=None):
    try:
        rubric = rubric_metrics(grading_records(samples))
        run = batch_metrics(samples)
        run |= run_totals(samples, rubric.get("judge/cost_usd", 0.0))
        metrics = {f"rollout/rubric/{k}": v for k, v in rubric.items()}
        metrics |= {f"rollout/run/{k}": v for k, v in run.items()}
        _log(args, rollout_id, metrics, "rollout/step")
    except Exception:
        logger.warning("rubric rollout metrics failed", exc_info=True)
    return False


def log_eval(rollout_id, args, data, extra_metrics=None):
    try:
        metrics = {}
        for key, d in (data or {}).items():
            samples = d.get("samples") if isinstance(d, dict) else None
            for k, v in rubric_metrics(grading_records(samples)).items():
                metrics[f"eval/{key}/rubric/{k}"] = v
            for k, v in batch_metrics(samples).items():
                metrics[f"eval/{key}/run/{k}"] = v
        _log(args, rollout_id, metrics, "eval/step")
    except Exception:
        logger.warning("rubric eval metrics failed", exc_info=True)
    return False
