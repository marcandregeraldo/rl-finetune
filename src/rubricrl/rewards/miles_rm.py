"""The HealthBench reward for Miles, the function `--custom-rm-path` names:

    --custom-rm-path rubricrl.rewards.miles_rm.healthbench_reward


Miles calls `await healthbench_reward(args, sample)` once per sample, right after
SGLang finishes it, for training and eval samples alike (the call doesn't say
which). Each row's metadata, from `healthbench.export_miles`, carries what's needed:

- `metadata.eval` false (training): graded on `train_criteria`, the reward.
- `metadata.eval` true (in-loop eval): graded on `heldout_criteria`, so
  `eval/healthbench_val` in W&B is the held-out score under the training judge.

The score is the unclipped rubric score (`rewards.rubric.score`). A sample with no
usable verdict scores 0.0 and is counted in the log.

Configured by environment variables (set in `modal/miles_rft.py`):

    RUBRICRL_JUDGE        haiku | sonnet | a model id   (default haiku)
    RUBRICRL_EFFORT       low | medium | ...            (default low)
    RUBRICRL_THINKING     1 = adaptive thinking         (default off)
    RUBRICRL_CONCURRENCY  judge requests in flight      (default 64)
    RUBRICRL_CACHE        SQLite verdict cache          (default ~/.cache/rubricrl)
    RUBRICRL_LOG          JSONL, one line per sample    (default: no log)

Each log line has every criterion's verdict and explanation; `sync_traces` in
`modal/miles_rft.py` sends the log to W&B Weave (`rubricrl.observe`).

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time

from rubricrl.data.schema import Criterion
from rubricrl.judge.client import ClaudeJudge, JudgeConfig, VerdictCache
from rubricrl.rewards.rubric import axis_scores, score

logger = logging.getLogger(__name__)

# Qwen chat-template leftovers SGLang may return (no_stop_trim, special tokens kept).
_SPECIAL = re.compile(r"<\|(?:im_end|endoftext|im_start)\|>")
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

_judges: dict[int, ClaudeJudge] = {}  # one per event loop: the client binds to it
_cache: VerdictCache | None = None
_log_lock = threading.Lock()


def clean_response(text: str) -> str:
    """The answer the user would see: no empty think block, no special tokens."""
    return _SPECIAL.sub("", _THINK.sub("", text or "")).strip()


def _judge() -> ClaudeJudge:
    global _cache
    loop = id(asyncio.get_running_loop())
    if loop not in _judges:
        if _cache is None:
            path = os.environ.get("RUBRICRL_CACHE")
            _cache = VerdictCache(path) if path else VerdictCache()
        config = JudgeConfig.named(
            os.environ.get("RUBRICRL_JUDGE", "haiku"),
            effort=os.environ.get("RUBRICRL_EFFORT", "low"),
            thinking=os.environ.get("RUBRICRL_THINKING", "") == "1",
            concurrency=int(os.environ.get("RUBRICRL_CONCURRENCY", "64")),
        )
        _judges[loop] = ClaudeJudge(config, _cache)
    return _judges[loop]


def _log(record: dict) -> None:
    path = os.environ.get("RUBRICRL_LOG")
    if not path:
        return
    with _log_lock, open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


async def grade_sample(judge, metadata: dict, response: str) -> dict:
    """Grade one response on its half of the rubric. Returns the log record."""
    is_eval = bool(metadata.get("eval"))
    key = "heldout_criteria" if is_eval else "train_criteria"
    criteria = [Criterion.from_dict(c) for c in metadata[key]]
    answer = clean_response(response)
    verdicts = await judge.grade_criteria(metadata["messages"], answer, criteria)
    met = [v.met for v in verdicts]
    s = score(criteria, met)
    return {
        "prompt_id": metadata.get("prompt_id"),
        "eval": is_eval,
        "score": s,
        "reward": 0.0 if s is None else s,
        "axes": axis_scores(criteria, met),
        "criteria": [
            {
                "id": c.id,
                "text": c.text,
                "points": c.points,
                "met": v.met,
                "explanation": v.explanation,
            }
            for c, v in zip(criteria, verdicts, strict=True)
        ],
        "failed": sum(m is None for m in met),
        "chars": len(answer),
        "response": answer,
    }


async def healthbench_reward(args, sample, **kwargs) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    record = await grade_sample(_judge(), metadata, sample.response or "")
    if record["score"] is None:
        logger.warning("no usable verdict for %s: reward 0.0", record["prompt_id"])
    record |= {
        "time": time.time(),
        "tokens": getattr(sample, "response_length", None),
        "status": str(getattr(sample, "status", "")),
    }
    _log(record)
    return record["reward"]
