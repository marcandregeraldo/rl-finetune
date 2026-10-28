"""Replay a training run's rewards through another judge: one step's worth of real
policy replies, graded on the exact reward path (`rewards.miles_rm.grade_sample`).

Two things a calibration on the meta-eval can't show: how long the judge takes
on what a training step actually sends it (each reply's ~8 criteria share the
conversation, so a server with prefix caching reuses it), and how often it
agrees with the judge the run was trained on, verdict by verdict and reward by
reward, on the policy's own replies.

    python -m rubricrl.analysis.replay RUN/rewards.jsonl --judge gpt-oss-20b \\
        --base-url http://127.0.0.1:8004 --verdicts 1500 --out replay.json

On Modal, next to the judge's server: `modal run modal/miles_rft.py::calibrate_judge
--replay RUN_ID`.
"""

import asyncio
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import click

from rubricrl.analysis.calibrate import cohen_kappa
from rubricrl.judge.client import JudgeConfig, VerdictCache, make_judge
from rubricrl.judge.prompt import REWARD_PROMPT_VERSION
from rubricrl.rewards.miles_rm import grade_sample


def one_step(records: Sequence[dict], verdicts: int) -> list[dict]:
    """The run's first training records, in the order they were graded, until
    they hold `verdicts` verdicts: about one step's worth."""
    train = sorted((r for r in records if not r["eval"]), key=lambda r: r["time"])
    out, total = [], 0
    for r in train:
        if total >= verdicts:
            break
        out.append(r)
        total += len(r["criteria"])
    return out


async def replay(judge, records: Sequence[dict], metadata: dict[str, dict]) -> dict:
    """Grade `records`' responses again with `judge`; compare with their verdicts."""
    t0 = time.monotonic()
    new = await asyncio.gather(
        *(grade_sample(judge, metadata[r["prompt_id"]], r["response"]) for r in records)
    )
    wall = time.monotonic() - t0
    pairs = [
        (a["met"], b["met"])
        for old, rec in zip(records, new, strict=True)
        for a, b in zip(old["criteria"], rec["criteria"], strict=True)
        if a["met"] is not None and b["met"] is not None
    ]
    scores = [
        (old["score"], rec["score"])
        for old, rec in zip(records, new, strict=True)
        if old["score"] is not None and rec["score"] is not None
    ]
    n = sum(len(rec["criteria"]) for rec in new)
    return {
        "records": len(records),
        "verdicts": n,
        "failed": sum(rec["failed"] for rec in new),
        "wall_s": round(wall, 1),
        "verdicts_per_s": round(n / wall, 2) if wall else None,
        "agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        "kappa": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]),
        "rate_met": {
            "run": statistics.fmean(a for a, _ in pairs) if pairs else None,
            "replay": statistics.fmean(b for _, b in pairs) if pairs else None,
        },
        "score_mean": {
            "run": statistics.fmean(a for a, _ in scores) if scores else None,
            "replay": statistics.fmean(b for _, b in scores) if scores else None,
        },
        "score_pearson": statistics.correlation(*zip(*scores, strict=True))
        if len(scores) > 2
        else None,
        # "<record>:<criterion id>" -> [the run's verdict, the replay's]: to
        # compare judges (the 8 replies of a group share criterion ids)
        "verdicts_by_id": {
            f"{i}:{a['id']}": [a["met"], b["met"]]
            for i, (old, rec) in enumerate(zip(records, new, strict=True))
            for a, b in zip(old["criteria"], rec["criteria"], strict=True)
        },
    }


def load_metadata(path: Path) -> dict[str, dict]:
    """Miles rows (`data/miles/*.jsonl`) by prompt id: the reward's metadata."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {r["metadata"]["prompt_id"]: r["metadata"] for r in rows}


@click.command(help=__doc__.split("\n\n")[0])
@click.argument("log", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/miles/healthbench_train.jsonl"),
    show_default=True,
)
@click.option("--judge", "judge_name", required=True)
@click.option("--base-url", default=None)
@click.option("--effort", default="low", show_default=True)
@click.option("--prompt-version", default=REWARD_PROMPT_VERSION, show_default=True)
@click.option("--concurrency", default=256, show_default=True)
@click.option("--verdicts", default=1500, show_default=True, help="about one step")
@click.option("--cache", default=None, help="SQLite path (default: a fresh one)")
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(
    log: Path,
    data: Path,
    judge_name: str,
    base_url: str | None,
    effort: str,
    prompt_version: str,
    concurrency: int,
    verdicts: int,
    cache: str | None,
    out: Path | None,
) -> None:
    records = one_step(
        [json.loads(line) for line in log.read_text().splitlines()], verdicts
    )
    config = JudgeConfig.named(
        judge_name,
        effort=effort,
        concurrency=concurrency,
        prompt_version=prompt_version,
        base_url=base_url,
    )
    path = cache or str(Path(f"/tmp/replay-{time.time_ns()}.sqlite"))
    judge = make_judge(config, VerdictCache(path))
    result = {"judge": judge_name, "prompt_version": prompt_version, "log": str(log)}
    result |= asyncio.run(replay(judge, records, load_metadata(data)))
    print(json.dumps(result, indent=1))
    if out is not None:
        out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
