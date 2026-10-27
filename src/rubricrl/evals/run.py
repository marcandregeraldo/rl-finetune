"""Score sampled responses on HealthBench, the same way for every model and checkpoint.

Input: a samples JSONL from `modal/sample.py` (one line per response: prompt_id,
sample, response, finish_reason, completion_tokens). Every response is graded on
its **full** rubric by each judge, once per criterion; the training half and the
held-out half (`healthbench.criteria_split`) are subsets, so one pass gives all
three scores.

What separates learning the task from learning the grader is in `compare`:

- **gain on training criteria vs gain on held-out criteria.** The two halves have
  different criteria, so compare their *gains*, not their levels.
- **gain under the training judge (Haiku) vs under the held-out judge (Sonnet).**

CIs are 95% bootstrap over conversations (a conversation's K samples averaged
first); `compare` pairs conversations.

    python -m rubricrl.evals.run score runs/samples/qwen3.5-4b.jsonl   # haiku, sonnet:1
    python -m rubricrl.evals.run table runs/evals/qwen3.5-4b runs/evals/rft-60
    python -m rubricrl.evals.run compare runs/evals/qwen3.5-4b runs/evals/rft-60
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import click
import numpy as np

from rubricrl.data import healthbench
from rubricrl.data.schema import Example
from rubricrl.judge.client import PRICES, ClaudeJudge, JudgeConfig, VerdictCache
from rubricrl.judge.prompt import render
from rubricrl.rewards.miles_rm import clean_response
from rubricrl.rewards.rubric import axis_scores, clipped_mean, score

OUT = Path("runs/evals")
DEFAULT_JUDGES = ("haiku", "sonnet:1")  # Sonnet on sample 0 only: ~15x the price
HALVES = ("full", "train", "heldout")
N_BOOT = 2000


def load_samples(path: Path, max_samples: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if max_samples is not None:
        rows = [r for r in rows if r["sample"] < max_samples]
    return rows


def grade_record(example: Example, met: dict[str, bool | None]) -> dict:
    """Scores for one response under one judge, from its verdicts by criterion id."""
    split = healthbench.criteria_split(example)
    halves = {"full": example.rubric, "train": split.train, "heldout": split.heldout}
    out = {}
    for name, criteria in halves.items():
        out[name] = score(criteria, [met[c.id] for c in criteria])
    out["axes"] = axis_scores(example.rubric, [met[c.id] for c in example.rubric])
    out["failed"] = sum(m is None for m in met.values())
    return out


async def score_samples(
    samples: list[dict], examples: dict[str, Example], judge
) -> list[dict[str, bool | None]]:
    """Verdicts by criterion id, per sample."""
    done, t0 = 0, time.monotonic()

    async def one(s: dict) -> dict[str, bool | None]:
        nonlocal done
        e = examples[s["prompt_id"]]
        verdicts = await judge.grade_criteria(
            e.messages, clean_response(s["response"]), e.rubric
        )
        done += 1
        if done % 25 == 0 or done == len(samples):
            print(
                f"  {done}/{len(samples)} responses, {time.monotonic() - t0:.0f}s",
                flush=True,
            )
        return {c.id: v.met for c, v in zip(e.rubric, verdicts, strict=True)}

    return list(await asyncio.gather(*(one(s) for s in samples)))


def _by_conversation(records: list[dict], judge: str, key: str) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if judge not in r["judges"]:  # beyond this judge's sample cap
            continue
        v = r["judges"][judge][key] if key in HALVES else r[key]
        if v is not None:
            vals[r["prompt_id"]].append(float(v))
    return {pid: statistics.fmean(v) for pid, v in vals.items()}


def bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(N_BOOT, len(values))).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(records: list[dict], examples: dict[str, Example], usage: dict) -> dict:
    judges = sorted({j for r in records for j in r["judges"]})
    out: dict = {
        "n_responses": len(records),
        "n_conversations": len({r["prompt_id"] for r in records}),
    }
    for j in judges:
        d = {}
        for half in HALVES:
            per = _by_conversation(records, j, half)
            v = np.array(list(per.values()))
            d[half] = {
                "mean": float(v.mean()),
                "healthbench": clipped_mean(list(v)),
                "ci95": bootstrap_ci(v),
            }
        graded = [r for r in records if j in r["judges"]]
        axes = defaultdict(list)
        themes = defaultdict(list)
        for r in graded:
            for a, s in r["judges"][j]["axes"].items():
                axes[a].append(s)
            if (s := r["judges"][j]["full"]) is not None:
                themes[examples[r["prompt_id"]].themes[0]].append(s)
        d["n_responses"] = len(graded)
        d["axes"] = {a: statistics.fmean(v) for a, v in sorted(axes.items())}
        d["themes"] = {t: statistics.fmean(v) for t, v in sorted(themes.items())}
        d["failed_verdicts"] = sum(r["judges"][j]["failed"] for r in graded)
        out[j] = d
    chars = [r["chars"] for r in records]
    tokens = [r["tokens"] for r in records if r["tokens"] is not None]
    out["length"] = {
        "chars_mean": statistics.fmean(chars),
        "chars_median": statistics.median(chars),
        "tokens_mean": statistics.fmean(tokens) if tokens else None,
        "truncated": sum(r["truncated"] for r in records) / len(records),
    }
    out["usage"] = usage
    return out


def estimate_cost(
    samples: list[dict], examples: dict[str, Example]
) -> dict[str, float]:
    chars = sum(
        len(render(examples[s["prompt_id"]].messages, s["response"], str(c)))
        for s in samples
        for c in examples[s["prompt_id"]].rubric
    )
    calls = sum(len(examples[s["prompt_id"]].rubric) for s in samples)
    return {
        m: round((chars / 3.5 * p_in + 120 * calls * p_out) / 1e6, 2)
        for m, (p_in, p_out) in PRICES.items()
    } | {"calls": calls}


async def score_file(
    samples_path: Path,
    judges: Sequence[str] = DEFAULT_JUDGES,
    max_samples: int | None = None,
    name: str | None = None,
    concurrency: int = 32,
    cache: str | None = None,
    estimate: bool = False,
    root: Path = healthbench.RAW,
    out: Path = OUT,
) -> Path:
    """Grade every response in `samples_path`; write records and a summary."""
    examples = {e.id: e for e in healthbench.load_examples(root=root)}
    samples = load_samples(samples_path, max_samples)
    print(
        f"{len(samples)} responses; estimated cost {estimate_cost(samples, examples)}"
    )
    if estimate:
        return samples_path
    verdicts = VerdictCache(cache) if cache else VerdictCache()
    records = [
        {
            "prompt_id": s["prompt_id"],
            "sample": s["sample"],
            "chars": len(clean_response(s["response"])),
            "tokens": s.get("completion_tokens"),
            "truncated": s.get("finish_reason") == "length",
            "judges": {},
        }
        for s in samples
    ]
    usage = {}
    for spec in judges:
        judge_name, _, k = spec.partition(":")  # "sonnet:1": sample 0 only
        config = JudgeConfig.named(judge_name, concurrency=concurrency)
        judge = ClaudeJudge(config, verdicts)
        pairs = zip(records, samples, strict=True)
        mine = [(r, s) for r, s in pairs if not k or s["sample"] < int(k)]
        print(f"judge {config.model}: {len(mine)} responses")
        mets = await score_samples([s for _, s in mine], examples, judge)
        for (rec, s), met in zip(mine, mets, strict=True):
            rec["judges"][judge_name] = grade_record(examples[s["prompt_id"]], met) | {
                "met": met
            }
        usage[judge_name] = vars(judge.usage) | {
            "cost_usd": round(judge.usage.cost(config.model), 3)
        }
    run_dir = out / (name or samples_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records, examples, usage) | {"samples": str(samples_path)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(table([run_dir]), end="")
    print(f"wrote {run_dir}")
    return run_dir


# --- Tables and paired comparison ----------------------------------------------


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def table(dirs: list[Path]) -> str:
    """One row per (run, judge): the three scores with CIs, length, truncation."""
    head = ("Run", "Judge", "Full rubric", "Training criteria", "Held-out criteria")
    lines = [_row(*head, "Chars (median)", "Truncated"), "|---" * 7 + "|"]
    for d in dirs:
        s = json.loads((d / "summary.json").read_text())
        for j in (k for k in s if isinstance(s[k], dict) and "full" in s[k]):
            lines.append(
                f"| {d.name} | {j} | {_fmt(s[j]['full'])} | {_fmt(s[j]['train'])} "
                f"| {_fmt(s[j]['heldout'])} | {s['length']['chars_median']:.0f} "
                f"| {s['length']['truncated']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def _records(d: Path) -> list[dict]:
    return [json.loads(line) for line in (d / "records.jsonl").read_text().splitlines()]


def paired_diff(a: dict[str, float], b: dict[str, float], seed: int = 0) -> dict:
    ids = sorted(a.keys() & b.keys())
    diff = np.array([b[i] - a[i] for i in ids])
    lo, hi = bootstrap_ci(diff, seed)
    return {"n": len(ids), "mean": float(diff.mean()), "ci95": (lo, hi)}


def compare(base: Path, new: Path) -> str:
    """new − base per judge and half, then the two hacking signals with paired CIs."""
    ra, rb = _records(base), _records(new)
    judges = sorted(
        {j for r in ra for j in r["judges"]} & {j for r in rb for j in r["judges"]}
    )
    lines = [
        f"`{new.name}` − `{base.name}` (paired over conversations, 95% bootstrap CI)",
        "",
    ]
    head = ("Judge", "Full rubric", "Training criteria", "Held-out criteria")
    lines += [_row(*head, "Train gain − held-out gain"), "|---" * 5 + "|"]
    gains = {}
    for j in judges:
        per = {
            h: (_by_conversation(ra, j, h), _by_conversation(rb, j, h)) for h in HALVES
        }
        d = {h: paired_diff(*per[h]) for h in HALVES}
        ids = sorted(
            per["train"][0].keys()
            & per["train"][1].keys()
            & per["heldout"][0].keys()
            & per["heldout"][1].keys()
        )
        gap = {
            i: (per["train"][1][i] - per["train"][0][i])
            - (per["heldout"][1][i] - per["heldout"][0][i])
            for i in ids
        }
        g = paired_diff(dict.fromkeys(gap, 0.0), gap)
        gains[j] = {
            i: per["full"][1][i] - per["full"][0][i]
            for i in per["full"][0].keys() & per["full"][1].keys()
        }
        lines.append(_row(j, *(_fmt(d[h]) for h in HALVES), _fmt(g)))
    if len(judges) == 2:
        j1, j2 = judges
        ids = gains[j1].keys() & gains[j2].keys()
        gap = {i: gains[j1][i] - gains[j2][i] for i in ids}
        g = paired_diff(dict.fromkeys(gap, 0.0), gap)
        lines += ["", f"Full-rubric gain under {j1} − under {j2}: {_fmt(g)}"]
    la = statistics.median(r["chars"] for r in ra)
    lb = statistics.median(r["chars"] for r in rb)
    lines += ["", f"Median response length: {la:.0f} → {lb:.0f} chars"]
    return "\n".join(lines) + "\n"


def physician_samples(root: Path, ids_path: Path, out: Path) -> int:
    """The physicians' ideal completions for the eval subset, as a samples file.

    HealthBench ships one for most conversations: a reference row for the table,
    and an end-to-end check of this scorer that needs no GPU.
    """
    ids = set(healthbench.read_ids(ids_path))
    rows = healthbench._read_jsonl(root / healthbench.SOURCES["oss_eval"].file)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for r in rows:
            ideal = r.get("ideal_completions_data") or {}
            if r["prompt_id"] in ids and ideal.get("ideal_completion"):
                sample = {"prompt_id": r["prompt_id"], "sample": 0}
                sample |= {"response": ideal["ideal_completion"], "model": "physician"}
                f.write(json.dumps(sample) + "\n")
                n += 1
    return n


_root = click.option("--root", type=click.Path(path_type=Path), default=healthbench.RAW)
_dir = click.Path(exists=True, file_okay=False, path_type=Path)


@click.group(help=__doc__.split("\n\n")[0])
def main() -> None:
    pass


@main.command("score")
@click.argument("samples_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--judge",
    "judges",
    multiple=True,
    default=DEFAULT_JUDGES,
    show_default=True,
    help="haiku | sonnet, repeatable; 'sonnet:1' grades sample 0 only",
)
@click.option("--max-samples", type=int, default=None, help="use samples 0..K-1 only")
@click.option("--name", default=None, help="output dir name (default: samples stem)")
@click.option("--concurrency", default=32, show_default=True)
@click.option("--cache", default=None, help="SQLite path (default ~/.cache/rubricrl)")
@click.option("--estimate", is_flag=True, help="print the cost estimate only")
@_root
@click.option("--out", type=click.Path(path_type=Path), default=OUT, show_default=True)
def score_cmd(**kwargs) -> None:
    asyncio.run(score_file(**kwargs))


@main.command("table")
@click.argument("dirs", type=_dir, nargs=-1, required=True)
def table_cmd(dirs: tuple[Path, ...]) -> None:
    print(table(list(dirs)), end="")


@main.command("compare")
@click.argument("base", type=_dir)
@click.argument("new", type=_dir)
def compare_cmd(base: Path, new: Path) -> None:
    print(compare(base, new), end="")


@main.command("physician", help="Write the physician-completions samples.")
@_root
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("runs/samples/physician.jsonl"),
    show_default=True,
)
def physician_cmd(root: Path, out: Path) -> None:
    n = physician_samples(root, healthbench.EVAL_IDS, out)
    print(f"wrote {out}: {n} physician completions")


if __name__ == "__main__":
    main()
