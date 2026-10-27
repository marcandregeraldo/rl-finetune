"""Judge calibration on HealthBench's oss_meta_eval: how often the judge agrees with
physicians, next to how often physicians agree with each other.

Metric (simple-evals' `healthbench_meta_eval`): every (judge label, physician
label) pair on the same judgment counts once. Precision of a class is the share of
pairs that agree among those where the judge said that class; recall, among those
where the physician did. F1 per class, then **balanced F1 = (F1_met + F1_not_met) / 2**,
overall and per rubric category. The physician ceiling is the same metric for
each physician against the other physicians on the same items, averaged over
physicians, on all 29.5K judgments (free: no API calls).

Also: test-retest (the same items graded twice; the API has no temperature, so
this is the judge's own noise) and agreement by completion length.

    python -m rubricrl.analysis.calibrate estimate --n 2000
    python -m rubricrl.analysis.calibrate run --judge haiku --n 2000
    python -m rubricrl.analysis.calibrate run --judge sonnet --n 2000
    python -m rubricrl.analysis.calibrate report      # Markdown for the README

The meta-eval rubrics are cluster-level ("consensus") criteria: 34 categories,
30 distinct texts, each one in its conversation's oss_eval rubric. The training
reward is mostly example-level criteria (49,184 of 57,237). The number calibrates
the judge on the HealthBench grading task, not on every criterion it will see.
"""

import asyncio
import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import click

from rubricrl.data import healthbench
from rubricrl.data.schema import Judgment
from rubricrl.judge.client import (
    PRICES,
    ClaudeJudge,
    JudgeConfig,
    Verdict,
    VerdictCache,
)
from rubricrl.judge.prompt import render

OUT = Path("analysis/out/calibration")


# --- Metrics -------------------------------------------------------------------


def pairwise_f1(pairs: Iterable[tuple[bool, bool]]) -> dict[str, float | int | None]:
    """(rater, reference) pairs -> per-class precision, recall, F1; balanced F1."""
    pairs = list(pairs)
    out: dict[str, float | int | None] = {"n_pairs": len(pairs)}
    f1s = []
    for cls, name in ((True, "pos"), (False, "neg")):
        said = [r == ref for r, ref in pairs if r == cls]
        truth = [r == ref for r, ref in pairs if ref == cls]
        p = sum(said) / len(said) if said else None
        r = sum(truth) / len(truth) if truth else None
        if p is None or r is None:
            f1 = None
        else:
            f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        out |= {f"precision_{name}": p, f"recall_{name}": r, f"f1_{name}": f1}
        f1s.append(f1)
    out["f1_balanced"] = None if None in f1s else sum(f1s) / 2
    return out


def judge_metrics(
    judgments: Sequence[Judgment], labels: Sequence[bool | None]
) -> dict[str, dict]:
    """Judge vs every physician label: overall and per category. None labels skipped."""
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for j, label in zip(judgments, labels, strict=True):
        if label is None:
            continue
        for ref in j.physician_labels:
            groups["overall"].append((label, ref))
            groups[j.category].append((label, ref))
    return {g: pairwise_f1(p) for g, p in sorted(groups.items())}


def physician_metrics(
    judgments: Sequence[Judgment], min_pairs: int = 1
) -> dict[str, dict]:
    """Each physician vs the other physicians; the mean and median over physicians."""
    per: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        for i, (pid, label) in enumerate(
            zip(j.physician_ids, j.physician_labels, strict=True)
        ):
            others = j.physician_labels[:i] + j.physician_labels[i + 1 :]
            for ref in others:
                per[pid]["overall"].append((label, ref))
                per[pid][j.category].append((label, ref))
    groups: dict[str, list[float]] = defaultdict(list)
    for by_group in per.values():
        for g, pairs in by_group.items():
            if len(pairs) >= min_pairs:
                f1 = pairwise_f1(pairs)["f1_balanced"]
                if f1 is not None:
                    groups[g].append(f1)
    return {
        g: {
            "physicians": len(v),
            "f1_balanced_mean": statistics.fmean(v),
            "f1_balanced_median": statistics.median(v),
        }
        for g, v in sorted(groups.items())
    }


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def by_length(
    judgments: Sequence[Judgment], labels: Sequence[bool | None], buckets: int = 4
) -> list[dict]:
    """Balanced F1 per completion-length quantile bucket."""
    order = sorted(range(len(judgments)), key=lambda i: len(judgments[i].completion))
    rows = []
    for b in range(buckets):
        idx = order[b * len(order) // buckets : (b + 1) * len(order) // buckets]
        js = [judgments[i] for i in idx]
        m = judge_metrics(js, [labels[i] for i in idx]).get("overall", {})
        lens = [len(j.completion) for j in js]
        rows.append(
            {
                "bucket": b,
                "chars_min": min(lens),
                "chars_max": max(lens),
                "n": len(js),
                "f1_balanced": m.get("f1_balanced"),
            }
        )
    return rows


# --- Sampling and grading ------------------------------------------------------


def sample_judgments(
    judgments: list[Judgment], n: int, seed: int = 0
) -> list[Judgment]:
    """`n` judgments stratified by category (largest remainder), deterministic."""
    by_cat: dict[str, list[Judgment]] = defaultdict(list)
    for j in sorted(judgments, key=lambda j: healthbench._unit("meta", seed, j.id)):
        by_cat[j.category].append(j)
    quotas = {c: n * len(v) / len(judgments) for c, v in by_cat.items()}
    counts = {c: int(q) for c, q in quotas.items()}
    for c in sorted(quotas, key=lambda c: (counts[c] - quotas[c], c))[
        : n - sum(counts.values())
    ]:
        counts[c] += 1
    return [j for c, v in sorted(by_cat.items()) for j in v[: counts[c]]]


async def grade_all(
    judge, judgments: Sequence[Judgment], repeat: int = 0
) -> list[Verdict]:
    done, t0 = 0, time.monotonic()

    async def one(j: Judgment) -> Verdict:
        nonlocal done
        v = await judge.grade(j.messages, j.completion, j.rubric, repeat)
        done += 1
        if done % 200 == 0 or done == len(judgments):
            t = time.monotonic() - t0
            print(f"  {done}/{len(judgments)} graded, {t:.0f}s", flush=True)
        return v

    return list(await asyncio.gather(*(one(j) for j in judgments)))


def estimate(judgments: Sequence[Judgment]) -> dict[str, float]:
    """Rough $ per judge: ~3.5 chars per input token, ~120 output tokens per verdict."""
    chars = sum(len(render(j.messages, j.completion, j.rubric)) for j in judgments)
    tokens_in, tokens_out = chars / 3.5, 120 * len(judgments)
    return {
        model: round((tokens_in * p_in + tokens_out * p_out) / 1e6, 2)
        for model, (p_in, p_out) in PRICES.items()
    }


def label(config: JudgeConfig) -> str:
    return f"{config.model}-{config.effort}" + ("-thinking" if config.thinking else "")


async def run(
    judge_name: str = "haiku",
    n: int = 2000,
    seed: int = 0,
    effort: str = "low",
    thinking: bool = False,
    concurrency: int = 32,
    retest: int = 200,
    cache: str | None = None,
    root: Path = healthbench.RAW,
    out: Path = OUT,
) -> dict:
    all_judgments = healthbench.load_judgments(root)
    subset = sample_judgments(all_judgments, n, seed)
    config = JudgeConfig.named(
        judge_name, effort=effort, thinking=thinking, concurrency=concurrency
    )
    judge = ClaudeJudge(config, VerdictCache(cache) if cache else VerdictCache())
    print(
        f"{config.model}: grading {len(subset)} judgments (estimate {estimate(subset)})"
    )
    verdicts = await grade_all(judge, subset)
    labels = [v.met for v in verdicts]
    retest_items = subset[:retest]
    print(f"{config.model}: regrading {len(retest_items)} for test-retest")
    retest = [v.met for v in await grade_all(judge, retest_items, repeat=1)]
    both = [
        (a, b)
        for a, b in zip(labels, retest, strict=False)
        if a is not None and b is not None
    ]
    result = {
        "model": config.model,
        "label": label(config),
        "effort": config.effort,
        "thinking": config.thinking,
        "n": len(subset),
        "seed": seed,
        "failed": sum(x is None for x in labels),
        "errors": dict(Counter(v.error for v in verdicts if v.met is None)),
        "judge": judge_metrics(subset, labels),
        "physicians_on_subset": physician_metrics(subset),
        "physicians_all": physician_metrics(all_judgments),
        "retest": {
            "n": len(both),
            "agreement": sum(a == b for a, b in both) / len(both) if both else None,
            "kappa": cohen_kappa([a for a, _ in both], [b for _, b in both]),
        },
        "by_length": by_length(subset, labels),
        "judge_rate_met": sum(bool(x) for x in labels)
        / max(1, sum(x is not None for x in labels)),
        "physician_rate_met": statistics.fmean(
            x for j in subset for x in j.physician_labels
        ),
        "usage": vars(judge.usage)
        | {"cost_usd": round(judge.usage.cost(config.model), 3)},
        "labels": {j.id: x for j, x in zip(subset, labels, strict=True)},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label(config)}-n{len(subset)}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}; cost ${result['usage']['cost_usd']}")
    return result


# --- Report --------------------------------------------------------------------


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _f(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def report(out: Path = OUT, n: int | None = None) -> str:
    """All results in `out` (only those on `n` judgments, if given)."""
    results = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
    results = [r for r in results if n is None or r["n"] == n]
    if not results:
        return f"no results in {out}\n"
    phys = results[0]["physicians_all"]
    head = ("Rater", "Balanced F1", "F1 met", "F1 not met", "Rate 'met'")
    lines = [
        _row(*head, "Test-retest agreement (κ)", "Failed", "Cost (new calls)"),
        "|---" * 8 + "|",
    ]
    for r in results:
        o, rt = r["judge"]["overall"], r["retest"]
        lines.append(
            _row(
                f"{r['label']} (n={r['n']})",
                *(_f(o[k]) for k in ("f1_balanced", "f1_pos", "f1_neg")),
                f"{r['judge_rate_met']:.2f}",
                f"{_f(rt['agreement'])} ({_f(rt['kappa'])})",
                f"{r['failed']}/{r['n']}",
                f"${r['usage']['cost_usd']}",
            )
        )
    p = phys["overall"]
    who = f"Physicians vs each other (mean over {p['physicians']}, all 29.5K)"
    rate = f"{results[0]['physician_rate_met']:.2f}"
    lines.append(_row(who, _f(p["f1_balanced_mean"]), "", "", rate, "", "", ""))
    cats = sorted(k for k in results[0]["judge"] if k != "overall")
    names = [f"{r['label']} n={r['n']}" for r in results]
    head = "| Category | " + " | ".join(names) + " | Physicians |"
    lines += ["", head, "|---|" + "---|" * (len(results) + 1)]
    for c in cats:
        vals = [_f(r["judge"].get(c, {}).get("f1_balanced")) for r in results]
        pc = phys.get(c, {}).get("f1_balanced_mean")
        lines.append(
            f"| {c.removeprefix('cluster:')} | " + " | ".join(vals) + f" | {_f(pc)} |"
        )
    lines += ["", "| Length bucket (chars) | " + " | ".join(names) + " |"]
    lines.append("|---|" + "---|" * len(results))
    for b in range(len(results[0]["by_length"])):
        row = results[0]["by_length"][b]
        vals = [_f(r["by_length"][b]["f1_balanced"]) for r in results]
        lines.append(
            f"| {row['chars_min']}–{row['chars_max']} | " + " | ".join(vals) + " |"
        )
    return "\n".join(lines) + "\n"


_n = click.option("--n", default=2000, show_default=True, help="judgments")
_seed = click.option("--seed", default=0, show_default=True)
_root = click.option("--root", type=click.Path(path_type=Path), default=healthbench.RAW)
_out = click.option(
    "--out", type=click.Path(path_type=Path), default=OUT, show_default=True
)


@click.group(help=__doc__.split("\n\n")[0])
def main() -> None:
    pass


@main.command("estimate")
@_n
@_seed
@_root
def estimate_cmd(n: int, seed: int, root: Path) -> None:
    subset = sample_judgments(healthbench.load_judgments(root), n, seed)
    print(json.dumps({"n": len(subset), "usd": estimate(subset)}))


@main.command("run")
@_n
@_seed
@_root
@_out
@click.option("--judge", "judge_name", default="haiku", show_default=True)
@click.option("--effort", default="low", show_default=True)
@click.option("--thinking", is_flag=True, help="adaptive thinking on")
@click.option("--concurrency", default=32, show_default=True)
@click.option("--retest", default=200, show_default=True, help="items graded twice")
@click.option("--cache", default=None, help="SQLite path (default ~/.cache/rubricrl)")
def run_cmd(**kwargs) -> None:
    asyncio.run(run(**kwargs))


@main.command("report")
@_out
@click.option("--n", type=int, default=None, help="only results on n judgments")
def report_cmd(out: Path, n: int | None) -> None:
    print(report(out, n), end="")


if __name__ == "__main__":
    main()
