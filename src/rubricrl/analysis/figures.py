"""The report's figures (`docs/report.md`), as SVGs in `docs/figures/`.

Every plotted number is read when the command runs: from the calibration results
(`analysis/out/calibration/`), the GEPA result (`analysis/out/optimize/<name>/`),
the eval summaries (`runs/evals/*/summary.json`) and the training runs' W&B
history. Rerun after new results land; nothing here is copied from the report.

    python -m rubricrl.analysis.figures all                  # every figure
    python -m rubricrl.analysis.figures calibration          # §4: balanced F1 per judge
    python -m rubricrl.analysis.figures length-bias          # §4.1: "met" by length
    python -m rubricrl.analysis.figures gepa                 # §4.1: GEPA's candidates
    python -m rubricrl.analysis.figures self-hosted          # §4.2: open-model judges
    python -m rubricrl.analysis.figures baselines            # §7: eval scores, 95% CIs
    python -m rubricrl.analysis.figures two-judges           # §7.1: gains, 2 judges
    python -m rubricrl.analysis.figures training --run aqcpypgj --run <id> ...  # §6

Needs the `observe` extra (matplotlib; wandb for `training`, which only reads):
`uv run --extra observe --extra figures python -m rubricrl.analysis.figures all`.

The SVGs have a transparent background and one set of colors that clears
contrast on both of GitHub's Markdown themes (white and #0d1117): mid-gray ink,
and the reference palette's dark-mode steps, which pass the light-mode checks
too. Color follows the judge prompt across figures: orange is simple-evals'
prompt, blue `gepa-v1` (the reward's), gray the physicians.

In the W&B history, metrics from different log calls are separate rows, each
carrying its own step key (`rollout/step`, `train/step`, `eval/step`); a metric is
plotted against the step key on its row, never `_step`. Two eval rounds can share
an `eval/step` value: both points are kept.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import click

from rubricrl.analysis import calibrate
from rubricrl.data import healthbench
from rubricrl.evals import run as evals
from rubricrl.judge.client import JudgeConfig
from rubricrl.judge.prompt import (
    PROMPT_VERSION,
    REWARD_PROMPT_VERSION,
    TRAJECTORY_VERSION,
)

OUT = Path("docs/figures")
OPTIMIZE = Path("analysis/out/optimize")
WANDB_PROJECT = "marcandregeraldo-none/rl-finetune"
# The training runs `training` / `all` plot by default, with the report's names (§6).
RUN_NAMES = {
    "aqcpypgj": "A: lr 1e-6",
    "vd00i7jk": "B: lr 3e-6",
    "s5yq2faq": "C: lr 3e-6 + dynamic sampling",
    "vmgc5lf8": "D: C + no std-normalization",
    "sjyp2hav": "E: D + gpt-oss-120b judge (stopped at 50)",
}
RUNS = tuple(RUN_NAMES)

# The calibration labels (`calibrate.label`) of the reward's judge and of the
# judge it replaced: Haiku, effort low, with `gepa-v1` / simple-evals' prompt.
REWARD = calibrate.label(JudgeConfig(), prompt_version=REWARD_PROMPT_VERSION)
OLD = calibrate.label(JudgeConfig())

# --- Style ---------------------------------------------------------------------

INK = "#7d7d7d"  # text and ticks: 4.1:1 on white, 4.6:1 on #0d1117
REF = "#8c8c8c"  # reference marks (physicians, other judges)
BLUE, ORANGE, AQUA = "#2a78d6", "#d95926", "#199e70"
# Categorical order for overlaid runs: the reference palette's dark-mode steps,
# checked for contrast on both GitHub surfaces.
SERIES = (BLUE, ORANGE, AQUA, "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
PROMPT_COLORS = {PROMPT_VERSION: ORANGE, REWARD_PROMPT_VERSION: BLUE}


def _plt():
    """matplotlib, imported on use (the `observe` extra), with the report style."""
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 7,
            "legend.frameon": False,
            "svg.fonttype": "path",  # text as outlines: renders the same everywhere
            "svg.hashsalt": "rubricrl",  # stable ids: reruns diff cleanly
        }
    )
    return plt


def _save(fig, out: Path, name: str, title: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.svg"
    fig.savefig(
        path,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Title": title, "Date": None},
    )
    fig.clf()
    print(f"wrote {path}")
    return path


def _pct(ax) -> None:
    """Shares as percentages, no trailing zeros (2.5%, 10%)."""
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.4g}%"))


# --- Names ---------------------------------------------------------------------


# Self-hosted judges (`modal/miles_rft.py` JUDGE_MODELS): their model cards' names.
OPEN_MODELS = {
    "gpt-oss-120b": "gpt-oss-120b",
    "gpt-oss-20b": "gpt-oss-20b",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
    "qwen3.5-4b": "Qwen3.5-4B",
}


def _self_hosted(r: dict) -> bool:
    return r["model"] in OPEN_MODELS


def _model(model: str) -> str:
    """'claude-haiku-5-5' -> 'Haiku 5.5'; 'haiku' -> 'Haiku'; open models as
    their model cards name them."""
    if model in OPEN_MODELS:
        return OPEN_MODELS[model]
    parts = model.removeprefix("claude-").split("-")
    return " ".join([parts[0].title(), ".".join(parts[1:])]).strip()


def _prompt(version: str | None) -> str:
    return {
        None: "simple-evals'",
        PROMPT_VERSION: "simple-evals'",
        TRAJECTORY_VERSION: "trajectory",
    }.get(version, version)


def _version(r: dict) -> str:
    """A calibration result's prompt version (older results leave it out)."""
    if r.get("prompt_version"):
        return r["prompt_version"]
    return TRAJECTORY_VERSION if r.get("grader") == "trajectory" else PROMPT_VERSION


def judge_name(r: dict) -> str:
    """A calibration result -> 'Haiku 5.5, low + thinking, gepa-v1'; open models:
    gpt-oss's reasoning effort, nothing for Qwen (thinking off)."""
    if _self_hosted(r):
        effort = (
            f"reasoning {r['effort']}, " if r["model"].startswith("gpt-oss") else ""
        )
        return f"{_model(r['model'])}, {effort}{_prompt(_version(r))}"
    effort = r["effort"] + (" + thinking" if r["thinking"] else "")
    return f"{_model(r['model'])}, {effort}, {_prompt(_version(r))}"


# --- Judge calibration (§4) ------------------------------------------------------


def load_calibration(src: Path, n: int) -> list[dict]:
    results = [json.loads(p.read_text()) for p in sorted(src.glob("*.json"))]
    results = [r for r in results if r["n"] == n]
    if not results:
        raise click.ClickException(f"no calibration results on n={n} in {src}")
    return results


def _calibration_color(r: dict) -> str:
    return {REWARD: BLUE, OLD: ORANGE}.get(r["label"], REF)


def calibration(src: Path = calibrate.OUT, out: Path = OUT, n: int = 1000) -> Path:
    """Balanced F1 of each judge configuration on the same `n` judgments, with
    the physicians' agreement with each other as the ceiling. Claude judges only
    (the self-hosted ones: `self_hosted`)."""
    plt = _plt()
    results = [r for r in load_calibration(src, n) if not _self_hosted(r)]
    results.sort(key=lambda r: r["judge"]["overall"]["f1_balanced"])
    f1 = [r["judge"]["overall"]["f1_balanced"] for r in results]
    ceiling = results[0]["physicians_all"]["overall"]["f1_balanced_mean"]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(results) + 1.0))
    y = range(len(results))
    ax.hlines(y, min(f1 + [ceiling]) - 0.03, f1, color=INK, alpha=0.25, lw=0.6)
    ax.scatter(f1, y, s=56, color=[_calibration_color(r) for r in results], zorder=3)
    for yi, x, r in zip(y, f1, results, strict=True):
        weight = "bold" if r["label"] in (REWARD, OLD) else "normal"
        ax.annotate(f"{x:.3f}", (x, yi), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, weight=weight)  # fmt: skip
    ax.axvline(ceiling, color=REF, lw=1)
    ax.annotate(
        f"physicians vs each other {ceiling:.3f}",
        (ceiling, len(results) - 0.4),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set_yticks(list(y), [judge_name(r) for r in results])
    for tick, r in zip(ax.get_yticklabels(), results, strict=True):
        if r["label"] in (REWARD, OLD):
            tick.set_fontweight("bold")
    ax.set_xlim(min(f1 + [ceiling]) - 0.03, max(f1 + [ceiling]) + 0.03)
    ax.set_ylim(-0.6, len(results) + 0.3)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"balanced F1 against physicians ({n:,} meta-eval judgments)")
    ax.set_title("Judge calibration: agreement with physicians")
    return _save(fig, out, "judge-calibration", "Judge calibration")


# --- Length bias (§4.1) ----------------------------------------------------------


def length_rows(results: Sequence[dict], root: Path) -> list[list[dict]]:
    """Each result's `by_length` with the rates of "met"; results written before
    `by_length` had them are recomputed from their labels (`calibrate.lengths`,
    local data only)."""
    if all("judge_rate_met" in b for r in results for b in r["by_length"]):
        return [r["by_length"] for r in results]
    return [row["buckets"] for row in calibrate.lengths(results, root)]


def length_bias(
    src: Path = calibrate.OUT,
    out: Path = OUT,
    n: int = 1000,
    root: Path = healthbench.RAW,
) -> Path:
    """Share of "met" by completion-length quartile: physicians, the old judge,
    and the reward's judge."""
    plt = _plt()
    by_label = {r["label"]: r for r in load_calibration(src, n)}
    missing = [x for x in (OLD, REWARD) if x not in by_label]
    if missing:
        raise click.ClickException(f"no calibration result for {missing} (n={n})")
    old, new = length_rows([by_label[OLD], by_label[REWARD]], root)
    x = range(len(old))
    lines = [
        ("physicians", [b["physician_rate_met"] for b in old], REF),
        (f"{_prompt(None)} prompt (old judge)",
         [b["judge_rate_met"] for b in old], ORANGE),
        (f"{REWARD_PROMPT_VERSION} (the reward)",
         [b["judge_rate_met"] for b in new], BLUE),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for name, ys, color in lines:
        ax.plot(x, ys, color=color, marker="o", ms=7)
        ax.annotate(f"{name}  {ys[-1]:.0%}", (x[-1], ys[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8.5)  # fmt: skip
        ax.annotate(f"{ys[0]:.0%}", (x[0], ys[0]), xytext=(-9, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=8)  # fmt: skip
    ax.set_xticks(
        list(x),
        [f"Q{b['bucket'] + 1}\n{b['chars_min']:,}–{b['chars_max']:,} chars"
         for b in old],
    )  # fmt: skip
    ax.set_xlim(-0.35, len(old) - 0.6)
    ax.set_ylim(0, 1)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel('says "met"')
    judge = f"{_model(by_label[REWARD]['model'])}, effort {by_label[REWARD]['effort']}"
    ax.set_title(f'Share of "met" by completion length ({judge}, n={n:,})')
    return _save(fig, out, "length-bias", "Length bias")


# --- Self-hosted judges (§4.2) ----------------------------------------------------


def self_hosted(
    src: Path = calibrate.OUT,
    out: Path = OUT,
    n: int = 1000,
    root: Path = healthbench.RAW,
) -> list[Path]:
    """Open-model judges against the Claude reward's: balanced F1 per model and
    prompt (orange simple-evals', blue `gepa-v1`, Haiku's rows in bold), and
    the share of "met" by length quartile with `gepa-v1`."""
    plt = _plt()
    results = load_calibration(src, n)
    by_label = {r["label"]: r for r in results}
    missing = [x for x in (OLD, REWARD) if x not in by_label]
    if missing:
        raise click.ClickException(f"no calibration result for {missing} (n={n})")
    rows = [r for r in results if _self_hosted(r)] + [by_label[OLD], by_label[REWARD]]
    if len(rows) == 2:
        raise click.ClickException(f"no self-hosted calibration results (n={n})")
    rows.sort(key=lambda r: r["judge"]["overall"]["f1_balanced"])
    f1 = [r["judge"]["overall"]["f1_balanced"] for r in rows]
    ceiling = rows[0]["physicians_all"]["overall"]["f1_balanced_mean"]
    lo, hi = min(f1 + [ceiling]) - 0.03, max(f1 + [ceiling]) + 0.03
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(rows) + 1.0))
    y = range(len(rows))
    colors = [PROMPT_COLORS.get(_version(r), REF) for r in rows]
    ax.hlines(y, lo, f1, color=INK, alpha=0.25, lw=0.6)
    ax.scatter(f1, y, s=56, color=colors, zorder=3)
    for yi, x, r in zip(y, f1, rows, strict=True):
        weight = "normal" if _self_hosted(r) else "bold"
        ax.annotate(f"{x:.3f}", (x, yi), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, weight=weight)  # fmt: skip
    ax.axvline(ceiling, color=REF, lw=1)
    ax.annotate(f"physicians vs each other {ceiling:.3f}", (ceiling, len(rows) - 0.4),
                xytext=(-4, 0), textcoords="offset points", ha="right",
                va="bottom", fontsize=8)  # fmt: skip
    ax.set_yticks(list(y), [judge_name(r) for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows, strict=True):
        if not _self_hosted(r):
            tick.set_fontweight("bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(rows) + 0.3)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"balanced F1 against physicians ({n:,} meta-eval judgments)")
    ax.set_title("Self-hosted judges vs Haiku (bold): agreement with physicians")
    paths = [_save(fig, out, "judge-calibration-self-hosted", "Self-hosted judges")]

    gepa_rows = sorted(
        (
            r
            for r in results
            if _self_hosted(r) and _version(r) == REWARD_PROMPT_VERSION
        ),
        key=lambda r: -r["judge"]["overall"]["f1_balanced"],
    )
    reward, *opens = length_rows([by_label[REWARD], *gepa_rows], root)
    x = range(len(reward))
    lines = [("physicians", [b["physician_rate_met"] for b in reward], REF),
             ("Haiku 5.5 low (the reward)", [b["judge_rate_met"] for b in reward],
              BLUE)]  # fmt: skip
    lines += [
        (_model(r["model"]), [b["judge_rate_met"] for b in bs], color)
        for r, bs, color in zip(
            gepa_rows, opens, (AQUA, SERIES[4], SERIES[6], SERIES[3]), strict=False
        )
    ]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for _, ys, color in lines:
        ax.plot(x, ys, color=color, marker="o", ms=6)
    ends = sorted(lines, key=lambda line: line[1][-1])
    label_y = [ends[0][1][-1]]
    for _, ys, _ in ends[1:]:  # spread the end labels 5 points apart
        label_y.append(max(ys[-1], label_y[-1] + 0.05))
    for (name, ys, color), ly in zip(ends, label_y, strict=True):
        ax.annotate(f"{name}  {ys[-1]:.0%}", (x[-1], ys[-1]), xytext=(x[-1] + 0.12, ly),
                    textcoords="data", va="center", fontsize=8,
                    color=color)  # fmt: skip
    ax.set_xticks(
        list(x),
        [f"Q{b['bucket'] + 1}\n{b['chars_min']:,}–{b['chars_max']:,} chars"
         for b in reward],
    )  # fmt: skip
    ax.set_xlim(-0.35, len(reward) - 0.4)
    ax.set_ylim(0, 1)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel('says "met"')
    ax.set_title(
        f'Share of "met" by completion length, {REWARD_PROMPT_VERSION} (n={n:,})'
    )
    paths.append(_save(fig, out, "length-bias-self-hosted", "Length bias, self-hosted"))
    return paths


# --- GEPA search (§4.1) ----------------------------------------------------------


def gepa(src: Path = OPTIMIZE / REWARD_PROMPT_VERSION, out: Path = OUT) -> Path:
    """Validation score of each candidate prompt in the order it joined GEPA's
    pool (0 is the seed, simple-evals' prompt); the selected one highlighted."""
    plt = _plt()
    result = json.loads((src / "result.json").read_text())
    cands = sorted(result["candidates"], key=lambda c: c["index"])
    idx = [c["index"] for c in cands]
    val = [c["val_score"] for c in cands]
    best, name = result["best_index"], result["config"]["name"]
    running = [max(val[: i + 1]) for i in range(len(val))]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.step(idx, running, where="post", color=REF, lw=1, alpha=0.7)
    colors = [BLUE if i == best else ORANGE if i == 0 else REF for i in idx]
    ax.scatter(idx, val, s=56, color=colors, zorder=3)
    b = idx.index(best)
    for i, text, dy in ((0, "seed: simple-evals'", -12), (b, name, 9)):
        ax.annotate(f"{text}  {val[i]:.3f}", (idx[i], val[i]), xytext=(0, dy),
                    textcoords="offset points", ha="center",
                    va="top" if dy < 0 else "bottom", fontsize=8.5,
                    weight="bold")  # fmt: skip
    ax.annotate("best so far", (idx[-1], running[-1]), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8)  # fmt: skip
    ax.set_xticks(idx)
    ax.set_xlim(-0.6, idx[-1] + 0.6)
    lo, hi = min(val), max(val)
    ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("candidate prompt, in the order it joined the pool")
    cfg = result["config"]
    ax.set_ylabel("validation score")
    ax.set_title(
        f"GEPA search ({name}): {len(cands)} candidates, "
        f"{cfg['n_val']} validation judgments"
    )
    return _save(fig, out, "gepa-search", "GEPA search")


# --- Baselines (§7) --------------------------------------------------------------

HALF_TITLES = {
    "full": "Full rubric",
    "train": "Training half",
    "heldout": "Held-out half",
}


def baseline_rows(src: Path) -> list[dict]:
    """One row per (eval run, judge) in `src/*/summary.json`, physicians first."""
    rows = []
    for path in sorted(src.glob("*/summary.json")):
        s = json.loads(path.read_text())
        policy = Path(s.get("samples", path.parent.name)).stem
        for key, scores in s.items():
            if not (isinstance(scores, dict) and "full" in scores):
                continue
            model, version, _ = evals.parse_judge(key)
            rows.append(
                {
                    "policy": policy,
                    "name": {"physician": "Physician ideal answers"}.get(
                        policy, policy
                    ),
                    "n": s["n_conversations"],
                    "judge": f"{_model(model)}, {_prompt(version)} prompt",
                    "color": PROMPT_COLORS.get(version or PROMPT_VERSION, AQUA),
                    "scores": scores,
                }
            )
    rows.sort(key=lambda r: (r["policy"] != "physician", r["policy"], r["judge"]))
    return rows


def baselines(src: Path = evals.OUT, out: Path = OUT) -> Path:
    """Mean score per conversation with its 95% bootstrap CI, on the full rubric
    and on each half, for every eval run and judge."""
    plt = _plt()
    rows = baseline_rows(src)
    if not rows:
        raise click.ClickException(f"no eval summaries in {src}")
    fig, axes = plt.subplots(
        1, len(evals.HALVES), figsize=(7.4, 0.55 * len(rows) + 1.1),
        sharex=True, sharey=True,
    )  # fmt: skip
    y = list(range(len(rows)))[::-1]
    for ax, half in zip(axes, evals.HALVES, strict=True):
        for yi, r in zip(y, rows, strict=True):
            s = r["scores"][half]
            lo, hi = s["ci95"]
            ax.plot([lo, hi], [yi, yi], color=r["color"], lw=2, alpha=0.45)
            ax.scatter([s["mean"]], [yi], s=48, color=r["color"], zorder=3)
            ax.annotate(f"{s['mean']:.2f}", (s["mean"], yi), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5)  # fmt: skip
        ax.axvline(0, color=INK, lw=0.6, alpha=0.6)
        ax.set_title(HALF_TITLES[half])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(
        y, [f"{r['name']} ({r['n']} conv.)\njudge: {r['judge']}" for r in rows]
    )
    axes[0].set_ylim(-0.6, len(rows) - 0.3)
    fig.supxlabel(
        "mean rubric score per conversation, 95% bootstrap CI", fontsize=9, y=-0.04
    )
    return _save(fig, out, "baselines", "Baselines")


# --- Gains under two judges (§7.1) -----------------------------------------------

# The base model's eval dir and the checkpoints, each graded by both judges
# (`evals.run merge`), with the judge that produced its reward.
GAIN_BASE = "qwen3.5-4b-gepa-v1"
GAIN_RUNS = (
    ("rft-C-lr3e-6-dynsam", "C", "haiku"),
    ("rft-D-lr3e-6-dynsam-nostd", "D", "haiku"),
    ("rft-E20-gpt-oss-120b", "E, step 20", "gpt-oss-120b"),
    ("rft-E50-gpt-oss-120b", "E, step 50", "gpt-oss-120b"),
)
GAIN_JUDGES = (("haiku@gepa-v1", BLUE), ("gpt-oss-120b@gepa-v1", AQUA))


def gain_rows(
    src: Path, base: str = GAIN_BASE, runs: Sequence[tuple] = GAIN_RUNS
) -> list[dict]:
    """Per checkpoint and judge: the full-rubric gain over the base model, paired
    over conversations (`evals.run compare`'s numbers)."""
    rb = evals._records(src / base)
    rows = []
    for name, label, trained_on in runs:
        rn = evals._records(src / name)
        for judge, color in GAIN_JUDGES:
            a = evals._by_conversation(rb, judge, "full")
            b = evals._by_conversation(rn, judge, "full")
            if not (a and b):
                raise click.ClickException(f"no {judge} verdicts in {base} or {name}")
            model, _, _ = evals.parse_judge(judge)
            rows.append(
                {
                    "run": label,
                    "key": judge,
                    "judge": _model(model),
                    "own": model == trained_on,
                    "color": color,
                    **evals.paired_diff(a, b),
                }
            )
    return rows


def two_judges(src: Path = evals.OUT, out: Path = OUT) -> Path:
    """Test-set gain over the base model under each judge, 95% paired CIs; the
    judge each checkpoint was trained on is the filled dot."""
    from matplotlib.lines import Line2D

    plt = _plt()
    rows = gain_rows(src)
    runs = list(dict.fromkeys(r["run"] for r in rows))
    fig, ax = plt.subplots(figsize=(7, 0.62 * len(runs) + 1.0))
    offsets = {j: 0.14 - 0.28 * i for i, (j, _) in enumerate(GAIN_JUDGES)}
    for r in rows:
        y = len(runs) - 1 - runs.index(r["run"])
        y += offsets[r["key"]]
        lo, hi = r["ci95"]
        ax.plot([lo, hi], [y, y], color=r["color"], lw=2, alpha=0.45)
        ax.scatter([r["mean"]], [y], s=48, zorder=3, linewidths=1.6,
                   color=r["color"] if r["own"] else "none",
                   edgecolors=r["color"])  # fmt: skip
        ax.annotate(f"{r['mean']:+.3f}", (hi, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5)  # fmt: skip
    ax.axvline(0, color=INK, lw=0.6, alpha=0.6)
    names = {r["run"]: f"{r['run']}\n(reward: {r['judge']})" for r in rows if r["own"]}
    ax.set_yticks(range(len(runs))[::-1], [names[x] for x in runs])
    ax.set_ylim(-0.6, len(runs) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    hollow = Line2D([], [], color=INK, marker="o", lw=0, ms=6, markerfacecolor="none",
                    label="hollow: not the judge it trained on")  # fmt: skip
    handles = [
        Line2D([], [], color=c, marker="o", lw=0, ms=6,
               label=f"graded by {_model(j.split('@')[0])}")
        for j, c in GAIN_JUDGES
    ] + [hollow]  # fmt: skip
    ax.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.45, -0.2))  # fmt: skip
    ax.set_xlabel("full-rubric gain over Qwen3.5-4B, test set, 95% paired CI")
    return _save(fig, out, "two-judges", "Gains under two judges")


# --- Training (§6) ---------------------------------------------------------------

# (title, [(W&B key, step key, style name)], y format)
PANELS = (
    ("Reward", [("rollout/raw_reward", "rollout/step", "training half (rollouts)"),
                ("eval/healthbench_val", "eval/step", "held-out half (eval)")], None),
    ("Mean reply length (tokens)",
     [("rollout/response_len/mean", "rollout/step", None)], None),
    ("Truncated replies", [("rollout/truncated_ratio", "rollout/step", None)], "%"),
    ("KL to the base model (train/kl_loss)",
     [("train/kl_loss", "train/step", None)], None),
    ("Entropy", [("rollout/entropy", "rollout/step", None)], None),
    ("Groups with no signal (all samples tied)",
     [("rollout/run/groups/no_signal", "rollout/step", None)], "%"),
    ("Negative criteria met",
     [("rollout/rubric/negative_met", "rollout/step", None)], "%"),
)  # fmt: skip


def _number(x) -> float | None:
    try:
        x = float(x)
    except TypeError, ValueError:
        return None
    return x if math.isfinite(x) else None


def series(rows: Sequence[dict], key: str, step: str) -> tuple[list, list]:
    """`key` against `step`, from the history rows that log both, in step order
    (stable: points sharing a step, like two eval rounds, keep their log order)."""
    pts = [
        (row[step], v)
        for row in sorted(rows, key=lambda r: r.get("_step", 0))
        if row.get(step) is not None and (v := _number(row.get(key))) is not None
    ]
    pts.sort(key=lambda p: p[0])
    return [p[0] for p in pts], [p[1] for p in pts]


def fetch_runs(run_ids: Sequence[str], project: str) -> list[dict]:
    """Each W&B run's id, tags, state and full history (reads only)."""
    import wandb

    api = wandb.Api(timeout=60)
    runs = []
    for rid in run_ids:
        r = api.run(f"{project}/{rid}")
        runs.append(
            {"id": r.id, "name": r.name, "tags": list(r.tags), "state": r.state,
             "rows": list(r.scan_history())}
        )  # fmt: skip
        print(f"{rid}: {len(runs[-1]['rows'])} history rows, {r.state}")
    return runs


def run_labels(runs: Sequence[dict]) -> tuple[list[str], list[str]]:
    """Legend labels: the report's name for a known run, else its id and the tags
    not shared by every run; and the shared tags."""
    shared = [t for t in runs[0]["tags"] if all(t in r["tags"] for r in runs)]
    labels = []
    for r in runs:
        if r["id"] in RUN_NAMES:
            labels.append(f"{RUN_NAMES[r['id']]} ({r['id']})")
            continue
        own = [t for t in r["tags"] if t not in shared]
        state = "" if r["state"] == "finished" else f" ({r['state']})"
        labels.append(r["id"] + (" · " + ", ".join(own) if own else "") + state)
    return labels, shared


def training(
    run_ids: Sequence[str] = RUNS, project: str = WANDB_PROJECT, out: Path = OUT
) -> Path:
    """Small multiples of the training runs' W&B metrics, runs overlaid."""
    from matplotlib.lines import Line2D

    plt = _plt()
    runs = fetch_runs(run_ids, project)
    labels, shared = run_labels(runs)
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.2), sharex=True)
    flat = list(axes.flat)
    for ax, (title, metrics, fmt) in zip(flat, PANELS, strict=False):
        for run, color in zip(runs, SERIES, strict=False):
            for i, (key, step, _) in enumerate(metrics):
                x, y = series(run["rows"], key, step)
                if i == 0:
                    ax.plot(x, y, color=color)
                else:  # the held-out eval: a few points
                    ax.plot(x, y, color=color, lw=1, marker="o", ms=7, alpha=0.9)
        styles = [name for _, _, name in metrics if name]
        if styles:
            handles = [
                Line2D([], [], color=INK, lw=2),
                Line2D([], [], color=INK, lw=1, marker="o", ms=6),
            ]
            ax.legend(handles, styles, fontsize=7.5, loc="best", handlelength=1.6)
        if fmt == "%":
            _pct(ax)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    ncols = axes.shape[1]
    for i, ax in enumerate(flat[: len(PANELS)]):
        if i + ncols >= len(PANELS):  # nothing plotted below it
            ax.set_xlabel("rollout step")
            ax.xaxis.set_tick_params(labelbottom=True)
    key = flat[len(PANELS)]  # the spare panel: which run is which
    key.axis("off")
    key.set_title("Runs", fontsize=9)
    for i, (label, color) in enumerate(zip(labels, SERIES, strict=False)):
        y = 0.9 - 0.12 * i
        key.plot([0.02, 0.12], [y, y], color=color, transform=key.transAxes)
        key.text(0.16, y, label, transform=key.transAxes, va="center", fontsize=8)
    if shared:
        note = "all runs: " + ", ".join(shared)
        key.text(0.02, 0.9 - 0.12 * len(runs), note, transform=key.transAxes,
                 va="top", fontsize=7.5, wrap=True)  # fmt: skip
    fig.tight_layout(h_pad=1.2)
    return _save(fig, out, "training", "Training runs")


# --- CLI -----------------------------------------------------------------------

_out = click.option(
    "--out", type=click.Path(path_type=Path), default=OUT, show_default=True
)
_n = click.option("--n", default=1000, show_default=True, help="calibration judgments")
_cal = click.option(
    "--calibration",
    "cal_src",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=calibrate.OUT,
    show_default=True,
)
_root = click.option("--root", type=click.Path(path_type=Path), default=healthbench.RAW)
_runs = click.option(
    "--run",
    "run_ids",
    multiple=True,
    default=RUNS,
    show_default=True,
    help="W&B run id, repeatable: the runs are overlaid",
)
_project = click.option("--project", default=WANDB_PROJECT, show_default=True)


@click.group(help=__doc__.split("\n\n")[0])
def main() -> None:
    pass


@main.command("calibration")
@_cal
@_n
@_out
def calibration_cmd(cal_src: Path, n: int, out: Path) -> None:
    """Balanced F1 per judge configuration, physician ceiling."""
    calibration(cal_src, out, n)


@main.command("length-bias")
@_cal
@_n
@_root
@_out
def length_bias_cmd(cal_src: Path, n: int, root: Path, out: Path) -> None:
    """Share of "met" by completion-length quartile."""
    length_bias(cal_src, out, n, root)


@main.command("self-hosted")
@_cal
@_n
@_root
@_out
def self_hosted_cmd(cal_src: Path, n: int, root: Path, out: Path) -> None:
    """Self-hosted judges vs Haiku: balanced F1, and "met" by length."""
    self_hosted(cal_src, out, n, root)


@main.command("gepa")
@click.option(
    "--result",
    "src",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=OPTIMIZE / REWARD_PROMPT_VERSION,
    show_default=True,
    help="a GEPA output dir (with result.json)",
)
@_out
def gepa_cmd(src: Path, out: Path) -> None:
    """Validation score of each GEPA candidate prompt."""
    gepa(src, out)


@main.command("baselines")
@click.option(
    "--evals",
    "src",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=evals.OUT,
    show_default=True,
)
@_out
def baselines_cmd(src: Path, out: Path) -> None:
    """Eval scores with 95% CIs: full rubric and each half."""
    baselines(src, out)


@main.command("two-judges")
@click.option(
    "--evals",
    "src",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=evals.OUT,
    show_default=True,
)
@_out
def two_judges_cmd(src: Path, out: Path) -> None:
    """Gain over the base model per checkpoint, under Haiku and gpt-oss."""
    two_judges(src, out)


@main.command("training")
@_runs
@_project
@_out
def training_cmd(run_ids: tuple[str, ...], project: str, out: Path) -> None:
    """Training metrics from W&B, one or more runs overlaid."""
    training(run_ids, project, out)


@main.command("all")
@_runs
@_project
@_out
def all_cmd(run_ids: tuple[str, ...], project: str, out: Path) -> None:
    """Every figure, from the default sources."""
    calibration(out=out)
    length_bias(out=out)
    self_hosted(out=out)
    gepa(out=out)
    baselines(out=out)
    two_judges(out=out)
    training(run_ids, project, out)


if __name__ == "__main__":
    main()
