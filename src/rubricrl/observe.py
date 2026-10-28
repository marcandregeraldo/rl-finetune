"""W&B Weave: one trace per graded response, its judge calls as children.

A trace is one unit of work, the way the reward is computed: one response, judged
on each of its criteria. The root (`grade_response`) holds the conversation as a
chat (every turn, then the graded response as the last assistant turn) and the
result; each child (`judge_verdict`) is one judge call: the rubric item, the
verdict, the explanation, tokens. Filter the Traces table on the op
`judge_verdict` to see every judge call across traces.

Optional (`uv sync --extra observe`): every `weave` import is inside a function, so
the base install and the Miles image never need it. Anthropic calls are not
auto-traced (`implicitly_patch_integrations` off): what is logged is decided here.

- **Calibration** (`analysis/calibrate.py run --weave PROJECT`), logged as it runs:
  a root per meta-eval completion, a child per rubric category it was graded on,
  with the physicians' labels and agreement with them. The retest pass is a
  second root per completion (`repeat` 1). Each judge configuration also gets a
  summary-only evaluation: balanced F1 is pairwise over physician labels, not a
  per-item mean.
- **Training** (`modal/miles_rft.py::sync_traces`): the reward's `rewards.jsonl`,
  after the fact. A root per graded sample (score, reward, axes), a child per
  criterion; each in-loop eval round also gets a summary-only evaluation.

**W&B key:** `WANDB_API_KEY`, else (macOS) the Keychain item `wandb-api-key`, set
in this process's environment only. Never logged or written anywhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence

from rubricrl.data.schema import Judgment
from rubricrl.judge.client import Verdict
from rubricrl.judge.prompt import GRADER_TEMPLATE, PROMPT_VERSION

PROJECT = "rl-finetune"
PROMPT_NAME = "healthbench-grader"
ROOT_OP = "grade_response"
OP = "judge_verdict"


def _ensure_wandb_key() -> None:
    if os.environ.get("WANDB_API_KEY") or not shutil.which("security"):
        return
    r = subprocess.run(
        ["security", "find-generic-password", "-s", "wandb-api-key", "-w"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        os.environ["WANDB_API_KEY"] = r.stdout.strip()


def init(project: str = PROJECT):
    import weave

    _ensure_wandb_key()
    settings = {"implicitly_patch_integrations": False, "print_call_link": False}
    return weave.init(project, settings=settings)


def _url(ev) -> str | None:
    try:
        return ev.ui_url
    except ValueError:  # no call id: Weave disabled (WEAVE_DISABLED=true)
        return None


def as_chat(messages: list[dict], response: str) -> list[dict]:
    """The conversation with the graded response as its last assistant turn."""
    return [*messages, {"role": "assistant", "content": response}]


def _response_name(call) -> str:
    a = call.attributes
    retest = " · retest" if a.get("repeat") else ""
    return f"{a.get('run')} · {a.get('completion_id')}{retest}"


def _verdict_name(call) -> str:
    return str(call.attributes.get("category")).removeprefix("cluster:")


def _without_conversation(inputs: dict) -> dict:
    """A judge call's logged inputs: the conversation is on its parent already."""
    return {k: v for k, v in inputs.items() if k not in ("messages", "completion")}


# --- Calibration ---------------------------------------------------------------


def publish_prompt(template: str = GRADER_TEMPLATE, version: str = PROMPT_VERSION):
    """The grader template, versioned by its text and tagged with its version
    (an optimized template: `judge/optimize.py`).

    It holds literal JSON braces: never `.format()` the published prompt.
    """
    import weave

    prompt = weave.StringPrompt(template)
    return weave.publish(prompt, name=PROMPT_NAME, tags=[version])


def judgment_row(j: Judgment) -> dict:
    return {
        "id": j.id,
        "prompt_id": j.prompt_id,
        "category": j.category,
        "messages": j.messages,
        "completion": j.completion,
        "rubric": j.rubric,
        "physician_labels": list(j.physician_labels),
        "completion_chars": len(j.completion),
    }


def publish_judgments(subset: Sequence[Judgment], seed: int = 0):
    """The meta-eval sample as a Dataset; the name says which (n, seed) it is."""
    import weave

    dataset = weave.Dataset(
        name=f"healthbench-meta-eval-n{len(subset)}-s{seed}",
        rows=[judgment_row(j) for j in subset],
    )
    weave.publish(dataset)
    return dataset


def agreement(met: bool | None, labels: Sequence[bool]) -> dict[str, float | bool]:
    """Agreement with the physicians; empty for a failed verdict, and no majority
    when the physicians are tied."""
    if met is None:
        return {}
    out: dict[str, float | bool] = {
        "physician_agreement": sum(x == met for x in labels) / len(labels)
    }
    votes = Counter(labels)
    if votes[True] != votes[False]:
        out["agrees_with_majority"] = met == (votes[True] > votes[False])
    return out


def verdict_output(v: Verdict) -> dict:
    return {
        "met": v.met,
        "explanation": v.explanation,
        "error": v.error,
        "cached": v.cached,
        "input_tokens": v.input_tokens,
        "output_tokens": v.output_tokens,
    }


def traced_grader(
    judge, attributes: dict
) -> Callable[[Sequence[Judgment], int], Awaitable[list[Verdict]]]:
    """Grade judgments with one root trace per completion, one child per judgment.

    `attributes` describe the run (`run_attributes`). Returns
    `grade(judgments, repeat) -> verdicts`, in the order given.
    """
    import weave

    judgments_by_id: dict[str, Judgment] = {}
    verdicts: dict[tuple[str, int], Verdict] = {}

    @weave.op(
        name=OP,
        call_display_name=_verdict_name,
        postprocess_inputs=_without_conversation,
    )
    async def judge_verdict(
        messages: list[dict],
        completion: str,
        rubric_item: str,
        repeat: int,
        physician_labels: list[bool],
    ) -> dict:
        v = await judge.grade(messages, completion, rubric_item, repeat)
        return verdict_output(v) | agreement(v.met, physician_labels)

    async def child(j: Judgment, repeat: int) -> dict:
        with weave.attributes({"item": j.id, "category": j.category}):
            out = await judge_verdict(
                j.messages, j.completion, j.rubric, repeat, list(j.physician_labels)
            )
        verdicts[(j.id, repeat)] = Verdict(
            **{k: out[k] for k in verdict_output(Verdict(None))}
        )
        return out | {"item": j.id, "category": j.category}

    @weave.op(name=ROOT_OP, call_display_name=_response_name)
    async def grade_response(
        messages: list[dict], criteria: list[dict], repeat: int
    ) -> dict:
        group = [judgments_by_id[c["id"]] for c in criteria]
        outs = await asyncio.gather(*(child(j, repeat) for j in group))
        agree = [o["agrees_with_majority"] for o in outs if "agrees_with_majority" in o]
        return {
            "criteria": len(outs),
            "met": sum(o["met"] is True for o in outs),
            "failed": sum(o["met"] is None for o in outs),
            "agrees_with_majority": sum(agree) / len(agree) if agree else None,
            "verdicts": [
                {k: o.get(k) for k in ("category", "met", "agrees_with_majority")}
                for o in outs
            ],
        }

    async def grade(judgments: Sequence[Judgment], repeat: int = 0) -> list[Verdict]:
        groups: dict[str, list[Judgment]] = defaultdict(list)
        for j in judgments:
            groups[j.completion_id].append(j)
            judgments_by_id[j.id] = j

        async def root(group: list[Judgment]) -> None:
            first = group[0]
            criteria = [
                {
                    "id": j.id,
                    "category": j.category,
                    "rubric_item": j.rubric,
                    "physician_labels": list(j.physician_labels),
                }
                for j in group
            ]
            extra = {
                "completion_id": first.completion_id,
                "prompt_id": first.prompt_id,
                "repeat": repeat,
            }
            with weave.attributes(attributes | extra):
                await grade_response(
                    as_chat(first.messages, first.completion), criteria, repeat
                )

        await asyncio.gather(*(root(g) for g in groups.values()))
        return [verdicts.pop((j.id, repeat)) for j in judgments]

    return grade


def run_attributes(
    label: str,
    config,
    n: int,
    seed: int,
    prompt_ref,
    prompt_version: str = PROMPT_VERSION,
) -> dict:
    """What identifies a calibration run, on its traces and its summary."""
    return {
        "run": label,
        "model": config.model,
        "effort": config.effort,
        "thinking": config.thinking,
        "n": n,
        "seed": seed,
        "prompt_version": prompt_version,
        "grader_prompt": str(prompt_ref.uri()) if prompt_ref is not None else None,
    }


def calibration_summary(result: dict) -> dict:
    o, rt = result["judge"]["overall"], result["retest"]
    return {
        "f1_balanced": o["f1_balanced"],
        "f1_met": o["f1_pos"],
        "f1_not_met": o["f1_neg"],
        "rate_met": result["judge_rate_met"],
        "physician_rate_met": result["physician_rate_met"],
        "physician_ceiling_f1_balanced": result["physicians_all"]["overall"][
            "f1_balanced_mean"
        ],
        "retest_agreement": rt["agreement"],
        "retest_kappa": rt["kappa"],
        "f1_balanced_by_length": {
            f"{b['chars_min']}-{b['chars_max']}": b["f1_balanced"]
            for b in result["by_length"]
        },
        "failed": result["failed"],
        "cost_usd_new_calls": result["usage"]["cost_usd"],
        # how fast it graded (older results have none); self-hosted: GPU $
        **{f"throughput_{k}": v for k, v in result.get("throughput", {}).items()},
    }


def log_calibration_summary(result: dict, attributes: dict, dataset) -> str | None:
    """One summary-only evaluation per judge configuration; returns its URL."""
    import weave

    ev = weave.EvaluationLogger(
        name="judge-calibration",
        model={"name": attributes["run"]} | attributes,
        dataset=dataset,
        eval_attributes=attributes,
    )
    ev.log_summary(calibration_summary(result))
    return _url(ev)


# --- Training records ----------------------------------------------------------


def sampled(run_id: str, index: int | str, rate: float) -> bool:
    digest = hashlib.sha256(f"{run_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def split_records(
    records: Sequence[dict],
    train_seen: int,
    batch: int,
    rate: float,
    run_id: str,
    final: bool = False,
) -> tuple[list[tuple[int, list[dict]]], list[tuple[int, dict]], int, int]:
    """Group a chunk of `rewards.jsonl` for logging.

    Returns (eval rounds as (step, records), sampled training records as (step,
    record), records consumed, training records seen). The step is approximate:
    training records so far // `batch` (prompts x samples per step). A trailing eval
    round may still be growing, so it is left for the next sync unless `final`.
    """
    rounds: list[tuple[int, list[dict]]] = []
    traces: list[tuple[int, dict]] = []
    consumed = 0
    current: list[dict] = []
    for i, rec in enumerate(records):
        if rec.get("eval"):
            current.append(rec)
            continue
        if current:
            rounds.append((train_seen // batch, current))
            current = []
        if sampled(run_id, train_seen, rate):
            traces.append((train_seen // batch, rec))
        train_seen += 1
        consumed = i + 1
    if current and final:
        rounds.append((train_seen // batch, current))
        consumed = len(records)
    elif not current:
        consumed = len(records)
    return rounds, traces, consumed, train_seen


def criteria_of(rec: dict) -> list[dict]:
    """The record's criteria; older logs only have verdicts by criterion id."""
    return rec.get("criteria") or [
        {"id": k, "met": m} for k, m in (rec.get("met") or {}).items()
    ]


def _log_response(
    run_id: str, step: int, rec: dict, conversations: dict[str, list[dict]]
) -> int:
    """One graded sample: a root with the chat and the score, a child per verdict."""
    import weave

    pid = rec.get("prompt_id")
    criteria = criteria_of(rec)
    kind = "eval" if rec.get("eval") else "train"
    root = weave.log_call(
        op=ROOT_OP,
        inputs={
            "messages": as_chat(conversations.get(pid, []), rec.get("response") or ""),
            "criteria": [
                {k: c.get(k) for k in ("id", "text", "points")} for c in criteria
            ],
        },
        output={k: rec.get(k) for k in ("score", "reward", "axes", "failed", "chars")}
        | {"met": sum(c.get("met") is True for c in criteria)},
        attributes={
            "run": run_id,
            "step": step,
            "eval": kind == "eval",
            "prompt_id": pid,
            "time": rec.get("time"),
            "tokens": rec.get("tokens"),
        },
        display_name=f"{run_id} · step {step} · {kind} · {pid}",
        use_stack=False,
    )
    for c in criteria:
        weave.log_call(
            op=OP,
            inputs={"rubric_item": f"[{c.get('points')}] {c.get('text')}"},
            output={"met": c.get("met"), "explanation": c.get("explanation")},
            attributes={"item": c.get("id"), "category": kind},
            parent=root,
            display_name=f"{kind} · {c.get('id')}",
            use_stack=False,
        )
    return len(criteria)


def log_training(
    run_id: str,
    rounds: Sequence[tuple[int, list[dict]]],
    traces: Sequence[tuple[int, dict]],
    conversations: dict[str, list[dict]] | None = None,
) -> dict:
    """A trace per given record (root + a child per verdict); each eval round also
    as a summary-only evaluation. `conversations`: prompt_id -> messages. Returns
    counts and the evaluations' URLs."""
    import weave

    conversations = conversations or {}
    urls, responses, verdicts = [], 0, 0
    for step, recs in rounds:
        for rec in recs:
            verdicts += _log_response(run_id, step, rec, conversations)
            responses += 1
        scores = [r["score"] for r in recs if r.get("score") is not None]
        ev = weave.EvaluationLogger(
            name="healthbench-val",
            model={"name": f"{run_id}@step{step:04d}", "run_id": run_id, "step": step},
            dataset="healthbench-val",
            eval_attributes={"run_id": run_id, "step": step},
        )
        ev.log_summary(
            {
                "heldout_score_mean": statistics.fmean(scores) if scores else None,
                "chars_median": statistics.median(r.get("chars", 0) for r in recs),
                "n": len(recs),
            }
        )
        urls.append(_url(ev))
    for step, rec in traces:
        verdicts += _log_response(run_id, step, rec, conversations)
        responses += 1
    return {
        "responses": responses,
        "verdicts": verdicts,
        "eval_rounds": len(rounds),
        "urls": urls,
    }


# --- The consultation environment ------------------------------------------------


def episode_tracer(project: str, rate: float | None = None):
    """A callback for `env.server`: one trace per finished episode.

    Root `consultation`: the whole transcript as a chat, the result, the hidden
    facts drawn out; a `judge_verdict` child per criterion. Eval episodes are all
    traced; training ones at `rate` (`RUBRICRL_TRACE_RATE`, default 0.1).
    """
    import weave

    from rubricrl.sim.patient import PATIENT_VERSION

    init(project)
    rate = float(os.environ.get("RUBRICRL_TRACE_RATE", "0.1")) if rate is None else rate
    run = os.environ.get("RUBRICRL_TRACE_RUN", "")

    def trace(scenario, obs) -> None:
        key = f"{obs.scenario_id}:{obs.messages[-1]['content'][:200]}"
        if obs.half != "eval" and not sampled(run or "env", key, rate):
            return
        root = weave.log_call(
            op="consultation",
            inputs={"messages": obs.messages, "scenario": scenario.id},
            output={
                "reward": obs.reward,
                "score": obs.score,
                "reveals": obs.reveals,
                "axes": obs.axes,
                "failed": obs.failed,
                "turns": obs.turn,
                "end_reason": obs.end_reason,
                "revealed_facts": obs.revealed_facts,
                "hidden_facts": [f.text for f in scenario.hidden_facts],
            },
            attributes={
                "run": run,
                "mode": obs.mode,
                "half": obs.half,
                "scenario": scenario.id,
                "theme": scenario.theme,
                "split": scenario.split,
                "scenario_version": scenario.version,
                "patient_version": PATIENT_VERSION if obs.mode == "patient" else None,
            },
            display_name=f"{obs.mode} · {obs.half} · {scenario.id}",
            use_stack=False,
        )
        for c in obs.criteria:
            weave.log_call(
                op=OP,
                inputs={"rubric_item": f"[{c.get('points')}] {c.get('text')}"},
                output={"met": c.get("met"), "explanation": c.get("explanation")},
                attributes={"item": c.get("id"), "source": c.get("source")},
                parent=root,
                display_name=str(c.get("id")),
                use_stack=False,
            )

    return trace


# --- W&B runs: the experiments, repeatable -------------------------------------------


def _git() -> dict:
    """The commit a run came from (passed in by Modal entrypoints, where there's
    no git checkout: RUBRICRL_GIT_SHA / RUBRICRL_GIT_DIRTY)."""
    sha = os.environ.get("RUBRICRL_GIT_SHA")
    if sha is None and shutil.which("git"):
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        sha = r.stdout.strip() or None
        d = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        dirty = bool(d.stdout.strip())
    else:
        dirty = os.environ.get("RUBRICRL_GIT_DIRTY") == "1"
    return {"git_sha": sha, "git_dirty": dirty}


def log_wandb_calibration(project: str, result: dict, config: dict) -> str | None:
    """One W&B run per calibration (`job_type="calibration"`): its config, the
    metrics in the summary, per-category and per-length tables, the confusion
    matrix against the physician majority, and the result JSON as an artifact."""
    import json
    import tempfile

    import wandb

    _ensure_wandb_key()
    run = wandb.init(
        project=project,
        job_type="calibration",
        group=config.get("prompt_version", PROMPT_VERSION),
        name=f"{result['label']}-n{result['n']}",
        config=config | _git(),
        tags=["self-hosted"] if config.get("self_hosted") else None,
        reinit="finish_previous",
    )
    run.summary.update(
        calibration_summary(result) | {"weave_url": result.get("weave_url")}
    )
    phys = result["physicians_all"]
    rows = [
        [
            c.removeprefix("cluster:"),
            m.get("f1_balanced"),
            phys.get(c, {}).get("f1_balanced_mean"),
            m.get("n_pairs"),
        ]
        for c, m in sorted(result["judge"].items())
        if c != "overall"
    ]
    by_cat = wandb.Table(
        columns=["category", "judge_f1_balanced", "physicians_f1_balanced", "pairs"],
        data=rows,
    )
    by_len = wandb.Table(
        columns=["bucket", "chars_min", "chars_max", "n", "f1_balanced"],
        data=[
            [b["bucket"], b["chars_min"], b["chars_max"], b["n"], b["f1_balanced"]]
            for b in result["by_length"]
        ],
    )
    run.log(
        {
            "by_category": by_cat,
            "by_length": by_len,
            "f1_by_category": wandb.plot.bar(
                by_cat, "category", "judge_f1_balanced", title="Balanced F1 by category"
            ),
            "f1_by_length": wandb.plot.line(
                by_len, "bucket", "f1_balanced", title="Balanced F1 by length quartile"
            ),
        }
    )
    if (cm := result.get("confusion")) is not None:
        run.log(
            {
                "confusion_vs_majority": wandb.Table(
                    columns=list(cm), data=[list(cm.values())]
                )
            }
        )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "result.json")
        with open(path, "w") as f:
            json.dump(result, f)
        art = wandb.Artifact(
            f"calibration-{result['label']}-n{result['n']}", "calibration-result"
        )
        art.add_file(path)
        run.log_artifact(art)
    url = run.url
    run.finish()
    return url


def log_wandb_scenarios(project: str, split: str, summary: dict, path) -> str | None:
    """A scenario build as a W&B run (`job_type="scenarios"`): what the mapping
    kept, rewrote and dropped, the cost, and the scenarios as an artifact."""
    import wandb

    _ensure_wandb_key()
    run = wandb.init(
        project=project,
        job_type="scenarios",
        name=f"scenarios-{split}-{summary.get('version')}",
        config={
            "split": split,
            "version": summary.get("version"),
            "model": summary.get("model"),
        }
        | _git(),
        reinit="finish_previous",
    )
    flat = {f"statuses/{k}": v for k, v in summary.get("statuses", {}).items()}
    flat |= {f"share/{k}": v for k, v in summary.get("share", {}).items()}
    run.summary.update(
        flat
        | {
            k: summary[k]
            for k in ("scenarios", "criteria", "hidden_facts_mean", "cost_usd")
            if k in summary
        }
    )
    themes = summary.get("by_theme", {})
    statuses = sorted({s for c in themes.values() for s in c})
    run.log(
        {
            "by_theme": wandb.Table(
                columns=["theme", *statuses],
                data=[[t, *(c.get(s, 0) for s in statuses)] for t, c in themes.items()],
            )
        }
    )
    art = wandb.Artifact(f"scenarios-{split}", "scenarios")
    art.add_file(str(path))
    run.log_artifact(art)
    url = run.url
    run.finish()
    return url


def log_wandb_check(project: str, check: str, out: dict) -> str | None:
    """A simulation check as a W&B run (`job_type="env-check"`): the scalar
    results in the summary, the per-item rows (disagreements, episodes) as tables."""
    import json

    import wandb

    _ensure_wandb_key()
    run = wandb.init(
        project=project, job_type="env-check", name=check, config=_git(),
        reinit="finish_previous",
    )  # fmt: skip

    def scalars(d: dict, prefix: str = "") -> dict:
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat |= scalars(v, f"{prefix}{k}/")
            elif isinstance(v, (int, float, str)) or v is None:
                flat[f"{prefix}{k}"] = v
        return flat

    run.summary.update(scalars(out))
    for key, rows in _row_lists(out):
        cols = sorted({c for r in rows for c in r})
        data = [
            [
                json.dumps(r.get(c)) if isinstance(r.get(c), (list, dict)) else r.get(c)
                for c in cols
            ]
            for r in rows
        ]
        run.log({key: wandb.Table(columns=cols, data=data)})
    url = run.url
    run.finish()
    return url


def _row_lists(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
            yield f"{prefix}{k}", v
        elif isinstance(v, dict):
            yield from _row_lists(v, f"{prefix}{k}/")
