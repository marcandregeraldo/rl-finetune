"""Judge prompt optimization with GEPA, against physician labels (oss_meta_eval).

GEPA (reflective prompt evolution) rewrites the grader template:
1. the judge grades a minibatch of meta-eval judgments with a candidate template;
2. a reflection model (Sonnet) reads the judge's verdicts and explanations next
   to the physicians' labels and writes a new template;
3. a new template that agrees with physicians more on that minibatch is scored
   on the validation set and joins the pool; the best on validation wins.

**The judge is the production one.** `ClaudeJudge` with its config (structured
output, effort, thinking) and the verdict cache. This is why it's the `gepa`
library with an adapter here, not DSPy: DSPy would call the model through its
own prompt format and LiteLLM, and optimize a judge that isn't the one the
reward uses.

**Data:** nothing here touches the calibration test set.
- **Test:** `calibrate`'s n=1000 subset (seed 0), scored afterwards with
  `calibrate run --template`, next to the baseline.
- **Held-out categories:** 6 of the 34 rubric categories never appear in train
  or val, so the test set shows whether a gain carries to rubric texts the
  optimizer never saw. The reward mostly uses criteria written for one
  conversation each, so the guidance has to be general.
- **Train / val:** judgments from conversations with no item in the test set,
  in the other 28 categories, with a physician majority (ties dropped),
  balanced 50/50 between met and not met. Physicians say "met" 78% of the time;
  balancing keeps the score from rewarding a judge that always says "met", as
  balanced F1 (the test metric) would not.

**Score per item:** the share of physicians the judge agrees with: the
calibration metric's pairs, item by item.

    python -m rubricrl.judge.optimize run --name gepa-v1 --wandb rl-finetune
    python -m rubricrl.analysis.calibrate run --judge haiku --effort medium \\
        --thinking --n 1000 --template analysis/out/optimize/gepa-v1/gepa-v1.txt

Output: `analysis/out/optimize/<name>/`: the best template (`<name>.txt`),
every candidate with its val scores and the reflection's diagnosis
(`result.json`), and GEPA's own run directory (`gepa/`, which a rerun resumes).
"""

import asyncio
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import click

from rubricrl.analysis.calibrate import (
    _f,
    judge_metrics,
    pairwise_f1,
    sample_judgments,
)
from rubricrl.data import healthbench
from rubricrl.data.schema import Judgment
from rubricrl.judge.client import ClaudeJudge, JudgeConfig, Verdict, VerdictCache
from rubricrl.judge.prompt import GRADER_TEMPLATE, render
from rubricrl.llm import JSONCache, StructuredLLM

OUT = Path("analysis/out/optimize")
COMPONENT = "grader_template"
PLACEHOLDERS = ("<<conversation>>", "<<rubric_item>>")
REFLECTION_VERSION = "gepa-reflection-v1"
MAX_CHARS = 6000  # per message shown to the reflection model


# --- Data ----------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    judgment: Judgment
    majority: bool  # the physicians' majority label


def majority(j: Judgment) -> bool | None:
    met = sum(j.physician_labels)
    return (
        None
        if 2 * met == len(j.physician_labels)
        else 2 * met > len(j.physician_labels)
    )


def heldout_categories(judgments: Sequence[Judgment], k: int = 6, seed: int = 0):
    cats = sorted({j.category for j in judgments})
    return set(sorted(cats, key=lambda c: healthbench._unit("cat", seed, c))[:k])


def splits(
    judgments: Sequence[Judgment],
    n_train: int = 400,
    n_val: int = 200,
    test_n: int = 1000,
    seed: int = 0,
) -> dict:
    """Train and val (balanced on the physician majority), disjoint from the
    calibration test set's conversations and from the held-out categories."""
    test = sample_judgments(list(judgments), test_n, seed)
    test_prompts = {j.prompt_id for j in test}
    held = heldout_categories(judgments, seed=seed)
    pool = [
        Item(j, m)
        for j in judgments
        if j.prompt_id not in test_prompts
        and j.category not in held
        and (m := majority(j)) is not None
    ]
    pool.sort(key=lambda it: healthbench._unit("optimize", seed, it.judgment.id))
    by_class = {c: [it for it in pool if it.majority == c] for c in (True, False)}
    half_train, half_val = n_train // 2, n_val // 2
    for c, items in by_class.items():
        if len(items) < half_train + half_val:
            raise ValueError(f"only {len(items)} items with majority {c}")
    train = [it for c in (True, False) for it in by_class[c][:half_train]]
    val = [
        it
        for c in (True, False)
        for it in by_class[c][half_train : half_train + half_val]
    ]
    order = lambda it: healthbench._unit("order", seed, it.judgment.id)  # noqa: E731
    return {
        "train": sorted(train, key=order),
        "val": sorted(val, key=order),
        "test": test,
        "heldout_categories": sorted(held),
    }


# --- Scoring and feedback ------------------------------------------------------


def template_problem(template: str) -> str | None:
    for p in PLACEHOLDERS:
        if template.count(p) != 1:
            return f"the template must contain {p} exactly once"
    return None


def agreement(met: bool | None, labels: Sequence[bool]) -> float:
    """The share of physicians who gave the judge's label (0 if no verdict)."""
    return 0.0 if met is None else sum(x == met for x in labels) / len(labels)


def _clip(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters]"


def feedback(item: Item, v: Verdict) -> str:
    j = item.judgment
    met, n = sum(j.physician_labels), len(j.physician_labels)
    truth = "met" if item.majority else "not met"
    if v.met is None:
        return f"The judge gave no verdict ({v.error}). Physicians: {truth}."
    said = "met" if v.met else "not met"
    verdict = "agrees with" if v.met == item.majority else "DISAGREES with"
    return (
        f"Physicians: {met} of {n} said the rubric item is met, so the reference "
        f"label is {truth}. The judge said {said}, which {verdict} the physicians."
    )


class JudgeAdapter:
    """GEPA's adapter: a candidate is {COMPONENT: grader template}; the judge
    grades each item with it; the score is agreement with the physicians."""

    propose_new_texts = None  # proposals come from `Reflector`

    def __init__(self, judge, loop: asyncio.AbstractEventLoop | None = None):
        self.judge = judge
        self.loop = loop or asyncio.new_event_loop()

    def evaluate(self, batch: list[Item], candidate: dict, capture_traces=False):
        from gepa import EvaluationBatch

        template = candidate[COMPONENT]
        if (problem := template_problem(template)) is not None:
            verdicts = [Verdict(None, error=problem) for _ in batch]
        else:
            verdicts = self.loop.run_until_complete(self.grade(batch, template))
        scores = [
            agreement(v.met, it.judgment.physician_labels)
            for it, v in zip(batch, verdicts, strict=True)
        ]
        return EvaluationBatch(
            outputs=[v.met for v in verdicts],
            scores=scores,
            trajectories=(
                [
                    {"item": it, "verdict": v}
                    for it, v in zip(batch, verdicts, strict=True)
                ]
                if capture_traces
                else None
            ),
        )

    async def grade(self, batch: Sequence[Item], template: str) -> list[Verdict]:
        return list(
            await asyncio.gather(
                *(
                    self.judge.grade_prompt(
                        render(
                            it.judgment.messages,
                            it.judgment.completion,
                            it.judgment.rubric,
                            template,
                        )
                    )
                    for it in batch
                )
            )
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for t in eval_batch.trajectories:
            item, v = t["item"], t["verdict"]
            j = item.judgment
            records.append(
                {
                    "Inputs": {
                        "conversation": "\n\n".join(
                            f"{m['role']}: {_clip(m['content'])}" for m in j.messages
                        ),
                        "response being graded": _clip(j.completion),
                        "rubric item": j.rubric,
                    },
                    "Judge output": {
                        "explanation": v.explanation or "(none)",
                        "criteria_met": v.met,
                    },
                    "Feedback": feedback(item, v),
                }
            )
        return {c: records for c in components_to_update}


# --- Reflection ----------------------------------------------------------------

REFLECTION_TEMPLATE = """I use the prompt template below to have a model grade whether one response in a medical conversation meets one rubric item. Physicians wrote the rubric items, and physicians also labelled whether each response meets each item. The goal is for the model's verdicts to agree with the physicians' labels.

Current template:
```
<curr_param>
```

Below are examples graded with this template: the inputs, the model's output, and feedback giving the physicians' labels.

<side_info>

Write an improved template.

Requirements:
- It must contain the placeholders <<conversation>> and <<rubric_item>> exactly once each. They are replaced by the conversation (ending with the assistant response being graded) and by the rubric item.
- The model's output format is enforced separately: a JSON object with "explanation" (a string, written first) and "criteria_met" (a boolean). Keep any instructions about the output consistent with that format.
- Rubric items can describe desirable or undesirable behaviour (for example "fails to ..."). "criteria_met" says whether the item's description holds of the response, not whether the response is good.
- The template will grade rubric items and conversations not shown here, including thousands of items written for one conversation each. Write general guidance on how physicians apply rubric items. Do not refer to the specific examples, rubric texts or medical topics above.
- Look for systematic differences between the model's verdicts and the physicians' (for example, being too strict or too lenient, and in which situations) and correct them. Keep what already works.

Return a short diagnosis of those differences, and the complete new template."""

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "template": {"type": "string"},
    },
    "required": ["diagnosis", "template"],
    "additionalProperties": False,
}


class Reflector:
    """GEPA's candidate proposer: the reflection model rewrites the template
    from a reflective dataset (structured output: no fence parsing)."""

    def __init__(self, llm, loop: asyncio.AbstractEventLoop):
        self.llm, self.loop = llm, loop
        self.proposals: list[dict] = []

    def prompt(self, current: str, records: Sequence[dict]) -> str:
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        return InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": current,
                "dataset_with_feedback": list(records),
                "prompt_template": REFLECTION_TEMPLATE,
            }
        )

    def __call__(self, candidate, reflective_dataset, components_to_update):
        new = {}
        for c in components_to_update:
            prompt = self.prompt(candidate[c], reflective_dataset[c])
            out = self.loop.run_until_complete(
                self.llm(prompt, REFLECTION_SCHEMA, REFLECTION_VERSION)
            )
            self.proposals.append(
                {
                    "diagnosis": out["diagnosis"],
                    "template": out["template"],
                    "valid": template_problem(out["template"]) is None,
                }
            )
            new[c] = out["template"]
        return new


class Budget:
    """GEPA stop condition: judge + reflection spend reaches `usd`.
    `spenders`: (Usage, model) pairs."""

    def __init__(self, usd: float, *spenders):
        self.usd, self.spenders = usd, spenders

    def spent(self) -> float:
        return sum(usage.cost(model) for usage, model in self.spenders)

    def __call__(self, gepa_state) -> bool:
        return self.spent() >= self.usd


# --- Run -----------------------------------------------------------------------


def val_metrics(items: Sequence[Item], labels: Sequence[bool | None]) -> dict:
    """Balanced F1 against every physician label (the calibration metric) and
    the rate of "met", on the (balanced) val set."""
    m = judge_metrics([it.judgment for it in items], labels)["overall"]
    given = [x for x in labels if x is not None]
    return {
        "f1_balanced": m["f1_balanced"],
        "f1_pos": m["f1_pos"],
        "f1_neg": m["f1_neg"],
        "rate_met": sum(given) / max(1, len(given)),
        "agreement": statistics.fmean(
            agreement(x, it.judgment.physician_labels)
            for it, x in zip(items, labels, strict=True)
        ),
    }


def run(
    name: str = "gepa-v1",
    judge_name: str = "haiku",
    effort: str = "medium",
    thinking: bool = True,
    reflection: str = "sonnet",
    reflection_effort: str = "medium",
    n_train: int = 400,
    n_val: int = 200,
    minibatch: int = 8,
    max_metric_calls: int = 3000,
    budget_usd: float = 8.0,
    seed: int = 0,
    gepa_seed: int | None = None,
    group: str | None = None,
    test_n: int = 1000,
    concurrency: int = 32,
    out: Path = OUT,
    wandb_project: str | None = None,
    judge_cache: str | None = None,
    llm_cache: str | None = None,
    root: Path = healthbench.RAW,
    judgments: Sequence[Judgment] | None = None,
    judge=None,
    reflection_llm=None,
) -> dict:
    """Optimize the grader template; write the best one and every candidate.
    `seed`: the data (train / val / test, held-out categories); `gepa_seed`
    (default: `seed`): GEPA's own sampling (minibatches, parent selection), to
    repeat a run on the same data. `group`: the W&B group (default: `name`).
    `judge_cache` / `llm_cache`: SQLite paths (default ~/.cache/rubricrl).
    `judge` / `reflection_llm` / `judgments`: injected by tests."""
    import gepa

    judgments = judgments if judgments is not None else healthbench.load_judgments(root)
    data = splits(judgments, n_train, n_val, test_n, seed)
    loop = asyncio.new_event_loop()
    config = JudgeConfig.named(
        judge_name, effort=effort, thinking=thinking, concurrency=concurrency
    )
    judge = judge or ClaudeJudge(
        config, VerdictCache(judge_cache) if judge_cache else VerdictCache()
    )
    reflection_llm = reflection_llm or StructuredLLM(
        JudgeConfig.named(
            reflection, effort=reflection_effort, thinking=True, max_tokens=16000
        ),
        JSONCache(llm_cache) if llm_cache else JSONCache(),
    )
    adapter = JudgeAdapter(judge, loop)
    reflector = Reflector(reflection_llm, loop)
    budget = Budget(
        budget_usd,
        (judge.usage, config.model),
        (reflection_llm.usage, reflection_llm.config.model),
    )
    run_config = {
        "name": name, "judge": config.model, "effort": effort, "thinking": thinking,
        "reflection": reflection_llm.config.model,
        "reflection_effort": reflection_effort, "n_train": len(data["train"]),
        "n_val": len(data["val"]), "minibatch": minibatch,
        "max_metric_calls": max_metric_calls, "budget_usd": budget_usd, "seed": seed,
        "gepa_seed": seed if gepa_seed is None else gepa_seed,
        "heldout_categories": data["heldout_categories"],
    }  # fmt: skip
    print(json.dumps(run_config))
    wandb_run = None
    if wandb_project:
        import wandb

        from rubricrl import observe

        observe._ensure_wandb_key()
        wandb_run = wandb.init(
            project=wandb_project,
            job_type="optimization",
            group=group or name,
            name=f"{name}-{config.model}-{effort}" + ("-thinking" if thinking else ""),
            config=run_config | observe._git(),
            reinit="finish_previous",
        )
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    result = gepa.optimize(
        seed_candidate={COMPONENT: GRADER_TEMPLATE},
        trainset=data["train"],
        valset=data["val"],
        adapter=adapter,
        custom_candidate_proposer=reflector,
        reflection_minibatch_size=minibatch,
        max_metric_calls=max_metric_calls,
        stop_callbacks=[budget],
        run_dir=str(folder / "gepa"),
        use_wandb=wandb_run is not None,
        wandb_attach_existing=wandb_run is not None,
        seed=run_config["gepa_seed"],
        display_progress_bar=False,
    )
    # Every candidate in the pool was scored on all of val: these are cache hits.
    candidates = []
    for i, cand in enumerate(result.candidates):
        labels = adapter.evaluate(data["val"], cand).outputs
        candidates.append(
            {
                "index": i,
                "parents": result.parents[i],
                "val_score": result.val_aggregate_scores[i],
                **val_metrics(data["val"], labels),
                "template": cand[COMPONENT],
            }
        )
    best = candidates[result.best_idx]
    template_path = folder / f"{name}.txt"
    template_path.write_text(best["template"])
    usage = {
        "judge": vars(judge.usage) | {"cost_usd": judge.usage.cost(config.model)},
        "reflection": vars(reflection_llm.usage)
        | {"cost_usd": reflection_llm.usage.cost(reflection_llm.config.model)},
    }
    summary = {
        "config": run_config,
        "best_index": result.best_idx,
        "baseline_val": {k: v for k, v in candidates[0].items() if k != "template"},
        "best_val": {k: v for k, v in best.items() if k != "template"},
        "metric_calls": result.total_metric_calls,
        "usage": usage,
        "cost_usd": round(budget.spent(), 3),
        "template_path": str(template_path),
        "candidates": candidates,
        "proposals": reflector.proposals,
    }
    if wandb_run is not None:
        summary["wandb_url"] = log_wandb_optimization(wandb_run, summary, folder)
    (folder / "result.json").write_text(json.dumps(summary, indent=1))
    print(
        f"best: candidate {result.best_idx} of {len(candidates)}, val balanced F1 "
        f"{_f(best['f1_balanced'])} (seed {_f(candidates[0]['f1_balanced'])}); "
        f"cost ${summary['cost_usd']}; wrote {template_path}"
    )
    loop.close()
    return summary


def log_wandb_optimization(run, summary: dict, folder: Path) -> str | None:
    """The candidate table, the best and baseline val metrics in the summary, and
    the best template plus the result as an artifact; then finish the run."""
    import wandb

    rows = [
        [c["index"], str(c["parents"]), c["val_score"], c["f1_balanced"],
         c["rate_met"], len(c["template"]), c["template"]]
        for c in summary["candidates"]
    ]  # fmt: skip
    cols = ["index", "parents", "val_agreement", "val_f1_balanced", "val_rate_met",
            "chars", "template"]  # fmt: skip
    run.log({"candidates": wandb.Table(columns=cols, data=rows)})
    run.summary.update(
        {f"baseline/{k}": v for k, v in summary["baseline_val"].items()}
        | {f"best/{k}": v for k, v in summary["best_val"].items()}
        | {"cost_usd": summary["cost_usd"], "metric_calls": summary["metric_calls"]}
    )
    art = wandb.Artifact(f"grader-template-{summary['config']['name']}", "prompt")
    art.add_file(summary["template_path"])
    (folder / "result.json").write_text(json.dumps(summary, indent=1))
    art.add_file(str(folder / "result.json"))
    run.log_artifact(art)
    url = run.url
    run.finish()
    return url


def compare(
    baseline: dict,
    optimized: dict,
    heldout: Sequence[str],
    root: Path = healthbench.RAW,
) -> dict:
    """Two calibration results on the same test set: balanced F1 on the
    categories the optimizer saw and on the held-out ones."""
    judgments = {
        j.id: j
        for j in sample_judgments(
            healthbench.load_judgments(root), baseline["n"], baseline["seed"]
        )
    }
    out = {}
    for name, r in (("baseline", baseline), ("optimized", optimized)):
        for part in ("seen", "heldout"):
            ids = [
                i
                for i, j in judgments.items()
                if (j.category in heldout) == (part == "heldout")
            ]
            pairs = [
                (r["labels"][i], ref)
                for i in ids
                if r["labels"][i] is not None
                for ref in judgments[i].physician_labels
            ]
            out[f"{name}/{part}"] = pairwise_f1(pairs)["f1_balanced"]
        out[f"{name}/overall"] = r["judge"]["overall"]["f1_balanced"]
        out[f"{name}/rate_met"] = r["judge_rate_met"]
        out[f"{name}/kappa"] = r["retest"]["kappa"]
    return out


# --- CLI -----------------------------------------------------------------------


@click.group(help=__doc__.split("\n\n")[0])
def main() -> None:
    pass


@main.command("splits")
@click.option("--seed", default=0, show_default=True)
def splits_cmd(seed: int) -> None:
    """Sizes and class balance of train / val, and the held-out categories."""
    data = splits(healthbench.load_judgments(), seed=seed)
    for k in ("train", "val"):
        c = Counter(it.majority for it in data[k])
        cats = len({it.judgment.category for it in data[k]})
        print(f"{k}: {len(data[k])} items, {dict(c)}, {cats} categories")
    print("held out:", ", ".join(data["heldout_categories"]))


@main.command("run")
@click.option("--name", default="gepa-v1", show_default=True)
@click.option("--judge", "judge_name", default="haiku", show_default=True)
@click.option("--effort", default="medium", show_default=True)
@click.option("--thinking/--no-thinking", default=True, show_default=True)
@click.option("--reflection", default="sonnet", show_default=True)
@click.option("--reflection-effort", default="medium", show_default=True)
@click.option("--n-train", default=400, show_default=True)
@click.option("--n-val", default=200, show_default=True)
@click.option("--minibatch", default=8, show_default=True)
@click.option("--max-metric-calls", default=3000, show_default=True)
@click.option("--budget-usd", default=8.0, show_default=True, help="stop at this spend")
@click.option("--seed", default=0, show_default=True, help="the data splits")
@click.option("--gepa-seed", type=int, default=None, help="GEPA's (default: --seed)")
@click.option("--concurrency", default=32, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=OUT, show_default=True)
@click.option("--wandb", "wandb_project", default=None, help="log a W&B run here")
def run_cmd(**kwargs) -> None:
    run(**kwargs)


@main.command("compare")
@click.argument("baseline", type=click.Path(exists=True, path_type=Path))
@click.argument("optimized", type=click.Path(exists=True, path_type=Path))
@click.option("--seed", default=0, show_default=True)
def compare_cmd(baseline: Path, optimized: Path, seed: int) -> None:
    """Two calibration result files: balanced F1 on seen vs held-out categories."""
    held = sorted(heldout_categories(healthbench.load_judgments(), seed=seed))
    out = compare(json.loads(baseline.read_text()), json.loads(optimized.read_text()),
                  held)  # fmt: skip
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
