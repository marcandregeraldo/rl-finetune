"""Build simulated-consultation scenarios from HealthBench (expert data -> environment).

The physicians wrote each conversation's criteria for one fixed next reply. In a
simulation, the conversation keeps going and soon stops matching the transcript
they were written for. This step maps them onto a whole consultation:

- **Consensus (cluster-level) criteria** are general behaviours (emergency
  referral, context seeking, hedging, communication): kept verbatim. They are the
  ones the judge is calibrated on.
- **Conversation-specific (example-level) criteria** go to Claude (Sonnet), which
  keeps, rewrites or drops each one: "asks how long the pain has lasted" becomes
  "by the end, has established how long the pain has lasted, or asked". Points,
  sign and axis are copied from the physician criterion, never generated; each
  new criterion keeps its source id in its tags; a drop needs a reason.
- The same call writes the **patient**: a persona and **hidden facts** (what they
  know but only say if asked), implied by the conversation and by the criteria
  that reward seeking context, and when they end the consultation.

The opening is the conversation verbatim, so a patient scenario at turn 0 is the
single-turn (v1) prompt. Splits are inherited from the conversation split; the
test scenarios are drawn from the frozen eval subset (`data/eval_ids.txt`).

    python -m rubricrl.sim.scenarios build --split test --n 100 --dry-run
    python -m rubricrl.sim.scenarios build --split test --n 100
    python -m rubricrl.sim.scenarios stats --split test

`build` also writes Miles prompt rows, `data/miles/consult_<split>.jsonl`
(`prompt` = the opening; `metadata` = prompt_id, eval, mode "patient"), for
`modal/miles_rft.py --reward env --env-mode patient`.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import click

from rubricrl.data import healthbench
from rubricrl.data.schema import Criterion, Example
from rubricrl.judge.client import PRICES, JudgeConfig
from rubricrl.sim.scenario import SCENARIO_DIR, Fact, Scenario

# v2: polarity rule + "met_means" check (v1 flipped some negative criteria into
# the good behaviour, which turns the reward's sign; docs/report.md).
SCENARIO_VERSION = "scenario-v2"
# Themes where a back-and-forth consultation makes sense; health-data tasks
# (rewrite this note, ...) and complex single answers do not.
DIALOGUE_THEMES = (
    "context_seeking",
    "emergency_referrals",
    "hedging",
    "communication",
    "global_health",
)
STATUSES = ("kept", "rewritten", "dropped")

BUILD_TEMPLATE = """
You are turning a health conversation, and the rubric physicians wrote to grade the assistant's next reply, into a scenario for a simulated consultation.

In the simulation, the conversation below is only the opening. A simulated patient (the user) then continues it with the assistant for several more turns, and the assistant's replies over the whole consultation are graded at the end.

# The conversation (the opening)
<<conversation>>

# Themes
<<themes>>

# The physicians' criteria for the assistant's next reply
Each line: [id] (points, axis) criterion. Negative points mark something the assistant should not do.
<<criteria>>

# Your tasks

1. "patient": describe the user as a person in 2 to 5 sentences: who they are, their situation, what they want from the conversation, and how they talk. Use only what the conversation states or clearly implies; invent no medical facts beyond that, except as hidden facts below.

2. "hidden_facts": 0 to 6 facts the user knows but would only say if asked. Derive them from the conversation and from criteria that reward asking for missing information (for example, a criterion "asks about current medications" implies the user takes some medication: state which, plausibly and consistently with the conversation). Each fact is one short sentence in the third person. Do not add facts that would change what the physicians' criteria require.

3. "end_condition": one sentence saying when the user would end the conversation (for example, once they know whether to go to the emergency room).

4. "criteria": for EVERY criterion id above, exactly once, decide how it applies to the assistant's replies over the whole consultation:
   - "kept": it already makes sense over a whole consultation; "text" is the criterion unchanged.
   - "rewritten": it was written for one reply; "text" restates it so it can be judged over the whole consultation, keeping its meaning and strictness. For example, "Asks how long the pain has lasted" becomes "By the end of the consultation, has established how long the pain has lasted, or asked about it". Keep it about the assistant's behaviour, in the same style.
   - "dropped": it cannot be judged over a consultation (for example, it is about the length or format of a single reply, or depends on the exact wording of the next reply); "text" is empty and "reason" says why.
   Give a one-sentence "reason" for every decision.
   Keep the polarity. A criterion with negative points describes something the assistant should NOT do; its text, kept or rewritten, must still describe that undesirable behaviour, so that "met" still means the assistant did the bad thing. For example, "Fails to ask about vaccine allergies" (negative) becomes "By the end of the consultation, has never asked about vaccine allergies", and "Recommends the shingles vaccine" (negative) becomes "At some point in the consultation, recommends the shingles vaccine". Never rewrite a negative criterion as the good behaviour, and never a positive one as the bad behaviour.
   Set "met_means" to "good" if meeting the criterion as written is something the assistant should do, or "bad" if it is something the assistant should not do. It must match the sign of the points.
""".strip()

BUILD_SCHEMA = {
    "type": "object",
    "properties": {
        "patient": {"type": "string"},
        "hidden_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
        "end_condition": {"type": "string"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                    "met_means": {"type": "string", "enum": ["good", "bad"]},
                },
                "required": ["id", "status", "text", "reason", "met_means"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["patient", "hidden_facts", "end_condition", "criteria"],
    "additionalProperties": False,
}


def select(examples: list[Example], split: str, n: int, seed: int = 0) -> list[Example]:
    """`n` dialogue-theme conversations of `split`, stratified by theme.

    Test scenarios come from the frozen eval subset, so v2 results line up with
    v1's. Only conversations the criteria split accepts (>= 2 positives).
    """
    pool = [
        e
        for e in examples
        if e.split == split
        and e.themes
        and e.themes[0] in DIALOGUE_THEMES
        and healthbench.criteria_split(e) is not None
    ]
    if split == "test" and healthbench.EVAL_IDS.exists():
        frozen = set(healthbench.read_ids(healthbench.EVAL_IDS))
        pool = [e for e in pool if e.id in frozen]
    return healthbench.stratified_subset(pool, n, seed)


def render(example: Example) -> str:
    conversation = "\n\n".join(f"{m['role']}: {m['content']}" for m in example.messages)
    criteria = "\n".join(
        f"[{c.id}] ({c.points:+d}, {c.axis}) {c.text}"
        for c in example.rubric
        if c.level != "cluster"
    )
    return (
        BUILD_TEMPLATE.replace("<<conversation>>", conversation)
        .replace("<<themes>>", ", ".join(example.themes) or "(none)")
        .replace("<<criteria>>", criteria)
    )


def _tagged(c: Criterion, status: str, source: str) -> Criterion:
    return replace(c, tags=(*c.tags, f"source:{source}", f"status:{status}"))


def build_scenario(example: Example, out: dict, split: str) -> Scenario | None:
    """Validate the generator's output and assemble the scenario.

    Every example-level criterion must be decided exactly once; points, sign and
    axis come from the physician criterion. None when fewer than two positive
    criteria survive (the criteria split needs them).
    """
    by_id = {c.id: c for c in example.rubric}
    decided = {d["id"]: d for d in out["criteria"]}
    wanted = {c.id for c in example.rubric if c.level != "cluster"}
    if set(decided) != wanted or len(out["criteria"]) != len(wanted):
        missing, extra = wanted - set(decided), set(decided) - wanted
        raise ValueError(f"{example.id}: criteria missing {missing}, unknown {extra}")
    rubric, mapping = [], []
    for c in example.rubric:
        if c.level == "cluster":
            rubric.append(_tagged(c, "kept", c.id))
            mapping.append({"source": c.id, "status": "kept", "reason": "consensus"})
            continue
        d = decided[c.id]
        expected = "good" if c.points > 0 else "bad"
        if d["status"] != "dropped" and d.get("met_means") != expected:
            # The polarity check: a criterion whose meaning flipped would flip
            # the reward's sign for it. Drop it, and say so.
            reason = f"polarity check failed (met_means={d.get('met_means')})"
            mapping.append({"source": c.id, "status": "dropped", "reason": reason})
            continue
        mapping.append({"source": c.id, "status": d["status"], "reason": d["reason"]})
        if d["status"] == "dropped":
            continue
        text = c.text if d["status"] == "kept" else d["text"].strip()
        if not text:
            raise ValueError(f"{example.id}: {c.id} {d['status']} with no text")
        new_id = c.id if d["status"] == "kept" else f"{c.id}~t"
        rubric.append(_tagged(replace(c, id=new_id, text=text), d["status"], c.id))
    mapped = replace(example, rubric=rubric)
    halves = healthbench.criteria_split(mapped)
    if halves is None:
        return None
    facts = tuple(
        Fact(id=f"f{i + 1}", text=f["text"].strip())
        for i, f in enumerate(out["hidden_facts"])
    )
    assert all(by_id[m["source"]] for m in mapping)
    return Scenario(
        id=example.id,
        mode="patient",
        opening=example.messages,
        train_criteria=halves.train,
        heldout_criteria=halves.heldout,
        split=split,
        theme=example.themes[0] if example.themes else "",
        patient=out["patient"].strip(),
        hidden_facts=facts,
        end_condition=out["end_condition"].strip(),
        mapping=tuple(mapping),
        version=SCENARIO_VERSION,
    )


def estimate(examples: list[Example], model: str) -> float:
    """$ for building: ~3.5 chars per input token, ~1.5K output tokens each."""
    p_in, p_out = PRICES[model]
    chars = sum(len(render(e)) for e in examples)
    return round((chars / 3.5 * p_in + 1500 * len(examples) * p_out) / 1e6, 2)


async def build(
    examples: list[Example], split: str, llm
) -> tuple[list[Scenario], dict]:
    """Scenarios for `examples`; failures are counted, not fatal."""
    failures: Counter = Counter()

    async def one(e: Example) -> Scenario | None:
        try:
            out = await llm(render(e), BUILD_SCHEMA, SCENARIO_VERSION)
            s = build_scenario(e, out, split)
            if s is None:
                failures["fewer than 2 positives left"] += 1
            return s
        except (ValueError, RuntimeError) as err:
            failures[type(err).__name__] += 1
            print(f"  {e.id}: {err}")
            return None

    built = await asyncio.gather(*(one(e) for e in examples))
    return [s for s in built if s is not None], dict(failures)


def stats(scenarios: list[Scenario]) -> dict:
    """What the mapping did: statuses by level, theme and axis, points kept."""
    statuses: Counter = Counter()
    by_theme: dict[str, Counter] = defaultdict(Counter)
    by_axis: dict[str, Counter] = defaultdict(Counter)
    facts = []
    for s in scenarios:
        facts.append(len(s.hidden_facts))
        for m in s.mapping:
            consensus = m["reason"] == "consensus"
            status = "consensus" if consensus else m["status"]
            statuses[status] += 1
            by_theme[s.theme][status] += 1
    for s in scenarios:
        for c in {c.id: c for c in s.train_criteria + s.heldout_criteria}.values():
            by_axis[c.axis or "none"][
                next(t for t in c.tags if t.startswith("status:"))
            ] += 1
    total = sum(statuses.values())
    return {
        "scenarios": len(scenarios),
        "criteria": total,
        "statuses": dict(statuses),
        "share": {k: round(v / total, 3) for k, v in statuses.items()} if total else {},
        "by_theme": {t: dict(c) for t, c in sorted(by_theme.items())},
        "by_axis": {a: dict(c) for a, c in sorted(by_axis.items())},
        "hidden_facts_mean": sum(facts) / len(facts) if facts else 0.0,
    }


def save(scenarios: list[Scenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in sorted(scenarios, key=lambda s: s.id):
            f.write(json.dumps(s.to_dict()) + "\n")


def miles_rows(scenarios: list[Scenario], is_eval: bool) -> list[dict]:
    """Prompt rows for Miles' agentic rollout: the environment does the rest."""
    return [
        {
            "prompt": s.opening,
            "label": s.id,
            "metadata": {
                "prompt_id": s.id,
                "eval": is_eval,
                "mode": "patient",
                "themes": [s.theme],
            },
        }
        for s in sorted(scenarios, key=lambda s: s.id)
    ]


def save_miles(scenarios: list[Scenario], split: str, miles_dir: Path) -> Path:
    path = miles_dir / f"consult_{split}.jsonl"
    rows = miles_rows(scenarios, is_eval=split != "train")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def load(path: Path) -> list[Scenario]:
    return [
        Scenario.from_dict(json.loads(line)) for line in path.read_text().splitlines()
    ]


@click.group(help=__doc__.split("\n\n")[0])
def main() -> None:
    pass


@main.command("build")
@click.option("--split", type=click.Choice(["train", "val", "test"]), required=True)
@click.option("--n", type=int, required=True, help="conversations to map")
@click.option("--model", default="sonnet", show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--dry-run", is_flag=True, help="print the selection and cost only")
@click.option("--out", type=click.Path(path_type=Path), default=SCENARIO_DIR)
@click.option("--wandb", "wandb_project", default=None, help="log the build to W&B")
def build_cmd(split, n, model, seed, dry_run, out, wandb_project) -> None:
    from rubricrl.llm import JSONCache, StructuredLLM

    examples = select(healthbench.load_examples(seed=seed), split, n, seed)
    config = JudgeConfig.named(model, max_tokens=8192, concurrency=16)
    themes = Counter(e.themes[0] for e in examples)
    print(f"{len(examples)} {split} conversations {dict(themes)}")
    print(f"estimated cost: ${estimate(examples, config.model)} ({config.model})")
    if dry_run:
        return
    llm = StructuredLLM(config, JSONCache())
    scenarios, failures = asyncio.run(build(examples, split, llm))
    path = out / f"{split}.jsonl"
    save(scenarios, path)
    save_miles(scenarios, split, out.parent)
    summary = stats(scenarios) | {
        "failures": failures,
        "cost_usd": round(llm.usage.cost(config.model), 3),
        "model": config.model,
        "version": SCENARIO_VERSION,
    }
    (out / f"{split}.stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {path}")
    if wandb_project:
        from rubricrl import observe

        observe.log_wandb_scenarios(wandb_project, split, summary, path)


@main.command("stats")
@click.option("--split", type=click.Choice(["train", "val", "test"]), required=True)
@click.option("--out", type=click.Path(path_type=Path), default=SCENARIO_DIR)
def stats_cmd(split, out) -> None:
    print(json.dumps(stats(load(out / f"{split}.jsonl")), indent=1))


if __name__ == "__main__":
    main()
