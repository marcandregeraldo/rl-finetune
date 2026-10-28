"""Does the simulated environment reward what the physicians would? Two checks.

There are no physician labels on simulated consultations, so these are the
evidence that the mapping and the simulator behave:

- **fidelity**: on HealthBench's own single-turn transcripts (the physicians'
  ideal completions), each mapped criterion is judged next to the physicians'
  original. A faithful rewrite gives the same verdict on the same answer; a
  disagreement points at a rewrite that changed the meaning (or flipped its
  polarity, which scenario-v1 did).
- **doctors**: scripted doctors play full consultations against the simulated
  patient. A careful doctor (asks before advising, Sonnet) should score clearly
  above a minimal one (the same generic sentence every turn, no LLM), and draw out
  more of the hidden facts. If it doesn't, the environment doesn't reward what it
  should. Also measured: how often the patient reveals a hidden fact in answer to
  a doctor message that asked nothing (the simulator leaking).

    python -m rubricrl.sim.checks fidelity --split test --n 50
    python -m rubricrl.sim.checks doctors --split test --n 20 --wandb rl-finetune
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import click

from rubricrl.data import healthbench
from rubricrl.judge.client import (
    ClaudeJudge,
    JudgeConfig,
    Usage,
    VerdictCache,
    api_key,
)
from rubricrl.judge.prompt import render, render_transcript
from rubricrl.sim.scenario import SCENARIO_DIR, Scenario
from rubricrl.sim.scenarios import load

OUT = Path("analysis/out/sim")

DOCTORS = {
    "careful": (
        "sonnet",
        "You are a careful, warm physician talking with a patient by chat. Before "
        "giving advice, ask the one or two questions that would most change it "
        "(one short message at a time). When you have what you need, give clear, "
        "safe, specific advice, including when to seek urgent care. Keep each "
        "message under 150 words.",
    ),
}
# A scripted lower bound. An LLM told to be terse and never ask (Haiku) ignored
# it: it asked for details and wrote full answers.
MINIMAL_REPLY = "I'd suggest talking to a healthcare provider about this."


async def minimal_doctor(messages: list[dict]) -> str:
    return MINIMAL_REPLY


def ideal_completions(root: Path = healthbench.RAW) -> dict[str, str]:
    rows = healthbench._read_jsonl(root / healthbench.SOURCES["oss_eval"].file)
    return {
        r["prompt_id"]: r["ideal_completions_data"]["ideal_completion"]
        for r in rows
        if (r.get("ideal_completions_data") or {}).get("ideal_completion")
    }


async def fidelity(scenarios: list[Scenario], ideals: dict[str, str], judge) -> dict:
    """Original vs mapped criterion, same answer: the verdicts should agree."""
    examples = {e.id: e for e in healthbench.load_examples()}
    pairs = []
    for s in scenarios:
        if s.id not in ideals:
            continue
        source = {c.id: c for c in examples[s.id].rubric}
        answer = ideals[s.id]
        seen = set()
        for c in s.train_criteria + s.heldout_criteria:
            if c.id in seen or "status:rewritten" not in c.tags:
                continue
            seen.add(c.id)
            orig = source[next(t[7:] for t in c.tags if t.startswith("source:"))]
            pairs.append((s, orig, c, answer))

    async def one(s, orig, new, answer):
        a = await judge.grade_prompt(render(s.opening, answer, str(orig)))
        b = await judge.grade_prompt(
            render_transcript(
                s.opening, [{"role": "assistant", "content": answer}], str(new)
            )
        )
        return {
            "scenario": s.id,
            "theme": s.theme,
            "points": orig.points,
            "original": orig.text,
            "mapped": new.text,
            "original_met": a.met,
            "mapped_met": b.met,
        }

    rows = await asyncio.gather(*(one(*p) for p in pairs))
    valid = [
        r for r in rows if r["original_met"] is not None and r["mapped_met"] is not None
    ]
    agree = [r["original_met"] == r["mapped_met"] for r in valid]
    by_sign = {
        sign: statistics.fmean(
            r["original_met"] == r["mapped_met"]
            for r in valid
            if (r["points"] > 0) == (sign == "positive")
        )
        for sign in ("positive", "negative")
        if any((r["points"] > 0) == (sign == "positive") for r in valid)
    }
    by_theme = {
        t: statistics.fmean(
            r["original_met"] == r["mapped_met"] for r in valid if r["theme"] == t
        )
        for t in sorted({r["theme"] for r in valid})
    }
    return {
        "pairs": len(valid),
        "scenarios": len({r["scenario"] for r in valid}),
        "agreement": statistics.fmean(agree) if agree else None,
        "agreement_by_sign": by_sign,
        "agreement_by_theme": by_theme,
        "disagreements": [r for r in valid if r["original_met"] != r["mapped_met"]],
    }


class Doctor:
    """A scripted doctor: Claude with a style prompt, replying to the transcript.
    Replies are cached (`llm.JSONCache`), so a rerun of a check is free."""

    def __init__(self, model: str, system: str, client=None, cache=None):
        self.config = JudgeConfig.named(model, max_tokens=1024)
        self.system = system
        self._client = client
        self.cache = cache
        self.usage = Usage()

    async def __call__(self, messages: list[dict]) -> str:
        key = hashlib.sha256(
            json.dumps([self.config.model, self.system, messages]).encode()
        ).hexdigest()
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return hit["reply"]
        reply = await self._call(messages)
        if self.cache is not None:
            self.cache.put(key, self.config.model, {"reply": reply}, (0, 0))
        return reply

    async def _call(self, messages: list[dict]) -> str:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=api_key(), max_retries=6)
        msg = await self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.system,
            messages=messages,
            thinking=self.config.thinking_param(),
            output_config={"effort": "low"},
        )
        self.usage.calls += 1
        self.usage.input_tokens += msg.usage.input_tokens
        self.usage.output_tokens += msg.usage.output_tokens
        return "".join(b.text for b in msg.content if b.type == "text").strip()


async def play(env, doctor, scenario_id: str, half: str = "train") -> dict:
    """One consultation in-process; the environment's final observation."""
    from rubricrl.env.models import DoctorAction

    obs = await env.reset_async(scenario_id=scenario_id, mode="patient", half=half)
    while not obs.done:
        reply = await doctor(_alternating(obs.messages))
        obs = await env.step_async(DoctorAction(message=reply or "(no reply)"))
    return obs.model_dump()


def _alternating(messages: list[dict]) -> list[dict]:
    """The API wants user/assistant turns alternating, starting with the user:
    merge consecutive same-role turns (an opening may hold several)."""
    out: list[dict] = []
    for m in messages:
        if m["role"] not in ("user", "assistant"):
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {
                "role": m["role"],
                "content": out[-1]["content"] + "\n\n" + m["content"],
            }
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class _Fresh:
    """One environment per episode, as the server makes one per session."""

    def __init__(self, make_env):
        self.make_env = make_env

    async def reset_async(self, **kw):
        self.env = self.make_env()
        return await self.env.reset_async(**kw)

    async def step_async(self, action):
        return await self.env.step_async(action)


async def doctors(scenarios: list[Scenario], make_env, styles=DOCTORS) -> dict:
    """Every style on every scenario; reward, facts drawn out, turns.

    `make_env()` -> a fresh `ConsultEnvironment`; `styles`: name -> a doctor
    (`Doctor`, or any async messages -> reply).
    """
    results = {}
    facts = {s.id: len(s.hidden_facts) for s in scenarios}
    for style, doctor in styles.items():
        episodes = await asyncio.gather(
            *(play(_Fresh(make_env), doctor, s.id) for s in scenarios)
        )
        with_facts = [e for e in episodes if facts[e["scenario_id"]]]
        reveals = [r for e in episodes for r in _reveals(e)]
        results[style] = {
            "reward_mean": statistics.fmean(e["reward"] for e in episodes),
            "facts_elicited": statistics.fmean(
                len(e["revealed_facts"]) / facts[e["scenario_id"]] for e in with_facts
            )
            if with_facts
            else None,
            "turns_mean": statistics.fmean(e["turn"] for e in episodes),
            "end_reasons": dict(Counter(e["end_reason"] for e in episodes)),
            "failed_verdicts": sum(e["failed"] for e in episodes),
            "reveals": len(reveals),
            "unprompted_reveal_rate": (
                statistics.fmean(not asked for asked in reveals) if reveals else None
            ),
            "episodes": episodes,
        }
    return results


def _reveals(episode: dict) -> list[bool]:
    """For each hidden fact the patient revealed: did the doctor's message just
    before it ask anything? (Proxy: it contains a question mark.)"""
    msgs = episode["messages"]
    return [
        "?" in msgs[r["index"] - 1]["content"]
        for r in episode.get("reveals", [])
        for _ in r["facts"]
    ]


@click.group(help=__doc__.split("\n\n")[0])
def main() -> None:
    pass


def _scenarios(split: str, n: int, scenario_dir: Path) -> list[Scenario]:
    return load(scenario_dir / f"{split}.jsonl")[:n]


@main.command("fidelity")
@click.option("--split", default="test", show_default=True)
@click.option("--n", default=50, show_default=True, help="scenarios")
@click.option("--scenario-dir", type=click.Path(path_type=Path), default=SCENARIO_DIR)
@click.option("--wandb", "wandb_project", default=None)
def fidelity_cmd(split, n, scenario_dir, wandb_project) -> None:
    judge = ClaudeJudge(JudgeConfig(), VerdictCache())
    out = asyncio.run(
        fidelity(_scenarios(split, n, scenario_dir), ideal_completions(), judge)
    )
    out["cost_usd"] = round(judge.usage.cost(judge.config.model), 3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fidelity-{split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))
    print(f"{len(out['disagreements'])} disagreements -> {OUT}/fidelity-{split}.json")
    if wandb_project:
        from rubricrl import observe

        observe.log_wandb_check(wandb_project, "fidelity", out)


@main.command("doctors")
@click.option("--split", default="test", show_default=True)
@click.option("--n", default=20, show_default=True, help="scenarios")
@click.option("--scenario-dir", type=click.Path(path_type=Path), default=SCENARIO_DIR)
@click.option("--wandb", "wandb_project", default=None, help="also traces to Weave")
def doctors_cmd(split, n, scenario_dir, wandb_project) -> None:
    from rubricrl.env import server
    from rubricrl.sim.scenario import ScenarioStore

    store = ScenarioStore.load(scenario_dir.parent, scenario_dir)
    tracer = None
    if wandb_project:
        from rubricrl import observe

        tracer = observe.episode_tracer(wandb_project, rate=1.0)
    resources = server.Resources(store=store, tracer=tracer)
    from rubricrl.llm import JSONCache

    styles = {name: Doctor(*spec, cache=JSONCache()) for name, spec in DOCTORS.items()}
    styles["minimal"] = minimal_doctor

    async def run_all() -> dict:
        out = await doctors(
            _scenarios(split, n, scenario_dir),
            lambda: server.ConsultEnvironment(resources),
            styles,
        )
        judge, patient = resources.judge, resources.patient.llm  # this loop's
        out["cost_usd"] = {
            "judge": round(judge.usage.cost(judge.config.model), 3),
            "patient": round(patient.usage.cost(patient.config.model), 3),
            **{
                f"doctor_{k}": round(d.usage.cost(d.config.model), 3)
                for k, d in styles.items()
                if isinstance(d, Doctor)
            },
        }
        return out

    out = asyncio.run(run_all())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"doctors-{split}.json").write_text(json.dumps(out, indent=1))
    summary = {
        k: {m: v for m, v in d.items() if m != "episodes"} if isinstance(d, dict) else d
        for k, d in out.items()
    }
    print(json.dumps(summary, indent=1))
    if wandb_project:
        from rubricrl import observe

        observe.log_wandb_check(wandb_project, "doctors", out)


if __name__ == "__main__":
    main()
