"""The simulated patient: continues a HealthBench conversation as the user.

It knows a persona and some hidden facts (`sim/scenarios.py` derives both from
the physicians' conversation and rubric), and reveals a fact only when the
doctor asks for it: that is what makes seeking context rewardable. It replies
briefly, as a lay person, and ends the consultation when its end condition is
met. Its output is structured (`{reply, revealed_fact_ids, end}`), so the
episode records which facts the doctor drew out.

Haiku 5.5 by default (`RUBRICRL_PATIENT`); a different model at eval time is a
held-out simulator, a third reward-hacking detector next to the held-out
criteria and the held-out judge.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

from rubricrl.sim.scenario import Scenario

# v2: stricter about volunteering hidden facts (v1, on Haiku, added some unasked;
# `sim.checks doctors` measures it as the unprompted-reveal rate).
PATIENT_VERSION = "patient-v2"
END = "[END]"

PATIENT_TEMPLATE = """
You are role-playing a person talking to a health assistant. Stay in character for the whole conversation.

# Who you are
<<patient>>

# What you know but only say if asked
These facts are true for you. Reveal a fact only when the assistant's last message asks about it, directly or in substance (for example, "any other medicines?" asks about medications). Never volunteer them, never list them, never reveal one that was not asked about, and do not add them as extra details or follow-up requests even if they seem relevant. If the assistant's last message asks no question, reveal no hidden fact.
<<facts>>

# When you end the conversation
<<end_condition>>
Also end it if the assistant has clearly finished and has nothing more to ask.

# Rules
- Write the next message of the user, in the user's own voice: plain language, like a lay person, usually 1 to 4 sentences.
- Answer what the assistant asked. If it asks several questions, answer them.
- Do not diagnose yourself and do not use medical knowledge the person would not have.
- Do not say you are an AI or a simulation.
- Set "end" to true when you end the conversation; "reply" is then your last message (for example a thank-you).
- "revealed_fact_ids" lists the ids of the hidden facts this message reveals (empty if none). Every one of them must answer a question in the assistant's last message.

# The conversation so far
<<conversation>>

Write the user's next message.
""".strip()

PATIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "revealed_fact_ids": {"type": "array", "items": {"type": "string"}},
        "end": {"type": "boolean"},
    },
    "required": ["reply", "revealed_fact_ids", "end"],
    "additionalProperties": False,
}


@dataclass
class PatientTurn:
    reply: str
    revealed: list[str]
    end: bool


def render(scenario: Scenario, transcript: list[dict]) -> str:
    facts = "\n".join(f"- {f.id}: {f.text}" for f in scenario.hidden_facts) or "(none)"
    conversation = "\n\n".join(f"{m['role']}: {m['content']}" for m in transcript)
    return (
        PATIENT_TEMPLATE.replace("<<patient>>", scenario.patient)
        .replace("<<facts>>", facts)
        .replace("<<end_condition>>", scenario.end_condition)
        .replace("<<conversation>>", conversation)
    )


class PatientSimulator:
    """`await patient(scenario, transcript)` -> the patient's next turn.

    `llm` is a `rubricrl.llm.StructuredLLM` (or `FakeLLM` in tests). Outputs are
    cached on the full transcript, so replaying an episode is free and the same.
    """

    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, scenario: Scenario, transcript: list[dict]) -> PatientTurn:
        out = await self.llm(
            render(scenario, transcript), PATIENT_SCHEMA, PATIENT_VERSION
        )
        known = {f.id for f in scenario.hidden_facts}
        reply = out["reply"].replace(END, "").strip()
        revealed = [f for f in out["revealed_fact_ids"] if f in known]
        return PatientTurn(reply, revealed, bool(out["end"]))
