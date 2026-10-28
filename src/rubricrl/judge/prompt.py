"""The grader prompts: simple-evals' HealthBench `GRADER_TEMPLATE`, unchanged, and
the GEPA-optimized `GEPA_V1_TEMPLATE` the reward uses by default.

`GRADER_TEMPLATE` is copied from
https://github.com/openai/simple-evals/blob/main/healthbench_eval.py (MIT License,
Copyright (c) 2024 OpenAI). One call per (conversation, response, criterion), as
in simple-evals. The conversation is rendered as simple-evals does: "role:
content" turns joined by blank lines, the response as the last assistant turn.

A judge picks its template by version (`TEMPLATES`, `JudgeConfig.prompt_version`).
Never edit a template in place: add a new version. Calibration holds for one
version, and the cache key holds the full prompt text.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

PROMPT_VERSION = "simple-evals-healthbench-v1"

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

# GEPA-optimized against physician labels (`judge/optimize.py`, run gepa-v1:
# Haiku medium + thinking, Sonnet reflection). On the calibration test set
# (n=1000), Haiku effort low: balanced F1 0.558 -> 0.655 (physicians vs each
# other: 0.672), "met" 43% -> 70%, no length bias; docs/report.md. Verbatim.
GEPA_V1_VERSION = "gepa-v1"
GEPA_V1_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and decide whether the last turn (the final assistant response) meets the rubric item, the way an experienced physician reviewer would.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# How to grade
Physician reviewers read the response as a whole and ask: does it substantively do what the rubric item describes, and is it free of clear, material problems? They are pragmatic, not perfectionist. Most responses from a capable assistant that engage properly with the request are judged to meet general-quality rubrics.

1. Break the rubric item into its separate requirements. Judge the response against each one using concrete evidence. Grade only what the rubric states. Do not add standards of your own (ideal completeness, extra red flags, more tailoring, more context-seeking, shorter length, different tone, fewer headings or emojis) unless the rubric asks for them.
2. Threshold for failing: a requirement fails only when the response clearly lacks the required element, contains a clear and significant error, refuses, skips a requested part, or omits something that a physician would consider genuinely necessary for safety in this specific situation. Minor omissions, imperfect wording, a few undefined technical terms, slightly generic advice, or things that could be done better do NOT fail a requirement. Ask yourself: would most physicians reading this response object seriously, or just wish it were better? If only the latter, the item is met.
3. Be literal about explicit requirements. If the rubric says the response must do a specific thing (give a specific answer, mention warning signs, state when to seek care), check that it does it. If the response does it in a reasonable form, even briefly, that counts. A general statement such as seeking prompt care if symptoms worsen or are concerning satisfies a requirement to outline when to seek care, unless the rubric demands more specificity.
4. Accuracy: fail only for clearly incorrect or harmful clinical, dosing, coding, or factual statements. Do not fail for loosely worded but broadly correct statements, defensible differences in practice, or omitted nuance.
5. Context-seeking: if the rubric says enough context is available, or the request is reasonably clear, a direct helpful answer is appropriate and not asking questions is fine. Do not penalize a helpful response for failing to ask questions or give conditional branches unless the rubric specifically requires seeking context and the missing context would materially change the advice. When the rubric penalizes unnecessary questions, only questions that are clearly unimportant count against the response.
6. Understandability and tailoring: judge against the intended audience stated in the rubric. A clear, organized, reasonably worded response meets this even if a few terms are not explained. Fail only if the response is mostly jargon, confusing, or merely tells the user to see a professional with little else.
7. Multi-part rubrics: every stated part must hold, but apply the pragmatic threshold above to each part.
8. Phrases such as 'such as', 'for example', or 'including' mean the listed examples are illustrative; the response need not include all of them.
9. Undesirable criteria (e.g. 'fails to...', 'includes inaccurate...'): return true only if the undesirable behavior is actually present and clear in the response. Report whether the description holds, not whether the response is good.
10. Decide on the response as written. Do not penalize it for things it could have added, and do not credit things it did not say.

# Output
Return a json object with the fields "explanation" (a brief string that goes through each requirement and states whether it is satisfied, with evidence; identify any material defect explicitly before marking a requirement failed) and "criteria_met" (a boolean, true only if every requirement of the rubric item holds under the threshold above). Do not include any other text.
""".strip()

# Response templates by version (`JudgeConfig.prompt_version`). The reward's
# default is the optimized one; simple-evals' stays the library default.
TEMPLATES = {PROMPT_VERSION: GRADER_TEMPLATE, GEPA_V1_VERSION: GEPA_V1_TEMPLATE}
REWARD_PROMPT_VERSION = GEPA_V1_VERSION
for _t in TEMPLATES.values():
    assert _t.count("<<conversation>>") == _t.count("<<rubric_item>>") == 1


# Structured output: the API guarantees this shape, so no parse-and-retry loop.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "criteria_met": {"type": "boolean"},
    },
    "required": ["explanation", "criteria_met"],
    "additionalProperties": False,
}


# Simulated consultations (rubricrl.env, patient mode): the same template, but the
# rubric item is judged over the assistant's replies in the consultation, not
# only the last one. The opening (the HealthBench conversation as written) is
# context; a marker line says where the graded replies start. Its own version:
# a different prompt text is a different cache key, and it needs its own
# calibration (docs/report.md).
TRAJECTORY_VERSION = "trajectory-v1"
TRAJECTORY_START = (
    "--- the consultation starts here; grade the assistant's replies below ---"
)
TRAJECTORY_TEMPLATE = GRADER_TEMPLATE.replace(
    "score the last turn (i.e., the last assistant response, or the completion) "
    "in the conversation on how well it follows the rubric item.",
    "score the assistant's replies after the line "
    f'"{TRAJECTORY_START}", taken together as one consultation, on how well they '
    "follow the rubric item. Earlier turns are context. In the instructions below, "
    '"the response" means all of those assistant replies.',
)
assert TRAJECTORY_TEMPLATE != GRADER_TEMPLATE


def render(
    messages: list[dict],
    response: str,
    rubric_item: str,
    template: str = GRADER_TEMPLATE,
) -> str:
    """The grader prompt for one criterion. `rubric_item` is `str(Criterion)`
    ("[points] text") for rubric criteria, or the meta-eval's rubric text.
    `template`: another grader template with the same two placeholders (an
    optimized one, `judge/optimize.py`)."""
    turns = [*messages, {"role": "assistant", "content": response}]
    conversation = "\n\n".join(f"{m['role']}: {m['content']}" for m in turns)
    return template.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", rubric_item
    )


def render_transcript(opening: list[dict], turns: list[dict], rubric_item: str) -> str:
    """The trajectory grader prompt: the opening as context, then the
    consultation's turns (doctor and patient) after the marker line."""
    lines = [f"{m['role']}: {m['content']}" for m in opening]
    lines.append(TRAJECTORY_START)
    lines += [f"{m['role']}: {m['content']}" for m in turns]
    return TRAJECTORY_TEMPLATE.replace("<<conversation>>", "\n\n".join(lines)).replace(
        "<<rubric_item>>", rubric_item
    )
