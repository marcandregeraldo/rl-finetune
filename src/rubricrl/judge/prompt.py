"""The grader prompt: simple-evals' HealthBench `GRADER_TEMPLATE`, unchanged.

Copied from https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
(MIT License, Copyright (c) 2024 OpenAI). One call per (conversation, response,
criterion), as in simple-evals. The conversation is rendered as simple-evals does:
"role: content" turns joined by blank lines, the response as the last assistant
turn.

Change anything here (template or rendering) => bump PROMPT_VERSION: it is part
of the judge cache key, and judge calibration only holds for one version.

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


def render(messages: list[dict], response: str, rubric_item: str) -> str:
    """The grader prompt for one criterion. `rubric_item` is `str(Criterion)`
    ("[points] text") for rubric criteria, or the meta-eval's rubric text."""
    turns = [*messages, {"role": "assistant", "content": response}]
    conversation = "\n\n".join(f"{m['role']}: {m['content']}" for m in turns)
    return GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", rubric_item
    )
