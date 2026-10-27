"""One schema for HealthBench conversations and physician judgments.

Plain dataclasses that round-trip through dicts (JSONL), so the same records move
between the data CLI, the judge, the Miles reward function and the evals.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Message = dict[str, str]  # {"role": "user" | "assistant" | "system", "content": ...}


@dataclass(frozen=True)
class Criterion:
    """One rubric item. `points` < 0 marks an undesirable behaviour."""

    id: str  # f"{prompt_id}:{index in the source rubric}"
    text: str
    points: int  # as in the source (all ints)
    axis: str | None = None  # from the "axis:..." tag, e.g. "completeness"
    level: str | None = None  # "example" (written for this conversation) or "cluster"
    tags: tuple[str, ...] = ()

    def __str__(self) -> str:
        # The form simple-evals puts in the grader prompt (str(RubricItem)).
        return f"[{self.points}] {self.text}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Criterion:
        return cls(**{**d, "tags": tuple(d.get("tags", ()))})


@dataclass(frozen=True)
class Example:
    """A conversation that ends on a user turn, and the rubric for the reply."""

    id: str  # HealthBench prompt_id
    messages: list[Message]
    rubric: list[Criterion]
    themes: tuple[str, ...] = ()  # "theme:..." example tags, prefix stripped
    tags: tuple[str, ...] = ()  # every example tag, as in the source
    source: str = "oss_eval"
    split: str | None = (
        None  # "train" | "val" | "test" (healthbench.conversation_split)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "messages": self.messages,
            "rubric": [c.to_dict() for c in self.rubric],
            "themes": list(self.themes),
            "tags": list(self.tags),
            "source": self.source,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            id=d["id"],
            messages=d["messages"],
            rubric=[Criterion.from_dict(c) for c in d["rubric"]],
            themes=tuple(d.get("themes", ())),
            tags=tuple(d.get("tags", ())),
            source=d.get("source", "oss_eval"),
            split=d.get("split"),
        )


@dataclass(frozen=True)
class Judgment:
    """oss_meta_eval: physicians' verdicts on whether a completion meets a rubric."""

    id: str  # f"{completion_id}:{category}"
    prompt_id: str
    completion_id: str
    messages: list[Message]
    completion: str
    rubric: str  # a cluster-level rubric, as text (no points)
    category: str  # "cluster:..."
    physician_labels: tuple[bool, ...]
    physician_ids: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["physician_labels"] = list(self.physician_labels)
        d["physician_ids"] = list(self.physician_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Judgment:
        return cls(
            **{
                **d,
                "physician_labels": tuple(d["physician_labels"]),
                "physician_ids": tuple(d.get("physician_ids", ())),
            }
        )
