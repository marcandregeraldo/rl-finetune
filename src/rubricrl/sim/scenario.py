"""A consultation scenario: what the environment (`rubricrl.env`) plays out.

One schema for both modes, so v1 is v2 cut at one turn:

- **single** (v1): a HealthBench conversation as written; the doctor writes the
  next reply, graded on the physicians' criteria. Built from the Miles exports
  (`data/miles/healthbench_*.jsonl`), so the reward is exactly the one Miles
  trains on (`rewards.miles_rm`).
- **patient** (v2): the same conversation is only the opening; a simulated
  patient (`sim/patient.py`) continues it, with facts it reveals only when asked,
  and the whole consultation is graded on trajectory-level criteria mapped from
  the physicians' ones (`sim/scenarios.py`). Built into
  `data/miles/scenarios/*.jsonl`, next to the Miles exports, so one directory
  holds everything the environment serves.

Criteria keep their provenance in tags: `source:<physician criterion id>` and
`status:kept|rewritten`.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rubricrl.data.schema import Criterion, Message

MILES_DIR = Path("data/miles")
SCENARIO_DIR = MILES_DIR / "scenarios"
MODES = ("single", "patient")


@dataclass(frozen=True)
class Fact:
    id: str  # "f1", "f2", ...
    text: str


@dataclass(frozen=True)
class Scenario:
    id: str  # the HealthBench prompt_id it comes from
    mode: str  # "single" | "patient"
    opening: list[Message]  # the HealthBench conversation, verbatim
    train_criteria: list[Criterion]  # the reward
    heldout_criteria: list[Criterion]  # eval only
    split: str = ""
    theme: str = ""
    patient: str = ""  # persona (patient mode)
    hidden_facts: tuple[Fact, ...] = ()
    end_condition: str = ""
    mapping: tuple[dict, ...] = ()  # per physician criterion: status, reason
    version: str = ""  # the scenario generator's version (patient mode)

    def criteria(self, half: str) -> list[Criterion]:
        return self.heldout_criteria if half == "eval" else self.train_criteria

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "opening": self.opening,
            "train_criteria": [c.to_dict() for c in self.train_criteria],
            "heldout_criteria": [c.to_dict() for c in self.heldout_criteria],
            "split": self.split,
            "theme": self.theme,
            "patient": self.patient,
            "hidden_facts": [{"id": f.id, "text": f.text} for f in self.hidden_facts],
            "end_condition": self.end_condition,
            "mapping": list(self.mapping),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        return cls(
            id=d["id"],
            mode=d["mode"],
            opening=d["opening"],
            train_criteria=[Criterion.from_dict(c) for c in d["train_criteria"]],
            heldout_criteria=[Criterion.from_dict(c) for c in d["heldout_criteria"]],
            split=d.get("split", ""),
            theme=d.get("theme", ""),
            patient=d.get("patient", ""),
            hidden_facts=tuple(Fact(**f) for f in d.get("hidden_facts", ())),
            end_condition=d.get("end_condition", ""),
            mapping=tuple(d.get("mapping", ())),
            version=d.get("version", ""),
        )

    @classmethod
    def from_miles_row(cls, row: dict[str, Any], split: str = "") -> Scenario:
        """A single-turn scenario from a Miles export row (its metadata)."""
        md = row["metadata"]
        return cls(
            id=md["prompt_id"],
            mode="single",
            opening=md["messages"],
            train_criteria=[Criterion.from_dict(c) for c in md["train_criteria"]],
            heldout_criteria=[Criterion.from_dict(c) for c in md["heldout_criteria"]],
            split=split,
            theme=(md.get("themes") or [""])[0],
        )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class ScenarioStore:
    """Every scenario the environment can reset to, by (mode, id)."""

    scenarios: dict[tuple[str, str], Scenario] = field(default_factory=dict)

    @classmethod
    def load(
        cls, miles_dir: Path = MILES_DIR, scenario_dir: Path | None = None
    ) -> ScenarioStore:
        """Single-turn scenarios from the Miles exports in `miles_dir`; patient
        ones from `scenario_dir` (default `miles_dir/scenarios`)."""
        scenario_dir = scenario_dir or miles_dir / "scenarios"
        store = cls()
        for split in ("train", "val", "test"):
            path = miles_dir / f"healthbench_{split}.jsonl"
            if path.exists():
                for row in _read_jsonl(path):
                    store.add(Scenario.from_miles_row(row, split))
            path = scenario_dir / f"{split}.jsonl"
            if path.exists():
                for row in _read_jsonl(path):
                    store.add(Scenario.from_dict(row))
        return store

    def add(self, s: Scenario) -> None:
        self.scenarios[(s.mode, s.id)] = s

    def get(self, mode: str, scenario_id: str) -> Scenario:
        try:
            return self.scenarios[(mode, scenario_id)]
        except KeyError:
            raise KeyError(f"no {mode} scenario {scenario_id!r}") from None

    def ids(self, mode: str, split: str | None = None) -> list[str]:
        return sorted(
            sid
            for (m, sid), s in self.scenarios.items()
            if m == mode and (split is None or s.split == split)
        )

    def __len__(self) -> int:
        return len(self.scenarios)
