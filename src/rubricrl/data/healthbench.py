"""HealthBench: download, load, split, and export for Miles.

The files are simple-evals' (MIT, OpenAI), pinned by sha256. Splits are seeded
hashes of the conversation id, so they don't depend on file order:

- **Conversations:** train / val / test (75 / 5 / 20 %). Train feeds the RL run,
  val the in-loop eval, test the fixed eval subset (`data/eval_ids.txt`).
- **Criteria, per conversation:** the positive criteria are split in two disjoint
  halves, *train* (the reward) and *held-out* (eval only). Negative criteria go in
  both halves, so the reward never stops penalising harmful answers. A conversation
  needs at least two positive criteria to be split; the 50 with one are skipped.

    python -m rubricrl.data.healthbench download       # -> data/raw/
    python -m rubricrl.data.healthbench stats          # Markdown, for data/README.md
    python -m rubricrl.data.healthbench export-miles   # -> data/miles/*.jsonl
"""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import click

from rubricrl.data.schema import Criterion, Example, Judgment

BASE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench"
RAW = Path("data/raw")
MILES_OUT = Path("data/miles")
EVAL_IDS = Path("data/eval_ids.txt")


@dataclass(frozen=True)
class Source:
    file: str
    sha256: str


SOURCES = {
    "oss_eval": Source(
        "2025-05-07-06-14-12_oss_eval.jsonl",
        "e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f",
    ),
    "hard": Source(
        "hard_2025-05-08-21-00-10.jsonl",
        "b0320430e5cd974e746585594c1dd10b5a3fc2aff9c72b26106c2c4a069d74e9",
    ),
    "consensus": Source(
        "consensus_2025-05-09-20-00-46.jsonl",
        "fabe37553327238928b05357d8b309737b5d426d6dee432a3376da618d0969d6",
    ),
    "meta_eval": Source(
        "2025-05-07-06-14-12_oss_meta_eval.jsonl",
        "6a343ed3fcb6f28353c2c6c157ae30ddf2db874c947b27ba474ad54da9c5dbac",
    ),
}
SPLITS = (("train", 0.75), ("val", 0.05), ("test", 0.20))

# The policy's tokenizer, for the prompt-length filter.
TOKENIZER = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


# --- Download and load ---------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(root: Path = RAW, names: Iterable[str] = SOURCES) -> list[Path]:
    """Fetch the files that are missing; check every file's sha256."""
    import httpx

    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        src = SOURCES[name]
        path = root / src.file
        if not path.exists():
            tmp = path.with_suffix(".part")
            with httpx.stream("GET", f"{BASE_URL}/{src.file}", timeout=60) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.rename(path)
        if (got := _sha256(path)) != src.sha256:
            raise ValueError(f"{path}: sha256 {got}, expected {src.sha256}")
        paths.append(path)
    return paths


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _tag_value(tags: Iterable[str], prefix: str) -> str | None:
    return next((t.removeprefix(prefix) for t in tags if t.startswith(prefix)), None)


def parse_example(row: dict, source: str = "oss_eval", seed: int = 0) -> Example:
    pid = row["prompt_id"]
    rubric = [
        Criterion(
            id=f"{pid}:{i}",
            text=r["criterion"],
            points=r["points"],
            axis=_tag_value(r["tags"], "axis:"),
            level=_tag_value(r["tags"], "level:"),
            tags=tuple(r["tags"]),
        )
        for i, r in enumerate(row["rubrics"])
    ]
    tags = tuple(row["example_tags"])
    themes = tuple(t.removeprefix("theme:") for t in tags if t.startswith("theme:"))
    return Example(
        id=pid,
        messages=row["prompt"],
        rubric=rubric,
        themes=themes,
        tags=tags,
        source=source,
        split=conversation_split(pid, seed),
    )


def load_examples(
    name: str = "oss_eval", root: Path = RAW, seed: int = 0
) -> list[Example]:
    rows = _read_jsonl(root / SOURCES[name].file)
    return [parse_example(r, name, seed) for r in rows]


def parse_judgment(row: dict) -> Judgment:
    return Judgment(
        id=f"{row['completion_id']}:{row['category']}",
        prompt_id=row["prompt_id"],
        completion_id=row["completion_id"],
        messages=row["prompt"],
        completion=row["completion"],
        rubric=row["rubric"],
        category=row["category"],
        physician_labels=tuple(row["binary_labels"]),
        physician_ids=tuple(row["anonymized_physician_ids"]),
    )


def load_judgments(root: Path = RAW) -> list[Judgment]:
    return [parse_judgment(r) for r in _read_jsonl(root / SOURCES["meta_eval"].file)]


# --- Splits --------------------------------------------------------------------


def _unit(*parts: object) -> float:
    """A uniform number in [0, 1) from a seeded hash of `parts`."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def conversation_split(prompt_id: str, seed: int = 0) -> str:
    u, edge = _unit("conversation", seed, prompt_id), 0.0
    for name, frac in SPLITS:
        edge += frac
        if u < edge:
            return name
    return SPLITS[-1][0]


@dataclass(frozen=True)
class CriteriaSplit:
    train: list[Criterion]  # the reward
    heldout: list[Criterion]  # eval only

    def to_dict(self) -> dict:
        return {
            "train_criteria": [c.to_dict() for c in self.train],
            "heldout_criteria": [c.to_dict() for c in self.heldout],
        }


def criteria_split(example: Example, seed: int = 0) -> CriteriaSplit | None:
    """Positives split in two disjoint halves; negatives in both. None if < 2 positives.

    Each half keeps the source order. With an odd count, train gets the extra one.
    """
    positives = [c for c in example.rubric if c.points > 0]
    if len(positives) < 2:
        return None
    order = sorted(positives, key=lambda c: _unit("criterion", seed, c.id))
    train_ids = {c.id for c in order[0::2]}
    train = [c for c in example.rubric if c.points < 0 or c.id in train_ids]
    heldout = [
        c
        for c in example.rubric
        if c.points < 0 or (c.points > 0 and c.id not in train_ids)
    ]
    return CriteriaSplit(train=train, heldout=heldout)


def stratified_subset(
    examples: list[Example], n: int, seed: int = 0, key: Callable | None = None
) -> list[Example]:
    """`n` examples, proportional by first theme (largest remainder), in hash order."""
    key = key or (lambda e: e.themes[0] if e.themes else "")
    groups: dict[str, list[Example]] = defaultdict(list)
    for e in sorted(examples, key=lambda e: _unit("subset", seed, e.id)):
        groups[key(e)].append(e)
    total = len(examples)
    if n >= total:
        return sorted(examples, key=lambda e: e.id)
    quotas = {g: n * len(v) / total for g, v in groups.items()}
    counts = {g: int(q) for g, q in quotas.items()}
    by_remainder = sorted(quotas, key=lambda g: (counts[g] - quotas[g], g))
    for g in by_remainder[: n - sum(counts.values())]:
        counts[g] += 1
    picked = [e for g, v in groups.items() for e in v[: counts[g]]]
    return sorted(picked, key=lambda e: e.id)


# --- Prompt length ---------------------------------------------------------------


def qwen_token_counter() -> Callable[[list[dict]], int]:
    """Approximate prompt tokens: content tokens + the chat template's ~5 per turn.

    Miles' `--rollout-max-prompt-len` applies the exact limit after templating.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])

    def count(messages: list[dict]) -> int:
        n = sum(
            len(tok.encode(m["content"], add_special_tokens=False)) for m in messages
        )
        return n + 5 * len(messages) + 3

    return count


# --- Export for Miles ----------------------------------------------------------


def miles_row(example: Example, split: CriteriaSplit, eval: bool) -> dict:
    """One Miles prompt row. The reward function reads everything from `metadata`.

    Miles turns `prompt` into the chat-templated string, so the messages are
    repeated in metadata for the judge.
    """
    return {
        "prompt": example.messages,
        "label": example.id,
        "metadata": {
            "prompt_id": example.id,
            "messages": example.messages,
            "themes": list(example.themes),
            "eval": eval,
            **split.to_dict(),
        },
    }


def read_ids(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def export_miles(
    examples: list[Example],
    out: Path = MILES_OUT,
    eval_ids_path: Path = EVAL_IDS,
    n_val: int = 64,
    n_test: int = 200,
    max_prompt_tokens: int = 2048,
    count_tokens: Callable[[list[dict]], int] | None = None,
    seed: int = 0,
) -> dict[str, int]:
    """Write train / val / test JSONL for Miles; freeze the test ids on first run.

    If `eval_ids_path` exists, its ids are the test subset (and must still be
    eligible test conversations); otherwise it is written.
    """
    splits = {e.id: criteria_split(e, seed) for e in examples}
    count_tokens = count_tokens or qwen_token_counter()
    eligible = [
        e
        for e in examples
        if splits[e.id] is not None and count_tokens(e.messages) <= max_prompt_tokens
    ]
    by_split = defaultdict(list)
    for e in eligible:
        by_split[e.split].append(e)

    if eval_ids_path.exists():
        ids = read_ids(eval_ids_path)
        test_by_id = {e.id: e for e in by_split["test"]}
        missing = [i for i in ids if i not in test_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} ids in {eval_ids_path} are not eligible test ids"
            )
        test = [test_by_id[i] for i in ids]
    else:
        test = stratified_subset(by_split["test"], n_test, seed)
        header = (
            f"# HealthBench oss_eval test subset: {len(test)} prompt_ids, seed {seed}\n"
        )
        eval_ids_path.parent.mkdir(parents=True, exist_ok=True)
        eval_ids_path.write_text(header + "".join(f"{e.id}\n" for e in test))
    val = stratified_subset(by_split["val"], n_val, seed)

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "healthbench_train.jsonl": (by_split["train"], False),
        "healthbench_val.jsonl": (val, True),
        "healthbench_test.jsonl": (test, True),
    }
    counts = {}
    for name, (rows, is_eval) in files.items():
        with (out / name).open("w") as f:
            for e in rows:
                f.write(json.dumps(miles_row(e, splits[e.id], is_eval)) + "\n")
        counts[name] = len(rows)
    counts["dropped_one_positive"] = sum(s is None for s in splits.values())
    counts["dropped_long_prompt"] = (
        len(examples) - counts["dropped_one_positive"] - len(eligible)
    )
    return counts


# --- Stats ---------------------------------------------------------------------


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def stats(examples: list[Example], judgments: list[Judgment] | None = None) -> str:
    """Markdown tables: splits, rubric sizes, axes, themes, the meta-eval."""
    out = []
    splits = Counter(e.split for e in examples)
    eligible = Counter(e.split for e in examples if criteria_split(e) is not None)
    out.append(
        _table(
            ["Split", "Conversations", "With >= 2 positive criteria"],
            [[s, splits[s], eligible[s]] for s, _ in SPLITS]
            + [["all", len(examples), sum(eligible.values())]],
        )
    )

    crit = [c for e in examples for c in e.rubric]
    sizes = sorted(len(e.rubric) for e in examples)
    pos = [c for c in crit if c.points > 0]
    neg = [c for c in crit if c.points < 0]
    turns = Counter(len(e.messages) for e in examples)
    levels = Counter(c.level for c in crit)
    out.append(
        _table(
            ["Rubrics", "Value"],
            [
                ["criteria", len(crit)],
                [
                    "per conversation: min / median / max",
                    f"{sizes[0]} / {sizes[len(sizes) // 2]} / {sizes[-1]}",
                ],
                ["positive / negative", f"{len(pos)} / {len(neg)}"],
                [
                    "example-level / cluster-level",
                    f"{levels['example']} / {levels['cluster']}",
                ],
                [
                    "single-turn conversations",
                    f"{turns[1]} ({turns[1] / len(examples):.0%})",
                ],
            ],
        )
    )

    axes = Counter(c.axis for c in crit)
    out.append(
        _table(
            ["Axis", "Criteria", "Share"],
            [[a, n, f"{n / len(crit):.0%}"] for a, n in axes.most_common()],
        )
    )
    themes = Counter(e.themes[0] for e in examples if e.themes)
    out.append(
        _table(["Theme", "Conversations"], [[t, n] for t, n in themes.most_common()])
    )

    if judgments:
        labels = [x for j in judgments for x in j.physician_labels]
        physicians = {p for j in judgments for p in j.physician_ids}
        out.append(
            _table(
                ["oss_meta_eval", "Value"],
                [
                    ["judgments (completion x rubric)", len(judgments)],
                    ["physician labels", len(labels)],
                    ["share labelled 'met'", f"{sum(labels) / len(labels):.0%}"],
                    ["physicians", len(physicians)],
                    ["categories", len({j.category for j in judgments})],
                ],
            )
        )
    return "\n\n".join(out) + "\n"


_root = click.option(
    "--root", type=click.Path(path_type=Path), default=RAW, show_default=True
)
_seed = click.option("--seed", default=0, show_default=True)


@click.group(help=__doc__.split("\n\n")[0])
def main() -> None:
    pass


@main.command("download")
@_root
def download_cmd(root: Path) -> None:
    for path in download(root):
        print(f"ok {path}")


@main.command("stats")
@_root
@_seed
def stats_cmd(root: Path, seed: int) -> None:
    print(stats(load_examples(root=root, seed=seed), load_judgments(root)))


@main.command("export-miles")
@_root
@_seed
@click.option(
    "--out", type=click.Path(path_type=Path), default=MILES_OUT, show_default=True
)
@click.option("--n-val", default=64, show_default=True)
@click.option("--n-test", default=200, show_default=True)
@click.option("--max-prompt-tokens", default=2048, show_default=True)
def export_miles_cmd(
    root: Path, seed: int, out: Path, n_val: int, n_test: int, max_prompt_tokens: int
) -> None:
    counts = export_miles(
        load_examples(root=root, seed=seed),
        out=out,
        n_val=n_val,
        n_test=n_test,
        max_prompt_tokens=max_prompt_tokens,
        seed=seed,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
