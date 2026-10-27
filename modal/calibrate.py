"""Judge calibration on Modal, logged to W&B Weave.

`rubricrl.analysis.calibrate run` in a CPU container: the job is API-bound, so no
GPU. In the `rl-finetune` Weave project (`rubricrl.observe`), every judge call
is its own `judge_verdict` trace (verdict, explanation, tokens, agreement with
the physicians), and each run's metrics a summary-only evaluation.

**The verdict cache** lives on the `rubricrl-judge` Volume, and is merged both
ways with the Mac's (`~/.cache/rubricrl/judge.sqlite`) on every run, so a verdict
paid for on either side is never paid for again (`--no-sync-cache` skips that).
The container works on a local copy: SQLite's WAL mode isn't safe on a network
filesystem.

**Results** are written back to `analysis/out/calibration/`, so
`python -m rubricrl.analysis.calibrate report` works as before.

Once: Modal secrets `anthropic` (ANTHROPIC_API_KEY) and `wandb` (WANDB_API_KEY).
Then (prefix with `uv run`):

    modal run modal/calibrate.py --backfill          # the 4 README runs, $0: cached
    modal run modal/calibrate.py --judge haiku --effort medium --thinking --n 1000
    modal run modal/calibrate.py --judge sonnet --n 1000 --retest 100
    modal run modal/calibrate.py --grader trajectory --n 1000    # the v2 grader prompt
    modal run modal/calibrate.py --effort low --n 1000 \
        --template analysis/out/optimize/gepa-v1/gepa-v1.txt   # an optimized prompt

`--backfill` requires every verdict to be cached (it stops before any API call
otherwise) and keeps the local result JSONs, adding only their Weave and W&B
links: their usage and cost are those of the original runs.
"""

import json
import tempfile
from pathlib import Path

import modal

app = modal.App("rubricrl-calibrate")
MINUTES = 60
ROOT = Path(__file__).parent.parent
JUDGE = Path("/judge")  # Volume rubricrl-judge: judge.sqlite, raw/, calibration/
OUT = ROOT / "analysis" / "out" / "calibration"
LOCAL_CACHE = Path.home() / ".cache" / "rubricrl" / "judge.sqlite"
WEAVE_PROJECT = "rl-finetune"

# The runs in the README's calibration table: (judge, effort, thinking, n, retest).
BACKFILL = [
    ("haiku", "low", False, 2000, 200),
    ("haiku", "low", False, 1000, 200),
    ("haiku", "medium", True, 1000, 100),
    ("sonnet", "low", False, 1000, 100),
]

# The Python series of .python-version; read only locally (see modal/ray_cluster.py).
PYTHON_SERIES = (
    ".".join((ROOT / ".python-version").read_text().strip().split(".")[:2])
    if modal.is_local()
    else None
)

image = (
    modal.Image.debian_slim(python_version=PYTHON_SERIES)
    # Exactly what uv.lock pins, as on the Mac; uv_sync skips the project itself.
    .uv_sync(
        str(ROOT), extras=["observe"], extra_options="--no-dev", uv_version="0.13.5"
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("rubricrl")
)
judge_volume = modal.Volume.from_name("rubricrl-judge", create_if_missing=True)


@app.function(
    image=image,
    volumes={str(JUDGE): judge_volume},
    secrets=[modal.Secret.from_name("anthropic"), modal.Secret.from_name("wandb")],
    cpu=2,
    memory=4096,
    timeout=120 * MINUTES,
)
def calibrate(
    options: dict, weave_project: str = WEAVE_PROJECT, git: dict | None = None
) -> dict:
    """One calibration run; also a W&B run (`job_type="calibration"`) in the same
    project, with the Mac's git commit in its config."""
    import os

    for k, v in (git or {}).items():
        os.environ[k] = v
    import asyncio
    import shutil

    from rubricrl.analysis import calibrate as cal
    from rubricrl.data import healthbench
    from rubricrl.judge.client import VerdictCache

    raw = JUDGE / "raw"
    healthbench.download(raw, ["meta_eval"])  # sha256-checked; skipped if present
    work = Path(tempfile.mkdtemp())
    if text := options.pop("template_text", None):  # --template, sent as text
        options["template"] = work / f"{options['prompt_version']}.txt"
        options["template"].write_text(text)
    cache = VerdictCache(work / "judge.sqlite")
    for name in ("judge.sqlite", "incoming.sqlite"):  # incoming: the Mac's
        if (JUDGE / name).exists():
            shutil.copy(JUDGE / name, work / name.replace(".", "-src."))
            added = cache.merge(work / name.replace(".", "-src."))
            print(f"cache: +{added} verdicts from {name}")
    result = asyncio.run(
        cal.run(
            **options,
            cache=str(work / "judge.sqlite"),
            root=raw,
            out=JUDGE / "calibration",
            weave_project=weave_project,
            wandb_project=weave_project,
        )
    )
    cache.snapshot(work / "snapshot.sqlite")
    shutil.copy(work / "snapshot.sqlite", JUDGE / "judge.sqlite")
    (JUDGE / "incoming.sqlite").unlink(missing_ok=True)
    judge_volume.commit()
    return result


def _git() -> dict:
    """This checkout's commit, for the run's config (no git inside the container)."""
    import subprocess

    def out(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout

    return {
        "RUBRICRL_GIT_SHA": out("git", "rev-parse", "HEAD").strip(),
        "RUBRICRL_GIT_DIRTY": "1"
        if out("git", "status", "--porcelain").strip()
        else "0",
    }


def _push_cache() -> None:
    from rubricrl.judge.client import VerdictCache

    if not LOCAL_CACHE.exists():
        return
    with tempfile.TemporaryDirectory() as d:
        snap = Path(d) / "judge.sqlite"
        VerdictCache(LOCAL_CACHE).snapshot(snap)
        with judge_volume.batch_upload(force=True) as batch:
            batch.put_file(snap, "/incoming.sqlite")
    print(f"cache: uploaded {LOCAL_CACHE}")


def _pull_cache() -> None:
    from rubricrl.judge.client import VerdictCache

    with tempfile.TemporaryDirectory() as d:
        remote = Path(d) / "remote.sqlite"
        with remote.open("wb") as f:
            for chunk in judge_volume.read_file("judge.sqlite"):
                f.write(chunk)
        added = VerdictCache(LOCAL_CACHE).merge(remote)
    print(f"cache: +{added} verdicts into {LOCAL_CACHE}")


def _save(result: dict, keep_existing: bool) -> Path:
    """Write the result JSON locally; with `keep_existing`, only add the links."""
    path = OUT / f"{result['label']}-n{result['n']}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if keep_existing and path.exists():
        old = json.loads(path.read_text())
        same = old["labels"] == result["labels"]
        print(f"{path.name}: labels {'match' if same else 'DIFFER from'} the original")
        result = old | {k: result.get(k) for k in ("weave_url", "wandb_url")}
    path.write_text(json.dumps(result, indent=1))
    return path


@app.local_entrypoint()
def main(
    judge: str = "haiku",
    effort: str = "low",
    thinking: bool = False,
    n: int = 1000,
    retest: int = 200,
    seed: int = 0,
    concurrency: int = 32,
    require_cached: bool = False,
    backfill: bool = False,
    grader: str = "response",
    template: str = "",
    prompt_version: str = "",
    sync_cache: bool = True,
    weave_project: str = WEAVE_PROJECT,
):
    """One calibration run (the flags), or --backfill: the README's four.
    `--template FILE`: a response template instead of simple-evals' (an optimized
    one, `judge/optimize.py`), named `--prompt-version` (default: its stem)."""
    runs = BACKFILL if backfill else [(judge, effort, thinking, n, retest)]
    if sync_cache:
        _push_cache()
    keys = ("judge_name", "effort", "thinking", "n", "retest")
    for run in runs:
        options = dict(zip(keys, run, strict=True)) | {
            "grader": "response" if backfill else grader,
            "seed": seed,
            "concurrency": concurrency,
            "require_cached": require_cached or backfill,
        }
        if template and not backfill:
            options |= {
                "template_text": Path(template).read_text(),
                "prompt_version": prompt_version or Path(template).stem,
            }
        result = calibrate.remote(options, weave_project, _git())
        path = _save(result, keep_existing=backfill)
        f1 = result["judge"]["overall"]["f1_balanced"]
        print(f"{result['label']} n={result['n']}: balanced F1 {f1:.3f} -> {path}")
        print(f"  weave: {result.get('weave_url')}")
        print(f"  wandb: {result.get('wandb_url')}")
    if sync_cache:
        _pull_cache()
