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

`--backfill` requires every verdict to be cached (it stops before any API call
otherwise) and keeps the local result JSONs, adding only their `weave_url`: their
usage and cost are those of the original runs.
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
def calibrate(options: dict, weave_project: str = WEAVE_PROJECT) -> dict:
    import asyncio
    import shutil

    from rubricrl.analysis import calibrate as cal
    from rubricrl.data import healthbench
    from rubricrl.judge.client import VerdictCache

    raw = JUDGE / "raw"
    healthbench.download(raw, ["meta_eval"])  # sha256-checked; skipped if present
    work = Path(tempfile.mkdtemp())
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
        )
    )
    cache.snapshot(work / "snapshot.sqlite")
    shutil.copy(work / "snapshot.sqlite", JUDGE / "judge.sqlite")
    (JUDGE / "incoming.sqlite").unlink(missing_ok=True)
    judge_volume.commit()
    return result


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
    """Write the result JSON locally; with `keep_existing`, only add `weave_url`."""
    path = OUT / f"{result['label']}-n{result['n']}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if keep_existing and path.exists():
        old = json.loads(path.read_text())
        same = old["labels"] == result["labels"]
        print(f"{path.name}: labels {'match' if same else 'DIFFER from'} the original")
        result = old | {"weave_url": result.get("weave_url")}
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
    sync_cache: bool = True,
    weave_project: str = WEAVE_PROJECT,
):
    """One calibration run (the flags), or --backfill: the README's four."""
    runs = BACKFILL if backfill else [(judge, effort, thinking, n, retest)]
    if sync_cache:
        _push_cache()
    keys = ("judge_name", "effort", "thinking", "n", "retest")
    for run in runs:
        options = dict(zip(keys, run, strict=True)) | {
            "seed": seed,
            "concurrency": concurrency,
            "require_cached": require_cached or backfill,
        }
        result = calibrate.remote(options, weave_project)
        path = _save(result, keep_existing=backfill)
        f1 = result["judge"]["overall"]["f1_balanced"]
        print(f"{result['label']} n={result['n']}: balanced F1 {f1:.3f} -> {path}")
        print(f"  weave: {result.get('weave_url')}")
    if sync_cache:
        _pull_cache()
