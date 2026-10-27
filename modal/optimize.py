"""Judge prompt optimization (GEPA) on Modal, then the winner on the test set.

One CPU container runs `rubricrl.judge.optimize run`, logged as a W&B run
(`job_type="optimization"`), then `rubricrl.analysis.calibrate run --template`
with the best template on the calibration test set (n=1000, a W&B
calibration run plus Weave traces), and adds the comparison with the baseline
(seen vs held-out categories) to the optimization run's summary. Follow it in
W&B from anywhere; `--detach` keeps it running when the laptop sleeps.

**Caches:** the judge's verdicts and the reflection model's outputs live on the
`rubricrl-judge` Volume (`judge.sqlite`, `llm.sqlite`) and are merged both ways
with the Mac's, as in `modal/calibrate.py`. A rerun with the same settings
replays the cached steps for free: a run stopped halfway resumes where it was.
They are written back to the Volume even if the run fails.

**Results** land on the Volume under `optimize/<name>/` and `calibration/`,
and are copied to `analysis/out/` when the local entrypoint is still attached
at the end; otherwise `--pull` fetches them later.

    uv run modal run --detach modal/optimize.py --name gepa-v1
    uv run modal run --detach modal/optimize.py --name gepa-v1 --gepa-seeds 1,2
    uv run modal run modal/optimize.py --pull --name gepa-v1
"""

import json
import tempfile
from pathlib import Path

import modal

app = modal.App("rubricrl-optimize")
MINUTES = 60
ROOT = Path(__file__).parent.parent
JUDGE = Path("/judge")  # Volume rubricrl-judge (shared with modal/calibrate.py)
OUT = ROOT / "analysis" / "out"
LOCAL = Path.home() / ".cache" / "rubricrl"
CACHES = ("judge.sqlite", "llm.sqlite")
PROJECT = "rl-finetune"


def baseline_path(judge: str, effort: str, thinking: bool) -> Path:
    """The current prompt's calibration with the same judge config: what the
    optimized template is compared with on the test set."""
    from rubricrl.analysis.calibrate import label
    from rubricrl.judge.client import JudgeConfig

    config = JudgeConfig.named(judge, effort=effort, thinking=thinking)
    return OUT / "calibration" / f"{label(config)}-n1000.json"


PYTHON_SERIES = (
    ".".join((ROOT / ".python-version").read_text().strip().split(".")[:2])
    if modal.is_local()
    else None
)

image = (
    modal.Image.debian_slim(python_version=PYTHON_SERIES)
    .uv_sync(
        str(ROOT),
        extras=["observe", "optimize"],
        extra_options="--no-dev",
        uv_version="0.13.5",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("rubricrl")
)
judge_volume = modal.Volume.from_name("rubricrl-judge", create_if_missing=True)


def _cache(name: str, path: Path):
    from rubricrl.judge.client import VerdictCache
    from rubricrl.llm import JSONCache

    return (VerdictCache if name == "judge.sqlite" else JSONCache)(path)


@app.function(
    image=image,
    volumes={str(JUDGE): judge_volume},
    secrets=[modal.Secret.from_name("anthropic"), modal.Secret.from_name("wandb")],
    cpu=2,
    memory=4096,
    timeout=12 * 60 * MINUTES,
)
def optimize(options: dict, baseline: dict, git: dict, test_n: int = 1000) -> dict:
    import asyncio
    import os
    import shutil

    for k, v in git.items():
        os.environ[k] = v
    from rubricrl.analysis import calibrate as cal
    from rubricrl.data import healthbench
    from rubricrl.judge import optimize as opt

    raw = JUDGE / "raw"
    healthbench.download(raw, ["meta_eval"])  # sha256-checked; skipped if present
    work = Path(tempfile.mkdtemp())
    caches = {}
    for name in CACHES:  # the Volume's, plus the Mac's (incoming-*)
        caches[name] = _cache(name, work / name)
        for src in (name, f"incoming-{name}"):
            if (JUDGE / src).exists():
                shutil.copy(JUDGE / src, work / f"src-{src}")
                added = caches[name].merge(work / f"src-{src}")
                print(f"cache: +{added} from {src}")
    folder = JUDGE / "optimize" / options["name"]
    try:
        summary = opt.run(
            **options,
            judge_cache=str(work / "judge.sqlite"),
            llm_cache=str(work / "llm.sqlite"),
            root=raw,
            out=JUDGE / "optimize",
            wandb_project=PROJECT,
        )
        judge_volume.commit()
        test = asyncio.run(
            cal.run(
                judge_name=options["judge_name"],
                n=test_n,
                effort=options["effort"],
                thinking=options["thinking"],
                retest=200,
                cache=str(work / "judge.sqlite"),
                root=raw,
                out=JUDGE / "calibration",
                weave_project=PROJECT,
                wandb_project=PROJECT,
                template=Path(summary["template_path"]),
                prompt_version=options["name"],
            )
        )
        held = summary["config"]["heldout_categories"]
        comparison = opt.compare(baseline, test, held, raw)
        summary["test"] = comparison | {"calibration_wandb_url": test["wandb_url"]}
        (folder / "result.json").write_text(json.dumps(summary, indent=1))
        _add_to_run(summary["wandb_url"], comparison, test["wandb_url"])
        print(json.dumps(comparison, indent=1))
        return {"summary": summary, "test": test}
    finally:
        judge_volume.reload()  # parallel runs (--gepa-seeds): keep theirs too
        for name, cache in caches.items():
            if (JUDGE / name).exists():
                shutil.copy(JUDGE / name, work / f"now-{name}")
                cache.merge(work / f"now-{name}")
            cache.snapshot(work / f"snap-{name}")
            shutil.copy(work / f"snap-{name}", JUDGE / name)
            (JUDGE / f"incoming-{name}").unlink(missing_ok=True)
        judge_volume.commit()


def _add_to_run(url: str | None, comparison: dict, test_url: str | None) -> None:
    """The test-set comparison, into the optimization run's summary."""
    if not url:
        return
    import wandb

    run = wandb.init(project=PROJECT, id=url.rstrip("/").split("/")[-1],
                     resume="must")  # fmt: skip
    run.summary.update(
        {f"test/{k}": v for k, v in comparison.items()}
        | {"test/calibration_run": test_url}
    )
    run.finish()


def _git() -> dict:
    import subprocess

    def out(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout

    return {
        "RUBRICRL_GIT_SHA": out("git", "rev-parse", "HEAD").strip(),
        "RUBRICRL_GIT_DIRTY": "1"
        if out("git", "status", "--porcelain").strip()
        else "0",
    }


def _push_caches() -> None:
    for name in CACHES:
        if not (LOCAL / name).exists():
            continue
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / name
            _cache(name, LOCAL / name).snapshot(snap)
            with judge_volume.batch_upload(force=True) as batch:
                batch.put_file(snap, f"/incoming-{name}")
        print(f"cache: uploaded {LOCAL / name}")


def _pull_caches() -> None:
    for name in CACHES:
        with tempfile.TemporaryDirectory() as d:
            remote = Path(d) / name
            with remote.open("wb") as f:
                for chunk in judge_volume.read_file(name):
                    f.write(chunk)
            added = _cache(name, LOCAL / name).merge(remote)
        print(f"cache: +{added} into {LOCAL / name}")


def _pull_results(name: str) -> None:
    """optimize/<name>/ (not GEPA's run state) and the test calibration JSON."""
    wanted = [
        e.path
        for e in judge_volume.listdir(f"optimize/{name}")
        if e.type == modal.volume.FileEntryType.FILE
    ] + [
        e.path
        for e in judge_volume.listdir("calibration")
        if e.path.endswith(f"-{name}-n1000.json")
    ]
    for path in wanted:
        dst = OUT / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as f:
            for chunk in judge_volume.read_file(path):
                f.write(chunk)
        print(f"pulled {dst}")


@app.local_entrypoint()
def main(
    name: str = "gepa-v1",
    judge: str = "haiku",
    effort: str = "medium",
    thinking: bool = True,
    reflection: str = "sonnet",
    reflection_effort: str = "medium",
    minibatch: int = 8,
    n_train: int = 400,
    n_val: int = 200,
    max_metric_calls: int = 3000,
    budget_usd: float = 8.0,
    seed: int = 0,
    gepa_seeds: str = "",
    sync_cache: bool = True,
    pull: bool = False,
):
    """Optimize, then score the best template on the test set; or --pull the
    results of a run that finished while detached.

    `--gepa-seeds 1,2`: repeats on the same data with other GEPA seeds, in
    parallel, named `<name>-s<seed>` (one W&B group, `<name>`)."""
    seeds = [int(x) for x in gepa_seeds.split(",")] if gepa_seeds else [None]
    names = [name if g is None else f"{name}-s{g}" for g in seeds]
    if pull:
        for n in names:
            _pull_results(n)
        if sync_cache:
            _pull_caches()
        return
    base = {
        "judge_name": judge, "effort": effort, "thinking": thinking,
        "reflection": reflection, "reflection_effort": reflection_effort,
        "minibatch": minibatch, "n_train": n_train, "n_val": n_val,
        "max_metric_calls": max_metric_calls,
        "budget_usd": budget_usd, "seed": seed, "group": name,
    }  # fmt: skip
    if sync_cache:
        _push_caches()
    baseline = json.loads(baseline_path(judge, effort, thinking).read_text())
    git = _git()
    calls = [
        (base | {"name": n, "gepa_seed": g}, baseline, git)
        for n, g in zip(names, seeds, strict=True)
    ]
    outs = list(optimize.starmap(calls, return_exceptions=True))
    for n in names:
        _pull_results(n)
    if sync_cache:
        _pull_caches()
    for n, out in zip(names, outs, strict=True):
        if isinstance(out, BaseException):
            print(f"{n}: FAILED {out!r}")
            continue
        s = out["summary"]
        print(f"{n}: best candidate {s['best_index']}; cost ${s['cost_usd']} + "
              f"test ${out['test']['usage']['cost_usd']}")  # fmt: skip
        print(json.dumps(s["test"], indent=1))
        print(f"wandb: {s.get('wandb_url')}")
