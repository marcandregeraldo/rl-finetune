"""GRPO with Miles on Modal, rewarded by the HealthBench rubric and a Claude judge.

Adapted from Modal's `14_clusters/miles_grpo.py` example
(https://github.com/modal-labs/modal-examples, MIT): the same pinned Miles v0.1.1
image, FSDP backend, colocated SGLang, and GRPO + TIS recipe, on **one node of 4
H100s** instead of two clustered 8-GPU nodes. What changes:

- **Model:** Qwen3.5-4B (Miles' FSDP backend has it: `fsdp_utils/models/qwen3_5.py`),
  thinking off. `--model qwen3-4b` is the fallback: the example's exact model.
- **Reward:** `rubricrl.rewards.miles_rm.healthbench_reward`: the rubric score on
  each conversation's *training* criteria, graded by Claude (Haiku 5.5 by default).
  The in-loop eval (`eval/healthbench_val`) is scored on the *held-out* criteria.
- **Data:** `data/miles/*.jsonl` (`python -m rubricrl.data.healthbench export-miles`),
  attached to the container (not baked into the image).
- **Responses capped at 1,024 tokens**, prompts at 2,048.

Miles runs in its own image, so this is the exception to the "Modal installs from
uv.lock" rule: the image is Miles', and `rubricrl` is added as source (its Miles
path needs only `anthropic`, pinned below, and numpy, already there).

Once: Modal secrets `anthropic` (ANTHROPIC_API_KEY), `wandb` (WANDB_API_KEY),
`huggingface-secret` (HF_TOKEN). Then (prefix with `uv run`):

    modal run modal/miles_rft.py --task math --smoke-test      # pipeline, no judge
    modal run modal/miles_rft.py --smoke-test                  # 2 steps, real judge
    modal run --detach modal/miles_rft.py --num-rollout 60     # the run
    modal run modal/miles_rft.py::export --run-id RUN_ID       # -> HF weights
    modal run modal/miles_rft.py::sync_traces --run-id RUN_ID  # judge log -> Weave

Outputs go to the `rubricrl-runs` Volume under RUN_ID: checkpoints, TensorBoard
events, `rewards.jsonl` (one line per graded sample: score, verdicts, response)
and the judge's SQLite cache. `sync_traces` sends `rewards.jsonl` to W&B Weave
(`rubricrl.observe`): a trace per graded sample (the conversation, the score, a
child per judge verdict), every eval record and a sample of the training ones,
plus a summary evaluation per eval round. It resumes where the last sync
stopped, so run it during the run (the Volume is committed every few minutes) and
once after with `--final`. Weave isn't installed in the Miles image: syncing runs
in its own.
"""

import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import modal

app = modal.App("rubricrl-miles-rft")
MINUTES = 60
TIMEOUT = 6 * 60 * MINUTES
GPU = "H100!"  # "!": never silently upgraded, so timings say which hardware
N_GPUS = 4
ROOT = Path(__file__).parent.parent
DATA = Path("/data")  # models and datasets (Volume rubricrl-data)
RUNS = Path("/runs")  # checkpoints and logs (Volume rubricrl-runs)
HB = Path("/rubricrl-data")  # data/miles, attached at container start
MILES = Path("/opt/miles")
MILES_REPO = "https://github.com/radixark/miles.git"
MILES_COMMIT = "2806267d060d51b1d3b62f85a1f9b145047aeef9"  # v0.1.1
ANTHROPIC_VERSION = "2.3.0"  # the version in uv.lock

# name -> (HF repo, revision). Same layout as modal/sample.py: DATA/models/<name>.
MODELS = {
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}
DATASETS = {
    "dapo-math-17k": (
        "zhuzilin/dapo-math-17k",
        "2e65612930298bde4c5d58fd97b3f23a483aaff9",
    ),
    "aime-2024": ("zhuzilin/aime-2024", "1c625e328db94ec7ef7ff169016b097c468d60b9"),
}

BOXED = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

data_volume = modal.Volume.from_name("rubricrl-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("rubricrl-runs", create_if_missing=True, version=2)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==0.34.4")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=download_image,
    volumes={DATA: data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * MINUTES,
)
def download(models: list[str], datasets: list[str] = ()) -> None:
    """Pinned snapshots into the data Volume; already-present files are skipped."""
    import json

    from huggingface_hub import snapshot_download

    for name in models:
        repo, rev = MODELS[name]
        snapshot_download(repo, revision=rev, local_dir=DATA / "models" / name)
    for name in datasets:
        repo, rev = DATASETS[name]
        snapshot_download(
            repo, repo_type="dataset", revision=rev, local_dir=DATA / name
        )
    if "aime-2024" in datasets:  # the math verifier needs a boxed answer
        aime = DATA / "aime-2024"
        with (
            (aime / "aime-2024.jsonl").open() as src,
            (aime / "aime-2024-boxed.jsonl").open("w") as dst,
        ):
            for line in src:
                row = json.loads(line)
                row["prompt"][-1]["content"] += BOXED
                dst.write(json.dumps(row) + "\n")
    data_volume.commit()


image = (
    modal.Image.from_registry(
        "radixark/miles:v0.1.1@sha256:6355834f16bacd35d5d40c43f142e3758376f7b2e8d678bccfe870c092bd96bf"
    )
    .entrypoint([])
    .run_commands(
        f"git clone --depth 1 --branch v0.1.1 {MILES_REPO} {MILES}",
        f"cd {MILES} && git checkout {MILES_COMMIT} && pip install --no-deps -e .",
        f"pip install anthropic=={ANTHROPIC_VERSION}",
    )
    .env(
        {
            # /root: rubricrl, for Miles' Ray actors that import the reward.
            "PYTHONPATH": f"{MILES}:/root/Megatron-LM:/root",
            "PYTHONUNBUFFERED": "1",
            "NCCL_NVLS_ENABLE": "0",
        }
    )
    .add_local_python_source("rubricrl")
    .add_local_dir(ROOT / "data" / "miles", str(HB))
)


# For sync_traces: uv.lock's environment with Weave, as on the Mac (the parity
# rule), not the Miles image. The Python series of .python-version, read locally.
PYTHON_SERIES = (
    ".".join((ROOT / ".python-version").read_text().strip().split(".")[:2])
    if modal.is_local()
    else None
)
observe_image = (
    modal.Image.debian_slim(python_version=PYTHON_SERIES)
    .uv_sync(
        str(ROOT), extras=["observe"], extra_options="--no-dev", uv_version="0.13.5"
    )
    .add_local_python_source("rubricrl")
    .add_local_dir(ROOT / "data" / "miles", str(HB))  # the conversations, by prompt id
)
COMMIT_EVERY_S = 300  # so sync_traces can read rewards.jsonl during the run


def training_args(
    task: str, model: str, run_dir: Path, num_rollout: int, smoke_test: bool
) -> list[str]:
    ckpt = DATA / "models" / model
    if task == "math":  # Modal's tested recipe on our model and node: no judge
        data = f"""
        --prompt-data {DATA}/dapo-math-17k/dapo-math-17k.jsonl
        --input-key prompt --label-key label --rm-type math
        --eval-prompt-data aime {DATA}/aime-2024/aime-2024-boxed.jsonl
        --rollout-max-response-len 2048 --eval-max-response-len 2048
        --n-samples-per-eval-prompt 1
        """
        prompts, n_samples = 4, 8
    else:
        val = f"{HB}/healthbench_val.jsonl" + ("@[0:8]" if smoke_test else "")
        data = f"""
        --prompt-data {HB}/healthbench_train.jsonl
        --input-key prompt --label-key label --metadata-key metadata
        --custom-rm-path rubricrl.rewards.miles_rm.healthbench_reward
        --eval-prompt-data healthbench_val {val}
        --rollout-max-response-len 1024 --eval-max-response-len 1024
        --rollout-max-prompt-len 2048
        --n-samples-per-eval-prompt {1 if smoke_test else 2}
        """
        prompts, n_samples = (4, 8) if smoke_test else (16, 8)
    return shlex.split(
        f"""
        --train-backend fsdp
        --hf-checkpoint {ckpt} --ref-load {ckpt}
        {data}
        --apply-chat-template
        --apply-chat-template-kwargs '{{"enable_thinking": false}}'
        --rollout-shuffle --balance-data
        --num-rollout {num_rollout}
        --rollout-batch-size {prompts} --n-samples-per-prompt {n_samples}
        --global-batch-size {prompts * n_samples}
        --rollout-temperature 1
        --eval-interval {1 if smoke_test else 10} --eval-top-p 1
        --advantage-estimator grpo --use-tis --tis-clip 2.0 --tis-clip-low 0.0
        --use-kl-loss --kl-loss-coef 0 --kl-loss-type low_var_kl
        --kl-coef 0 --entropy-coef 0
        --eps-clip 0.2 --eps-clip-high 0.28
        --optimizer adam --lr 1e-6 --lr-decay-style constant
        --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
        --rollout-num-gpus-per-engine 1
        --sglang-decode-log-interval 1000
        --sglang-mem-fraction-static 0.7
        --sglang-chunked-prefill-size 4096
        --update-weight-buffer-size 536870912
        --gradient-checkpointing --attn-implementation flash_attention_2
        --train-env-vars '{{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:False"}}'
        --use-dynamic-batch-size --max-tokens-per-gpu 16384
        --actor-num-nodes 1 --actor-num-gpus-per-node {N_GPUS}
        --colocate --use-fault-tolerance
        --save {run_dir}/checkpoints --save-interval {num_rollout if smoke_test else 20}
        --no-save-optim
        --use-tensorboard --tb-project-name rubricrl --tb-experiment-name {run_dir.name}
        --use-wandb --wandb-project rl-finetune --wandb-group {run_dir.name}
        """
    )


@app.function(
    image=image,
    gpu=f"{GPU}:{N_GPUS}",
    cpu=16,
    memory=128 * 1024,
    volumes={DATA: data_volume, RUNS: runs_volume},
    secrets=[
        modal.Secret.from_name("anthropic"),
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=TIMEOUT,
    single_use_containers=True,  # Ray processes die with the container
)
def train(
    run_id: str,
    task: str,
    model: str,
    num_rollout: int,
    smoke_test: bool,
    judge: str,
    judge_effort: str,
    judge_thinking: bool,
) -> str:
    import ray

    ip = socket.gethostbyname(socket.gethostname())
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = Path("/tmp/judge.sqlite")  # local disk; copied to the Volume at the end
    env = {
        **os.environ,
        "MASTER_ADDR": ip,
        "RAY_ADDRESS": f"{ip}:6379",
        "TENSORBOARD_DIR": str(run_dir / "tensorboard"),
        "RUBRICRL_JUDGE": judge,
        "RUBRICRL_EFFORT": judge_effort,
        "RUBRICRL_THINKING": "1" if judge_thinking else "",
        "RUBRICRL_CACHE": str(cache),
        "RUBRICRL_LOG": str(run_dir / "rewards.jsonl"),
    }
    args = training_args(task, model, run_dir, num_rollout, smoke_test)
    (run_dir / "args.txt").write_text(shlex.join(args) + "\n")
    data_volume.reload()
    subprocess.run(
        [
            "ray", "start", "--head", "--port=6379", f"--node-ip-address={ip}",
            f"--num-gpus={N_GPUS}", "--num-cpus=16", "--disable-usage-stats",
        ],
        env=env,
        check=True,
    )  # fmt: skip
    stop = threading.Event()

    def commit_periodically() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                runs_volume.commit()
            except Exception as e:  # never fail the run over a sync point
                print(f"periodic Volume commit failed: {e}", flush=True)

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        ray.init(address=f"{ip}:6379")
        print(f"Ray cluster ready: {ray.cluster_resources()}", flush=True)
        ray.shutdown()
        t0 = time.monotonic()
        subprocess.run(
            ["python", str(MILES / "train.py"), *args], cwd=MILES, env=env, check=True
        )
        print(f"training took {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    finally:
        stop.set()
        if cache.exists():
            shutil.copy(cache, run_dir / "judge.sqlite")
        runs_volume.commit()
    return str(run_dir)


@app.local_entrypoint()
def main(
    task: str = "healthbench",
    model: str = "qwen3.5-4b",
    num_rollout: int = 60,
    smoke_test: bool = False,
    judge: str = "haiku",
    judge_effort: str = "low",
    judge_thinking: bool = False,
):
    """--task healthbench|math, --model qwen3.5-4b|qwen3-4b, --judge haiku|sonnet."""
    if task not in ("healthbench", "math") or model not in MODELS:
        raise SystemExit(f"--task healthbench|math, --model one of {sorted(MODELS)}")
    if (
        task == "healthbench"
        and not (ROOT / "data/miles/healthbench_train.jsonl").exists()
    ):
        raise SystemExit("run `python -m rubricrl.data.healthbench export-miles` first")
    if smoke_test:
        num_rollout = 2
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{task}-{uuid.uuid4().hex[:6]}"
    print(f"Run ID: {run_id}")
    download.remote([model], ["dapo-math-17k", "aime-2024"] if task == "math" else [])
    result = train.spawn(
        run_id,
        task,
        model,
        num_rollout,
        smoke_test,
        judge,
        judge_effort,
        judge_thinking,
    ).get()
    print(f"Results in Volume {runs_volume.name}: {result}")


@app.function(
    image=image,
    cpu=8,
    memory=64 * 1024,
    volumes={DATA: data_volume, RUNS: runs_volume},
    timeout=30 * MINUTES,
)
def export(run_id: str, iteration: int = 0, model: str = "qwen3.5-4b") -> str:
    """FSDP checkpoint -> HF weights at RUN_ID/huggingface/iter_XXXXXXX (0 = latest)."""
    checkpoints = RUNS / run_id / "checkpoints"
    if iteration == 0:
        iteration = int((checkpoints / "latest_checkpointed_iteration.txt").read_text())
    checkpoint = checkpoints / f"iter_{iteration:07d}"
    output = RUNS / run_id / "huggingface" / checkpoint.name
    subprocess.run(
        [
            "python", str(MILES / "tools/convert_fsdp_to_hf.py"),
            "--input-dir", str(checkpoint),
            "--output-dir", str(output),
            "--origin-hf-dir", str(DATA / "models" / model),
        ],
        cwd=MILES,
        check=True,
    )  # fmt: skip
    runs_volume.commit()
    print(f"Hugging Face model saved to {output}")
    return str(output.relative_to(RUNS))


@app.function(
    image=observe_image,
    volumes={RUNS: runs_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=30 * MINUTES,
)
def sync_traces(run_id: str, rate: float = 0.1, final: bool = False) -> dict:
    """New lines of RUN_ID/rewards.jsonl -> Weave; the position is kept in the run dir.

    --rate   share of training records traced (eval records: all of them)
    --final  also send a trailing eval round (only once the run is over)
    """
    from rubricrl import observe

    runs_volume.reload()
    run_dir = RUNS / run_id
    state_path = run_dir / "weave_sync.json"
    state = {"offset": 0, "train_seen": 0, "urls": []}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    args = shlex.split((run_dir / "args.txt").read_text())
    batch = int(args[args.index("--global-batch-size") + 1])  # samples per step
    with (run_dir / "rewards.jsonl").open("rb") as f:
        f.seek(state["offset"])
        lines = f.read().split(b"\n")[:-1]  # complete lines only
    records = [json.loads(line) for line in lines]
    rounds, traces, consumed, train_seen = observe.split_records(
        records, state["train_seen"], batch, rate, run_id, final
    )
    conversations = {}
    for path in HB.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            metadata = json.loads(line)["metadata"]
            conversations[metadata["prompt_id"]] = metadata["messages"]
    observe.init()
    logged = observe.log_training(run_id, rounds, traces, conversations)
    state = {
        "offset": state["offset"] + sum(len(line) + 1 for line in lines[:consumed]),
        "train_seen": train_seen,
        "urls": state["urls"] + logged["urls"],
    }
    state_path.write_text(json.dumps(state, indent=1))
    runs_volume.commit()
    summary = {"records": consumed, "sampled_train_records": len(traces)} | logged
    print(summary)
    return summary
