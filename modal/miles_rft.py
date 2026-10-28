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
- **Responses capped at 8,192 tokens** (2,048 per turn in patient mode), prompts
  at 2,048.

Miles runs in its own image, so this is the exception to the "Modal installs from
uv.lock" rule: the image is Miles', and `rubricrl` is added as source (its Miles
path needs only `anthropic`, pinned below, and numpy, already there).

Once: Modal secrets `anthropic` (ANTHROPIC_API_KEY), `wandb` (WANDB_API_KEY),
`huggingface-secret` (HF_TOKEN). Then (prefix with `uv run`):

    modal run modal/miles_rft.py::main --task math --smoke-test   # pipeline, no judge
    modal run modal/miles_rft.py::main --smoke-test               # 2 steps, real judge
    modal run --detach modal/miles_rft.py::main --num-rollout 60  # the run
    modal run modal/miles_rft.py::export --run-id RUN_ID          # -> HF weights
    modal run modal/miles_rft.py::sync_traces --run-id RUN_ID     # judge log -> Weave
    modal run modal/miles_rft.py::main --smoke-test --reward env  # vs the OpenEnv env
    modal run modal/miles_rft.py::main --smoke-test --reward env --env-mode patient
    modal run --detach modal/miles_rft.py::main --reward env --no-dashboard
    modal run modal/miles_rft.py::ray_check --hold-s 600          # try the login (CPU)
    modal run modal/miles_rft.py::log_check                       # W&B keys (CPU)

`::main` is needed: the file has two local entrypoints (`calibrate_judge` too).

**Watching a run.** W&B (project rl-finetune, one run named RUN_ID, tagged with
the reward path and judge): Miles' `train/*` (loss, pg_loss, grad_norm, lr-pg_0,
KL, clip fractions, TIS), `rollout/*` (reward, response lengths, truncation,
zero-variance groups), `eval/<dataset>` (the held-out criteria score), `perf/*`
(timings), plus the rubric's `rollout/rubric/*` and `eval/<dataset>/rubric/*`
(per-axis scores, penalty criteria met, failed verdicts, judge cost, policy vs
environment time) and `*/run/*` (groups with no signal, perfect and nonpositive
shares, per-theme scores, tokens, running GPU + judge cost; after the panels of
Xiaomi's MiMo-V2.6 RL dashboard): `rubricrl.rewards.miles_log`.
Every run serves the Ray dashboard (actors, logs, GPU use) through a Modal
tunnel, behind a fresh Ray token, with its time-series charts from Prometheus and
Grafana (view-only, its own tunnel): the URLs and the command that copies the
token are in the Modal logs. `--no-metrics` skips Prometheus and Grafana,
`--no-dashboard` all of it (`modal container exec CONTAINER_ID ray status` still
reaches the cluster). `log_check` (CPU) prints every key a step logs.

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

import contextlib
import json
import os
import secrets
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
N_GPUS = 4  # the default; `--gpus 2` runs the same recipe on two
JUDGE_CONCURRENCY = 256  # judge requests in flight; the SDK retries 429s with backoff
GPU_USD_PER_HOUR = 3.95  # one H100 on Modal (list price): the W&B cost estimate
ROOT = Path(__file__).parent.parent
DATA = Path("/data")  # models and datasets (Volume rubricrl-data)
RUNS = Path("/runs")  # checkpoints and logs (Volume rubricrl-runs)
HB = Path("/rubricrl-data")  # data/miles, attached at container start
MILES = Path("/opt/miles")
DASHBOARD_PORT, GRAFANA_PORT, PROMETHEUS_PORT = 8265, 3000, 9090
RAY_TOKEN = Path("/root/.ray/auth_token")  # where Ray's clients look for it
# The Ray dashboard's time-series charts (`--dashboard`): Prometheus + Grafana,
# pinned and baked into the image, as in the logo-segmentation project. Ray's
# dashboard writes ready-made configs for both here when it starts.
PROMETHEUS_VERSION, GRAFANA_VERSION = "3.15.1", "13.2.4"
RAY_METRICS_DIR = Path("/tmp/ray/session_latest/metrics")
ENV_SRC, ENV_VENV = "/opt/rubricrl-env-src", "/opt/rubricrl-env"  # the env server
ENV_PORT = 8003
MILES_REPO = "https://github.com/radixark/miles.git"
MILES_COMMIT = "2806267d060d51b1d3b62f85a1f9b145047aeef9"  # v0.1.1
# Miles' 120 s wait for its session server, patched in the image (see `image`).
READY_PY, READY_VAR = (
    MILES / "miles/ray/rollout/router_manager.py",
    "_SERVER_READY_TIMEOUT_SECS",
)
READY_S = 600
ANTHROPIC_VERSION = "2.3.0"  # the version in uv.lock

# name -> (HF repo, revision). Same layout as modal/sample.py: DATA/models/<name>.
MODELS = {
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}
# Self-hosted judges (`calibrate_judge`, `--judge-model`): name -> (HF repo,
# revision, GPUs), served by the image's SGLang from DATA/models/<name>.
JUDGE_MODELS = {
    # 2: on one H100 the image's SGLang runs out of memory loading it (70 GB)
    "gpt-oss-120b": (
        "openai/gpt-oss-120b",
        "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        2,
    ),
    "gpt-oss-20b": (
        "openai/gpt-oss-20b",
        "6cee5e81ee83917806bbde320786a8fb61efebee",
        1,
    ),
    "qwen3.6-35b-a3b": (
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        2,
    ),
    "qwen3.5-4b": (*MODELS["qwen3.5-4b"], 1),
}
JUDGE_PORT = 8004  # next to the env server's; Miles' engines pick their own
JUDGE = Path("/judge")  # Volume rubricrl-judge: raw/ (the meta-eval), calibration/
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
judge_volume = modal.Volume.from_name("rubricrl-judge", create_if_missing=True)

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
        repo, rev = MODELS[name] if name in MODELS else JUDGE_MODELS[name][:2]
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
    # The consultation environment's server (--reward env) runs from its own venv,
    # uv.lock's exact versions (the parity rule): OpenEnv's server stack (FastAPI,
    # gradio, fastmcp) never touches Miles' Python. Miles' side only needs
    # `websockets` (rubricrl.env.wire).
    .add_local_file(ROOT / "pyproject.toml", f"{ENV_SRC}/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", f"{ENV_SRC}/uv.lock", copy=True)
    .add_local_file(ROOT / ".python-version", f"{ENV_SRC}/.python-version", copy=True)
    .run_commands(
        "pip install uv==0.13.5",
        f"cd {ENV_SRC} && UV_PROJECT_ENVIRONMENT={ENV_VENV} uv sync --frozen"
        " --no-dev --no-install-project --extra env --extra observe",
    )
    # Miles waits 120 s for its session server (--use-session-server). Alone it
    # starts in ~35 s (importing transformers / Megatron, `session_check`); in the
    # training container it starts with 4 SGLang engines and 4 trainers loading at
    # once, and missed the 120 s in the first smoke run. A later layer, so the
    # Miles install above stays cached; fails the build if the line moved.
    .run_commands(
        f"grep -q '^{READY_VAR} = 120$' {READY_PY}"
        f" && sed -i 's/^{READY_VAR} = 120$/{READY_VAR} = {READY_S}/' {READY_PY}"
        f" && grep -q '^{READY_VAR} = {READY_S}$' {READY_PY}"
    )
    .run_commands(
        "command -v curl || (apt-get update && apt-get install -y curl)",
        "mkdir -p /opt/prometheus /opt/grafana",
        "curl -fsSL https://github.com/prometheus/prometheus/releases/download/"
        f"v{PROMETHEUS_VERSION}/prometheus-{PROMETHEUS_VERSION}.linux-amd64.tar.gz"
        " | tar xz --strip-components=1 -C /opt/prometheus",
        "curl -fsSL https://dl.grafana.com/oss/release/"
        f"grafana-{GRAFANA_VERSION}.linux-amd64.tar.gz"
        " | tar xz --strip-components=1 -C /opt/grafana",
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


DYNAMIC_SAMPLING = (
    "--dynamic-sampling-filter-path"
    " miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter"
)

# The rubric's W&B metrics (per axis, penalty criteria, failed verdicts), next to
# Miles' own; both hooks keep Miles' default logging.
RUBRIC_LOGS = """
    --custom-rollout-log-function-path rubricrl.rewards.miles_log.log_rollout
    --custom-eval-rollout-log-function-path rubricrl.rewards.miles_log.log_eval
"""

# --tito-model for Miles' session server (token-in-token-out), by model family.
TITO = {"qwen3.5-4b": "qwen35", "qwen3-4b": "qwen3"}

# Token budgets (Qwen3.5-4B's context is 262K; memory is the limit). At 1,024,
# 53-66% of training replies (temperature 1) were cut off in the smoke run, and
# a cut-off reply is graded as incomplete: the reward would learn the cap.
PROMPT_TOKENS = 2048  # longer prompts are filtered out
REPLY_TOKENS = 8192  # per reply, single-turn (rm and env v1)
PATIENT_TURN_TOKENS = 2048  # per doctor turn, patient mode (6 turns)
MAX_TOKENS_PER_GPU = 16384  # per micro-batch; also caps a patient-mode session


def training_args(
    task: str,
    model: str,
    run_dir: Path,
    num_rollout: int,
    smoke_test: bool,
    reward: str = "rm",
    env_mode: str = "single",
    lr: float = 1e-6,
    dynamic_sampling: bool = False,
    save_interval: int = 5,
    std_normalization: bool = True,
    gpus: int = N_GPUS,
    prompts_per_step: int = 16,
) -> list[str]:
    """Miles' arguments. `reward`: "rm", the in-process reward function
    (rubricrl.rewards.miles_rm), or "env", episodes against the consultation
    environment (rubricrl.env) through Miles' agentic rollout; `env_mode`
    "single" (v1) or "patient" (v2, simulated patient).

    `dynamic_sampling`: DAPO's online prompt filter. A group whose samples all
    scored the same (no advantage, nothing to learn) is dropped and fresh prompts
    are drawn until the batch has `rollout-batch-size` useful groups; Miles logs
    the drops with the rollout metrics.

    `save_interval`: a checkpoint every N steps (and at the end), so a failed
    save (run 3cec94 lost its only, final checkpoint to an NCCL error mid-save)
    still leaves an earlier one, and mid-run checkpoints can be evaluated.

    `std_normalization=False`: GRPO advantages are reward - group mean, not also
    divided by the group's std (Dr. GRPO). Groups whose scores are nearly tied
    then stop getting their tiny differences scaled up to full-size advantages.

    `gpus`, `prompts_per_step`: the GPUs the trainers and engines share, and the
    prompts per step (x 8 replies, one optimizer step per batch)."""
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
    elif reward == "env":
        prefix = "healthbench" if env_mode == "single" else "consult"
        val = f"{HB}/{prefix}_val.jsonl" + ("@[0:8]" if smoke_test else "")
        # --max-seq-len: the whole session (prompt + every turn) stays within
        # one micro-batch (--max-tokens-per-gpu). Single: one 8K reply. Patient:
        # 6 doctor turns, so 2K each (2,048 + 6 x (2,048 + a short patient turn)).
        reply = REPLY_TOKENS if env_mode == "single" else PATIENT_TURN_TOKENS
        max_seq = PROMPT_TOKENS + reply if env_mode == "single" else MAX_TOKENS_PER_GPU
        data = f"""
        --prompt-data {HB}/{prefix}_train.jsonl
        --input-key prompt --label-key label --metadata-key metadata
        --custom-generate-function-path
        miles.rollout.generate_hub.agentic_tool_call.generate
        --custom-agent-function-path rubricrl.env.miles_agent.run
        --custom-rm-path rubricrl.env.miles_agent.reward_func
        --use-session-server --tito-model {TITO[model]} --max-seq-len {max_seq}
        --eval-prompt-data {prefix}_val {val}
        --rollout-max-response-len {reply} --eval-max-response-len {reply}
        --rollout-max-prompt-len {PROMPT_TOKENS}
        --n-samples-per-eval-prompt {1 if smoke_test else 2}
        """
        prompts, n_samples = (4, 8) if smoke_test else (prompts_per_step, 8)
    else:
        val = f"{HB}/healthbench_val.jsonl" + ("@[0:8]" if smoke_test else "")
        data = f"""
        --prompt-data {HB}/healthbench_train.jsonl
        --input-key prompt --label-key label --metadata-key metadata
        --custom-rm-path rubricrl.rewards.miles_rm.healthbench_reward
        --eval-prompt-data healthbench_val {val}
        --rollout-max-response-len {REPLY_TOKENS} --eval-max-response-len {REPLY_TOKENS}
        --rollout-max-prompt-len {PROMPT_TOKENS}
        --n-samples-per-eval-prompt {1 if smoke_test else 2}
        """
        prompts, n_samples = (4, 8) if smoke_test else (prompts_per_step, 8)
    # The agentic path renders the chat template in the session server (TITO):
    # prompts stay message lists, and the agent turns thinking off per request.
    template = (
        ""
        if task != "math" and reward == "env"
        else """--apply-chat-template
        --apply-chat-template-kwargs '{"enable_thinking": false}'"""
    )
    save_every = num_rollout if smoke_test else save_interval
    return shlex.split(
        f"""
        --train-backend fsdp
        --hf-checkpoint {ckpt} --ref-load {ckpt}
        {data}
        {template}
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
        --optimizer adam --lr {lr} --lr-decay-style constant
        --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
        --rollout-num-gpus-per-engine 1
        --sglang-decode-log-interval 1000
        --sglang-mem-fraction-static 0.7
        --sglang-chunked-prefill-size 4096
        --update-weight-buffer-size 536870912
        --gradient-checkpointing --attn-implementation flash_attention_2
        --train-env-vars '{{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:False"}}'
        --use-dynamic-batch-size --max-tokens-per-gpu {MAX_TOKENS_PER_GPU}
        --actor-num-nodes 1 --actor-num-gpus-per-node {gpus}
        --colocate --use-fault-tolerance
        --save {run_dir}/checkpoints --save-interval {save_every}
        --no-save-optim
        --use-tensorboard --tb-project-name rubricrl --tb-experiment-name {run_dir.name}
        --use-wandb --wandb-project rl-finetune --wandb-group {run_dir.name}
        --disable-wandb-random-suffix
        {DYNAMIC_SAMPLING if dynamic_sampling else ""}
        {"" if std_normalization else "--disable-grpo-std-normalization"}
        {"" if task == "math" else RUBRIC_LOGS}
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
    reward: str = "rm",
    env_mode: str = "single",
    judge_prompt: str = "gepa-v1",
    dashboard: bool = True,
    tags: str = "",
    metrics: bool = True,
    lr: float = 1e-6,
    dynamic_sampling: bool = False,
    save_interval: int = 5,
    std_normalization: bool = True,
    gpus: int = N_GPUS,
    prompts_per_step: int = 16,
    judge_concurrency: int = JUDGE_CONCURRENCY,
    judge_gpus: int = 0,
) -> str:
    """`judge_gpus` > 0: `judge` is a self-hosted model (JUDGE_MODELS), served by
    SGLang on the container's last `judge_gpus` GPUs; Miles gets the first
    `gpus`."""
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
        "RUBRICRL_PROMPT": judge_prompt,
        "RUBRICRL_CACHE": str(cache),
        "RUBRICRL_LOG": str(run_dir / "rewards.jsonl"),
        "RUBRICRL_ENV_URL": f"http://127.0.0.1:{ENV_PORT}",
        "RUBRICRL_ENV_MODE": env_mode,
        "WANDB_TAGS": tags,
        # for the running cost in W&B (rubricrl.rewards.miles_log)
        "RUBRICRL_RUN_START": str(time.time()),
        "RUBRICRL_GPUS": str(gpus + judge_gpus),
        "RUBRICRL_CONCURRENCY": str(judge_concurrency),
        "RUBRICRL_GPU_USD_PER_HOUR": str(GPU_USD_PER_HOUR),
    }
    if judge_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(gpus)))  # Ray, Miles
        env["RUBRICRL_JUDGE_URL"] = f"http://127.0.0.1:{JUDGE_PORT}"
    args = training_args(
        task, model, run_dir, num_rollout, smoke_test, reward, env_mode,
        lr=lr, dynamic_sampling=dynamic_sampling, save_interval=save_interval,
        std_normalization=std_normalization, gpus=gpus,
        prompts_per_step=prompts_per_step,
    )  # fmt: skip
    (run_dir / "args.txt").write_text(shlex.join(args) + "\n")
    data_volume.reload()
    judge_server = (
        start_judge_server(
            judge, list(range(gpus, gpus + judge_gpus)), run_dir / "judge_server.log"
        )
        if judge_gpus
        else None
    )
    sidecar = start_env_server(env, run_dir) if reward == "env" else None
    stack = contextlib.ExitStack()  # tunnels, Prometheus, Grafana
    stop = threading.Event()

    def commit_periodically() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                runs_volume.commit()
            except Exception as e:  # never fail the run over a sync point
                print(f"periodic Volume commit failed: {e}", flush=True)

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        links = open_dashboard(stack, env, metrics) if dashboard else {}
        start_ray(ip, env, gpus, 16, dashboard)
        start_metrics_stack(stack, links, run_dir)
        # A fresh process, as Miles' own: Ray reads RAY_AUTH_MODE when it is
        # imported, so a client in this process (imported before start_ray
        # turned token auth on) is refused by the cluster.
        subprocess.run(["ray", "status"], env=env, check=True, timeout=300)
        if links:
            print_dashboard(links)
        t0 = time.monotonic()
        subprocess.run(
            ["python", str(MILES / "train.py"), *args], cwd=MILES, env=env, check=True
        )
        print(f"training took {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    finally:
        stop.set()
        stack.close()
        if sidecar is not None:
            sidecar.terminate()
        if judge_server is not None:
            judge_server.terminate()
        if cache.exists():
            shutil.copy(cache, run_dir / "judge.sqlite")
        runs_volume.commit()
    return str(run_dir)


def open_dashboard(stack: contextlib.ExitStack, env: dict, metrics: bool) -> dict:
    """Tunnels to the Ray dashboard and (`metrics`) Grafana, opened before `ray
    start`: the dashboard reads from its environment where Grafana and Prometheus
    are, and the URL the browser loads Grafana's panels from."""
    links = {"ray": stack.enter_context(modal.forward(DASHBOARD_PORT)).url}
    if metrics:
        links["grafana"] = stack.enter_context(modal.forward(GRAFANA_PORT)).url
        env |= {
            "RAY_GRAFANA_HOST": f"http://localhost:{GRAFANA_PORT}",
            "RAY_PROMETHEUS_HOST": f"http://localhost:{PROMETHEUS_PORT}",
            "RAY_GRAFANA_IFRAME_HOST": links["grafana"].rstrip("/"),
        }
    return links


def start_metrics_stack(
    stack: contextlib.ExitStack, links: dict, log_dir: Path
) -> None:
    """Prometheus + Grafana on the configs Ray's dashboard generated, if `links`
    has a Grafana tunnel; both stop with `stack`.

    Prometheus listens on localhost only: just the dashboard and Grafana query it.
    Grafana is public through its tunnel (the browser loads its panels into the
    Ray dashboard), so it serves Ray's dashboards to anonymous *viewers* only:
    the admin login form is disabled and the admin password is random. What it
    shows is cluster metrics (CPU, GPU, memory, Ray tasks), no secrets.
    """
    if "grafana" not in links:
        return
    prometheus_yml = RAY_METRICS_DIR / "prometheus" / "prometheus.yml"
    grafana_ini = RAY_METRICS_DIR / "grafana" / "grafana.ini"
    deadline = time.time() + 60  # Ray writes them asynchronously
    while not (prometheus_yml.exists() and grafana_ini.exists()):
        if time.time() > deadline:
            raise TimeoutError(f"Ray wrote no metrics configs in {RAY_METRICS_DIR}")
        time.sleep(1)
    commands = {
        "prometheus": [
            "/opt/prometheus/prometheus", f"--config.file={prometheus_yml}",
            "--storage.tsdb.path=/tmp/prometheus",
            f"--web.listen-address=127.0.0.1:{PROMETHEUS_PORT}",
        ],
        "grafana": [
            "/opt/grafana/bin/grafana", "server", "--homepath", "/opt/grafana",
            "--config", str(grafana_ini),
        ],
    }  # fmt: skip
    grafana_env = {
        **os.environ,
        "GF_SERVER_ROOT_URL": links["grafana"],
        "GF_SERVER_HTTP_PORT": str(GRAFANA_PORT),
        "GF_SECURITY_ADMIN_PASSWORD": secrets.token_hex(16),
        "GF_AUTH_DISABLE_LOGIN_FORM": "true",
        # Ray's guidance for embedding Grafana in the dashboard's iframes.
        "GF_SECURITY_COOKIE_SECURE": "true",
        "GF_SECURITY_COOKIE_SAMESITE": "none",
        "GF_PATHS_DATA": "/tmp/grafana",
        "GF_ANALYTICS_REPORTING_ENABLED": "false",
        "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
    }
    for name, command in commands.items():
        log = stack.enter_context((log_dir / f"{name}.log").open("w"))
        proc = subprocess.Popen(
            command,
            env=grafana_env if name == "grafana" else None,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        stack.callback(proc.terminate)


def start_ray(ip: str, env: dict, gpus: int, cpus: int, dashboard: bool) -> None:
    """The Ray head. With `dashboard`, the dashboard listens on every interface (for
    a `modal.forward` tunnel) behind Ray's token auth: a fresh token per run in
    RAY_TOKEN, which every Ray client in this container (Miles, `ray status`)
    reads; a request to the tunnel without it is refused."""
    flags = []
    if dashboard:
        RAY_TOKEN.parent.mkdir(parents=True, exist_ok=True)
        RAY_TOKEN.write_text(secrets.token_urlsafe(32))
        RAY_TOKEN.chmod(0o600)
        os.environ["RAY_AUTH_MODE"] = env["RAY_AUTH_MODE"] = "token"
        flags = ["--dashboard-host=0.0.0.0", f"--dashboard-port={DASHBOARD_PORT}"]
    subprocess.run(
        [
            "ray", "start", "--head", "--port=6379", f"--node-ip-address={ip}",
            f"--num-gpus={gpus}", f"--num-cpus={cpus}", "--disable-usage-stats",
            *flags,
        ],
        env=env,
        check=True,
    )  # fmt: skip


def print_dashboard(links: dict) -> None:
    """The tunnel URLs, and how to get the token (never printed itself)."""
    task = os.environ.get("MODAL_TASK_ID", "CONTAINER_ID")
    grafana = f"Grafana: {links['grafana']} (view-only)\n" if "grafana" in links else ""
    print(
        f"Ray dashboard: {links['ray']}\n{grafana}"
        # pbcopy: the file has no trailing newline, so zsh shows a `%` after it
        # on screen, which is easy to copy by mistake. sleep: piped, `modal
        # container exec` drops the output of a command that exits at once
        # (about half the time, measured).
        f"  token: modal container exec {task} -- sh -c 'cat {RAY_TOKEN}; sleep 1'"
        " | pbcopy\n"
        f"  shell: modal container exec {task} ray status",
        flush=True,
    )


def start_env_server(env: dict, run_dir: Path) -> subprocess.Popen:
    """The consultation environment as a sidecar on localhost, from its own venv.

    It holds the judge (and the patient simulator), so the judge cache and the
    Weave traces (one per finished episode, project rl-finetune) come from here.
    """
    server_env = env | {
        "PYTHONPATH": "/root",  # rubricrl's source; nothing of Miles'
        "RUBRICRL_DATA": str(HB),
        "RUBRICRL_WEAVE": "rl-finetune",
        "RUBRICRL_TRACE_RUN": run_dir.name,
    }
    log = (run_dir / "env_server.log").open("w")
    proc = subprocess.Popen(
        [
            f"{ENV_VENV}/bin/python", "-m", "uvicorn", "rubricrl.env.server:app",
            "--host", "127.0.0.1", "--port", str(ENV_PORT), "--log-level", "warning",
        ],
        env=server_env, stdout=log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"env server exited; see {run_dir}/env_server.log")
        try:
            with socket.create_connection(("127.0.0.1", ENV_PORT), timeout=1):
                print(f"env server up on :{ENV_PORT}", flush=True)
                return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("env server did not start in 120 s")


def start_judge_server(
    model: str, gpus: list[int], log_path: Path, timeout_s: int = 30 * MINUTES
) -> subprocess.Popen:
    """A self-hosted judge: SGLang's OpenAI-compatible server for `model` on
    `gpus` (device indices), on localhost:JUDGE_PORT (`judge.client.
    OpenAICompatJudge`): tensor-parallel over the GPUs the model needs
    (JUDGE_MODELS), data-parallel over any more. gpt-oss: its reasoning goes to a
    separate field, so the reply holds only the verdict."""
    tp = JUDGE_MODELS[model][2]
    if len(gpus) % tp:
        raise ValueError(f"{model}: a multiple of {tp} GPUs")
    flags = ["--reasoning-parser", "gpt-oss"] if model.startswith("gpt-oss") else []
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            "python", "-m", "sglang.launch_server",
            "--model-path", str(DATA / "models" / model), "--served-model-name", model,
            "--tp", str(tp), "--dp-size", str(len(gpus) // tp),
            "--host", "127.0.0.1", "--port", str(JUDGE_PORT),
            "--mem-fraction-static", "0.85", "--context-length", "16384", *flags,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))},
        stdout=log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    import urllib.request

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"judge server exited:\n{log_path.read_text()[-3000:]}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{JUDGE_PORT}/health", timeout=5
            ):
                up = time.monotonic() - t0
                print(f"judge {model} up on :{JUDGE_PORT} in {up:.0f}s", flush=True)
                return proc
        except OSError:
            time.sleep(5)
    proc.terminate()
    raise RuntimeError(f"judge server not up in {timeout_s} s; see {log_path}")


@app.local_entrypoint()
def main(
    task: str = "healthbench",
    model: str = "qwen3.5-4b",
    num_rollout: int = 60,
    smoke_test: bool = False,
    judge: str = "haiku",
    judge_effort: str = "low",
    judge_thinking: bool = False,
    judge_prompt: str = "gepa-v1",
    reward: str = "rm",
    env_mode: str = "single",
    dashboard: bool = True,
    metrics: bool = True,
    lr: float = 1e-6,
    dynamic_sampling: bool = False,
    save_interval: int = 5,
    std_normalization: bool = True,
    gpus: int = N_GPUS,
    prompts_per_step: int = 16,
    judge_concurrency: int = JUDGE_CONCURRENCY,
    judge_model: str = "",
    judge_gpus: int = 1,
):
    """--task healthbench|math, --model qwen3.5-4b|qwen3-4b, --judge haiku|sonnet,
    --reward rm|env (env: episodes against rubricrl.env), --env-mode single|patient,
    --judge-prompt gepa-v1|simple-evals-healthbench-v1 (the grader template),
    --no-dashboard (no Ray dashboard tunnel; by default it is on, behind a token,
    with Prometheus + Grafana for its charts), --no-metrics (dashboard only),
    --lr (default 1e-6), --dynamic-sampling (drop groups with no score spread),
    --save-interval (checkpoint every N steps, default 5),
    --no-std-normalization (Dr. GRPO advantages: reward - group mean only),
    --gpus 2|4 (H100s, default 4), --prompts-per-step (default 16, x 8 replies),
    --judge-concurrency (judge requests in flight, default 256),
    --judge-model gpt-oss-120b|... (a self-hosted judge, JUDGE_MODELS, instead of
    --judge; on --judge-gpus more H100s, default 1, next to the --gpus)."""
    if task not in ("healthbench", "math") or model not in MODELS:
        raise SystemExit(f"--task healthbench|math, --model one of {sorted(MODELS)}")
    if reward not in ("rm", "env") or env_mode not in ("single", "patient"):
        raise SystemExit("--reward rm|env, --env-mode single|patient")
    from rubricrl.judge.prompt import TEMPLATES

    if judge_prompt not in TEMPLATES:
        raise SystemExit(f"--judge-prompt one of {sorted(TEMPLATES)}")
    prefix = "consult" if reward == "env" and env_mode == "patient" else "healthbench"
    if (
        task == "healthbench"
        and not (ROOT / f"data/miles/{prefix}_train.jsonl").exists()
    ):
        hint = (
            "python -m rubricrl.sim.scenarios build --split train"
            if prefix == "consult"
            else "python -m rubricrl.data.healthbench export-miles"
        )
        raise SystemExit(f"data/miles/{prefix}_train.jsonl missing: run `{hint}`")
    if gpus not in (1, 2, 4, 8):
        raise SystemExit("--gpus 1|2|4|8")
    if judge_model and judge_model not in JUDGE_MODELS:
        raise SystemExit(f"--judge-model one of {sorted(JUDGE_MODELS)}")
    if judge_model and (task != "healthbench" or env_mode != "single"):
        raise SystemExit("--judge-model: healthbench, single mode (no patient model)")
    if judge_model:
        judge, judge_gpus = judge_model, max(judge_gpus, JUDGE_MODELS[judge_model][2])
        if judge_gpus % JUDGE_MODELS[judge_model][2]:
            raise SystemExit(f"--judge-gpus: a multiple of {JUDGE_MODELS[judge][2]}")
    else:
        judge_gpus = 0
    if smoke_test:
        num_rollout = 2
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{task}-{uuid.uuid4().hex[:6]}"
    print(f"Run ID: {run_id}")
    tags = [task, f"reward:{reward}", f"model:{model}"]
    if task == "healthbench":
        thinking = "-thinking" if judge_thinking else ""
        tags += [f"judge:{judge}-{judge_effort}{thinking}", f"prompt:{judge_prompt}"]
        tags += [f"self-hosted-judge:{judge_gpus}gpu"] if judge_gpus else []
        tags += [f"mode:{env_mode}"] if reward == "env" else []
    tags += [f"lr:{lr:g}"] + (["dynamic-sampling"] if dynamic_sampling else [])
    tags += [] if std_normalization else ["no-std-norm"]
    tags += [f"gpus:{gpus}", f"prompts:{prompts_per_step}"]
    tags += ["smoke"] if smoke_test else []
    download.remote(
        [model, *([judge] if judge_gpus else [])],
        ["dapo-math-17k", "aime-2024"] if task == "math" else [],
    )
    # The GPU count is part of the function's resources: set it per run.
    result = (
        train.with_options(
            # a judge's SGLang schedulers need CPUs of their own next to Miles'
            gpu=f"{GPU}:{gpus + judge_gpus}",
            cpu=16 + 4 * judge_gpus,
            memory=(128 + 32 * judge_gpus) * 1024,
        )
        .spawn(
            run_id,
            task,
            model,
            num_rollout,
            smoke_test,
            judge,
            judge_effort,
            judge_thinking,
            reward=reward,
            env_mode=env_mode,
            judge_prompt=judge_prompt,
            dashboard=dashboard,
            tags=",".join(tags),
            metrics=metrics,
            lr=lr,
            dynamic_sampling=dynamic_sampling,
            save_interval=save_interval,
            std_normalization=std_normalization,
            gpus=gpus,
            prompts_per_step=prompts_per_step,
            judge_concurrency=judge_concurrency,
            judge_gpus=judge_gpus,
        )
        .get()
    )
    print(f"Results in Volume {runs_volume.name}: {result}")


@app.function(
    image=image,
    cpu=8,
    memory=64 * 1024,
    volumes={DATA: data_volume, RUNS: runs_volume},
    timeout=30 * MINUTES,
)
def export(run_id: str, iteration: int = 0, model: str = "qwen3.5-4b") -> str:
    """FSDP checkpoint -> HF weights at RUN_ID/huggingface/iter_XXXXXXX (0 = latest).

    Not Miles' `tools/convert_fsdp_to_hf.py`: for Qwen3.5 it builds a text-only
    model (`model.layers...`) while the checkpoint holds the full vision-language
    model (`model.language_model.layers...`), matches 1 of 724 tensors, and loads
    with `strict=False`, so it silently saves randomly initialised weights (the
    first export of run 11a3f8 sampled gibberish). Here the checkpoint's own names
    are kept, which are the base checkpoint's, and a missing text-model tensor is
    an error rather than a silent default."""
    import importlib.util

    checkpoints = RUNS / run_id / "checkpoints"
    if iteration == 0:
        iteration = int((checkpoints / "latest_checkpointed_iteration.txt").read_text())
    checkpoint = checkpoints / f"iter_{iteration:07d}"
    output = RUNS / run_id / "huggingface" / checkpoint.name
    spec = importlib.util.spec_from_file_location(
        "convert_fsdp_to_hf", MILES / "tools/convert_fsdp_to_hf.py"
    )
    miles_convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(miles_convert)  # its checkpoint reader only
    state = miles_convert._load_fsdp_state_dict(
        miles_convert._detect_model_dir(str(checkpoint))
    )
    trained = {
        k: v
        for k, v in state.items()
        if not k.startswith("optimizer") and hasattr(v, "dtype")
    }
    write_base_layout(trained, output, DATA / "models" / model)
    runs_volume.commit()
    print(f"Hugging Face model saved to {output}")
    return str(output.relative_to(RUNS))


CHECKPOINT_PREFIXES = ("model_state.model.", "model_state.", "")


def base_name(key: str, base_keys: set[str]) -> str:
    """A checkpoint tensor's name in the base Hugging Face checkpoint: the FSDP
    checkpoint's `model_state.model.` prefix dropped; a text-only name
    (`model.layers...`) nested under `model.language_model.` if that is where the
    base (a vision-language model) keeps it. Unchanged if nothing matches."""
    for prefix in CHECKPOINT_PREFIXES:
        if not key.startswith(prefix):
            continue
        name = key.removeprefix(prefix)
        nested = "model.language_model." + name.removeprefix("model.")
        for candidate in (name, nested):
            if candidate in base_keys:
                return candidate
    for prefix in CHECKPOINT_PREFIXES[:-1]:  # no match: still drop the prefix
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return key


def torch_equal(a, b) -> bool:
    import torch

    return a.shape == b.shape and bool(torch.equal(a.float(), b.float()))


def write_base_layout(trained: dict, output: Path, base: Path) -> None:
    """The trained tensors as a Hugging Face checkpoint laid out like `base`.

    Names follow the base (`base_name`), dtypes too (Qwen3.5 keeps the
    linear-attention `A_log` and norms in float32), and base tensors the trainer
    doesn't hold (the multi-token-prediction head) are copied unchanged. Every
    text-model tensor must come from the checkpoint: a missing one raises, so an
    export can't silently fall back to untrained or random weights. The base's
    config, tokenizer and template files are copied alongside."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_keys: set[str] = set()
    for path in sorted(base.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            base_keys |= set(f.keys())  # noqa: SIM118 (safe_open has no __iter__)
    renamed = {base_name(k, base_keys): v for k, v in trained.items()}
    tied = renamed.get("lm_head.weight")
    embed = renamed.get("model.language_model.embed_tokens.weight")
    if "lm_head.weight" not in base_keys and tied is not None and embed is not None:
        # The base ties the output layer to the embeddings (no lm_head stored);
        # the trainer saves the shared tensor under both names.
        if not torch_equal(tied, embed):
            raise RuntimeError("lm_head differs from embed_tokens: not tied")
        del renamed["lm_head.weight"]
    unknown = sorted(set(renamed) - base_keys)
    missing_text = sorted(
        k for k in base_keys - set(renamed) if k.startswith("model.language_model.")
    )
    if unknown or missing_text:
        raise RuntimeError(
            f"checkpoint doesn't match {base.name}: {len(unknown)} unknown "
            f"(e.g. {unknown[:3]}), {len(missing_text)} text-model tensors missing "
            f"(e.g. {missing_text[:3]})"
        )
    out = {}
    for path in sorted(base.glob("*.safetensors")):
        for key, tensor in load_file(str(path)).items():
            out[key] = renamed[key].to(tensor.dtype) if key in renamed else tensor
    print(
        f"export: {len(renamed)} trained tensors, "
        f"{len(out) - len(renamed)} copied from {base.name} (not in the checkpoint)"
    )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for path in base.iterdir():
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name != ("model.safetensors.index.json")
        ):
            shutil.copy(path, output / path.name)
    save_file(out, str(output / "model.safetensors"), metadata={"format": "pt"})


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


@app.function(
    image=image,
    cpu=16,
    memory=32 * 1024,
    volumes={DATA: data_volume},
    timeout=20 * MINUTES,
)
def session_check(model: str = "qwen3.5-4b", limit_s: int = 600) -> dict:
    """CPU check of Miles' session server (`--use-session-server`) on its own: the
    command Miles launches, with this model's TITO settings; how long until it
    accepts connections (Miles waits 120 s), and its output if it dies."""
    config = {
        "host": "127.0.0.1", "port": 20007, "instance_id": "check-0",
        "backend_url": "http://127.0.0.1:20008", "timeout": None,
        "hf_checkpoint": str(DATA / "models" / model),
        "chat_template_path": str(
            MILES / "miles/utils/chat_template_utils/templates/qwen3.5_fixed.jinja"
        ),
        "tito_model": TITO[model],
        "apply_chat_template_kwargs": {"preserve_thinking": True},
        "use_rollout_routing_replay": False, "use_rollout_indexer_replay": False,
        "use_sampling_support_replay": False, "sglang_speculative_algorithm": None,
        "num_layers": 32, "moe_router_topk": None, "save_debug_trajectory_data": None,
        "lora_rank": 0, "lora_adapter_path": None, "lora_train_only": False,
        "use_session_server": True, "session_message_matcher": "strict",
        "pause_generation_mode": "retract",
        "session_sample_picker_path":
            "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        "session_sample_postprocessor_path":
            "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    }  # fmt: skip
    t0 = time.time()
    proc = subprocess.Popen(
        ["/opt/sglang/bin/python", "-m", "miles.rollout.session.server",
         "--config-json", json.dumps(config)],
        cwd=MILES, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    ready = None
    while time.time() - t0 < limit_s and proc.poll() is None:
        try:
            socket.create_connection(("127.0.0.1", 20007), timeout=1).close()
            ready = round(time.time() - t0, 1)
            break
        except OSError:
            time.sleep(1)
    exited = proc.poll()
    proc.terminate()
    output = proc.communicate(timeout=30)[0]
    print(output[-6000:], flush=True)
    return {"ready_after_s": ready, "exited_with": exited}


RAY_CLIENT_SCRIPT = """
import ray
ray.init()


@ray.remote
def ping():
    return "ok"


print(ray.get(ping.remote()))
"""


@app.function(image=image, cpu=4, memory=8 * 1024, timeout=30 * MINUTES)
def ray_check(hold_s: int = 0) -> dict:
    """CPU check of the training run's `--dashboard` setup, in Miles' image:

    - Ray: token auth on; a Ray client in another process (as Miles') runs a
      task; `ray status` works; the dashboard tunnel refuses requests without
      the token and serves them with it.
    - Metrics: Prometheus scrapes Ray's metrics; Grafana serves Ray's dashboards
      through its tunnel to anonymous viewers, read-only; the Ray dashboard
      reaches both (its charts).

    `--hold-s N` keeps it up N seconds to log in yourself."""
    import urllib.error
    import urllib.request

    import ray

    def get(url: str, token: str | None = None, data: bytes | None = None):
        """(status, body) of a GET (POST with `data`); never raises."""
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, ""
        except OSError as e:
            return type(e).__name__, ""

    def wait(check, timeout_s: float = 120) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if check():
                return True
            time.sleep(3)
        return False

    ip = socket.gethostbyname(socket.gethostname())
    env = {**os.environ, "RAY_ADDRESS": f"{ip}:6379"}
    log_dir = Path("/tmp/metrics-logs")
    log_dir.mkdir(exist_ok=True)
    with contextlib.ExitStack() as stack:
        links = open_dashboard(stack, env, metrics=True)
        start_ray(ip, env, 0, 4, dashboard=True)
        start_metrics_stack(stack, links, log_dir)
        token = RAY_TOKEN.read_text().strip()
        run = {"capture_output": True, "text": True, "env": env, "timeout": 300}
        client = subprocess.run(["python", "-c", RAY_CLIENT_SCRIPT], **run)
        status = subprocess.run(["ray", "status"], **run)
        ray_url, grafana = links["ray"].rstrip("/"), links["grafana"].rstrip("/")
        prom = f"http://127.0.0.1:{PROMETHEUS_PORT}"
        out = {
            "ray": ray.__version__,
            "client_task": client.stdout.strip()[-200:] or client.stderr[-2000:],
            "ray_status_ok": status.returncode == 0,
            "container": os.environ.get("MODAL_TASK_ID"),
        }
        wait(lambda: isinstance(get(f"{ray_url}/api/version", token)[0], int))
        out["dashboard"] = {
            "no_token": get(f"{ray_url}/api/version")[0],
            "wrong_token": get(f"{ray_url}/api/version", "not-the-token")[0],
            "token": get(f"{ray_url}/api/version", token)[0],
        }

        def ray_series() -> int:
            q = "/api/v1/query?query=ray_node_cpu_count"
            code, body = get(prom + q)
            return len(json.loads(body)["data"]["result"]) if code == 200 else 0

        out["prometheus"] = {
            "ready": get(f"{prom}/-/ready")[0],
            "scrapes_ray": wait(lambda: ray_series() > 0),
        }
        wait(lambda: get(f"{grafana}/api/health")[0] == 200)
        code, body = get(f"{grafana}/api/search?type=dash-db")
        titles = [d["title"] for d in json.loads(body)] if code == 200 else []
        out["grafana"] = {
            "health": get(f"{grafana}/api/health")[0],
            "anonymous_dashboards": titles,
            "anonymous_write": get(
                f"{grafana}/api/dashboards/db", data=b'{"dashboard": {"title": "x"}}'
            )[0],  # refused: viewers only
        }
        out["dashboard_sees"] = {
            k: get(f"{ray_url}/api/{k}_health", token)[0]
            for k in ("grafana", "prometheus")
        }
        print(json.dumps(out, indent=1), flush=True)
        for name in ("prometheus", "grafana"):
            tail = (log_dir / f"{name}.log").read_text()[-1500:]
            print(f"--- {name}.log (tail)\n{tail}", flush=True)
        if hold_s:
            print_dashboard(links)
            print(f"holding for {hold_s} s", flush=True)
            time.sleep(hold_s)
    subprocess.run(["ray", "stop"], env=env, check=False)
    return out


# Runs in Miles' Python (the agent function's): argv = row JSON path, mode, port.
ENV_CHECK_SCRIPT = """
import asyncio, json, sys
import websockets
from rubricrl.env import miles_agent
from rubricrl.env.wire import Session
from rubricrl.rewards import miles_rm

row = json.load(open(sys.argv[1]))
mode, port = sys.argv[2], sys.argv[3]
pid = row["metadata"]["prompt_id"]


async def doctor(history):
    return "Could you tell me more? If it gets worse, please seek urgent care."


async def main():
    prompt = miles_rm._judge().config.prompt_version  # the in-process reward's
    async with Session(f"http://127.0.0.1:{port}") as env:
        result, turns = await miles_agent.episode(
            env, doctor, row["prompt"], pid, mode, "train"
        )
    rec = miles_agent.record(result.observation, pid, False, turns)
    print(json.dumps({
        "python": sys.version.split()[0], "websockets": websockets.__version__,
        "reward": rec["reward"], "turns": turns, "criteria": len(rec["criteria"]),
        "end_reason": rec["end_reason"], "reward_prompt": prompt,
    }))


asyncio.run(main())
"""


@app.function(
    image=image,
    cpu=4,
    memory=16 * 1024,
    secrets=[modal.Secret.from_name("anthropic"), modal.Secret.from_name("wandb")],
    timeout=30 * MINUTES,
)
def env_check(mode: str = "single") -> dict:
    """CPU check of `--reward env` minus the policy: the environment's venv builds
    in the Miles image, the sidecar starts, and Miles' own Python (the agent
    function's) plays one episode over the wire with a fixed doctor reply."""
    run_dir = Path("/tmp/env-check")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "RUBRICRL_CACHE": "/tmp/judge.sqlite"}
    proc = start_env_server(env, run_dir)
    prefix = "healthbench" if mode == "single" else "consult"
    row_path = run_dir / "row.json"
    row_path.write_text((HB / f"{prefix}_test.jsonl").read_text().splitlines()[0])
    try:
        out = subprocess.run(
            ["python", "-c", ENV_CHECK_SCRIPT, str(row_path), mode, str(ENV_PORT)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        print(out.stdout, out.stderr[-3000:], flush=True)
        if out.returncode:
            log = (run_dir / "env_server.log").read_text()[-3000:]
            raise RuntimeError(f"episode failed; env server log:\n{log}")
        return json.loads(out.stdout.strip().splitlines()[-1])
    finally:
        proc.terminate()


# Runs in Miles' Python: argv = Miles' training args (JSON), train rows, eval rows.
LOG_CHECK_SCRIPT = """
import asyncio, json, sys

argv, train_path, eval_path = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.argv = ["train.py", *argv]
from miles.utils.arguments import parse_args

args = parse_args()  # Miles' own parser and checks: an unknown flag fails here
from miles.ray.rollout import metrics
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample
from rubricrl.env import miles_agent

logged = []
tracking.log = lambda a, m, step_key: logged.append([step_key, dict(m)])
REPLIES = [
    "Please see a doctor.",
    "It is probably nothing; rest and drink water.",
    "Go to the emergency department now if you have chest pain or trouble breathing.",
    "I need more details: how long, how severe, and any other symptoms?",
    "Take ibuprofen 400 mg every 6 hours with food; see a doctor if it persists.",
    "This could be several things. Common causes include infection or strain. "
    "Watch for fever, worsening pain or new symptoms, and seek care if they appear.",
    "I am not able to help with that.",
    "Monitor it at home for two days; call your doctor if it does not improve.",
]


async def play(rows, k, is_eval):
    async def one(row, i):
        async def chat(base_url, messages, request_kwargs):
            return REPLIES[i % len(REPLIES)]

        miles_agent.chat = chat  # the policy: a fixed reply per sample
        md = dict(row["metadata"], eval=is_eval)
        out = await miles_agent.run("http://policy", row["prompt"], {}, md)
        s = Sample(group_index=g, index=g * k + i, prompt=row["prompt"],
                   response=REPLIES[i % len(REPLIES)], rollout_id=0)
        s.response_length = 100 + 50 * i
        s.loss_mask = [1] * s.response_length
        s.status = Sample.Status.TRUNCATED if i == 0 else Sample.Status.COMPLETED
        s.metadata = md | (out or {})
        s.reward = (out or {}).get("env", {}).get("reward", 0.0)
        s.non_generation_time = ((out or {}).get("agent_metrics") or {}).get(
            "total_tool_time", 0.0)
        return s

    samples = []
    for g, row in enumerate(rows):  # sequential: `chat` is patched per sample
        for i in range(k):
            samples.append(await one(row, i))
    return samples


def rows(path, n):
    return [json.loads(x) for x in open(path).read().splitlines()[:n]]


train = asyncio.run(play(rows(train_path, 2), args.n_samples_per_prompt, False))
evals = asyncio.run(play(rows(eval_path, 2), args.n_samples_per_eval_prompt, True))
metrics.log_rollout_data(0, args, train, None, 60.0)
metrics.log_eval_rollout_data(0, args, {"healthbench_val": {
    "rewards": [s.reward for s in evals], "samples": evals,
    "truncated": [s.status == Sample.Status.TRUNCATED for s in evals]}}, None)
print(json.dumps({
    "hooks": [args.custom_rollout_log_function_path,
              args.custom_eval_rollout_log_function_path],
    "wandb_random_suffix": args.wandb_random_suffix,
    "logged": logged,
}, default=str))
"""


@app.function(
    image=image,
    cpu=8,
    memory=32 * 1024,
    volumes={DATA: data_volume},
    secrets=[modal.Secret.from_name("anthropic"), modal.Secret.from_name("wandb")],
    timeout=30 * MINUTES,
)
def log_check(
    lr: float = 1e-6, dynamic_sampling: bool = False, std_normalization: bool = True
) -> dict:
    """CPU check of what a `--reward env` run logs per step, in Miles' image.

    Miles parses the real training arguments (so a bad flag fails here, not on 4
    GPUs); then 16 training and 4 eval episodes run against the environment with
    the real judge, a fixed reply standing in for the policy, and go through
    Miles' own rollout and eval logging, rubric hooks included. Prints every
    key and value W&B would get. The trainer's `train/*` keys need GPUs."""
    run_dir = Path("/tmp/log-check")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "RUBRICRL_CACHE": "/tmp/judge.sqlite",
        "RUBRICRL_ENV_URL": f"http://127.0.0.1:{ENV_PORT}",
        "RUBRICRL_ENV_MODE": "single",
    }
    proc = start_env_server(env, run_dir)
    argv = training_args(
        "healthbench", "qwen3.5-4b", run_dir, 20, False, "env",
        lr=lr, dynamic_sampling=dynamic_sampling, std_normalization=std_normalization,
    )  # fmt: skip
    try:
        out = subprocess.run(
            ["python", "-c", LOG_CHECK_SCRIPT, json.dumps(argv),
             str(HB / "healthbench_train.jsonl"), str(HB / "healthbench_val.jsonl")],
            capture_output=True, text=True, env=env, cwd=MILES, timeout=1500,
        )  # fmt: skip
    finally:
        proc.terminate()
    if out.returncode:
        print(out.stdout[-3000:], out.stderr[-6000:], flush=True)
        raise RuntimeError("log check failed")
    result = json.loads(out.stdout.strip().splitlines()[-1])
    for step_key, values in result["logged"]:
        print(f"--- logged against {step_key}")
        for k, v in sorted(values.items()):
            print(f"  {k} = {v}")
    print(json.dumps({k: v for k, v in result.items() if k != "logged"}), flush=True)
    return result


# --- Self-hosted judge calibration ---------------------------------------------


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=64 * 1024,
    volumes={DATA: data_volume, JUDGE: judge_volume, RUNS: runs_volume},
    secrets=[modal.Secret.from_name("wandb")],  # no API judge: no anthropic key
    timeout=4 * 60 * MINUTES,
)
def judge_calibration(
    model: str, runs: list[dict], git: dict, gpus: int = 0
) -> list[dict]:
    """`rubricrl.analysis.calibrate run` against a self-hosted judge: the model on
    this container's GPUs (SGLang), the calibration in the env venv (it has
    weave/wandb), logged to W&B and Weave like the Haiku runs (project
    rl-finetune). The verdict cache is per model and decoding mode, copied to
    local disk (SQLite's WAL isn't safe on a network filesystem) and back.

    A run with `bench`: one training step's worth of verdicts, timed alone (an
    empty cache, no Weave or W&B in the loop, not saved to the Volume). With
    `replay` (a run id): that run's first step re-graded (`analysis.replay`). With
    `evals` (sample-file names in /judge/evals/samples): each graded on the test set
    like `rubricrl.evals.run score`, records and summary returned."""
    gpus = gpus or JUDGE_MODELS[model][2]
    work = Path("/tmp/calibration")
    work.mkdir(parents=True, exist_ok=True)
    data_volume.reload()
    server = start_judge_server(model, list(range(gpus)), work / "judge_server.log")
    results = []
    try:
        probe_reasoning(model, runs[0]["effort"])
        for opts in runs:
            if names := opts.get("evals"):
                stored = JUDGE / "self-hosted" / f"{model}-evals.sqlite"
                cache = work / stored.name
                if stored.exists() and not cache.exists():
                    shutil.copy(stored, cache)
                version = opts["prompt_version"]
                for name in names:
                    run_name = f"{name}-{model}"
                    subprocess.run(
                        [f"{ENV_VENV}/bin/python", "-m", "rubricrl.evals.run", "score",
                         str(JUDGE / "evals" / "samples" / f"{name}.jsonl"),
                         "--judge", f"{model}@{version}" if version else model,
                         "--base-url", f"http://127.0.0.1:{JUDGE_PORT}",
                         "--concurrency", str(opts["concurrency"]),
                         "--cache", str(cache), "--root", str(JUDGE / "raw"),
                         "--out", str(work / "evals"), "--name", run_name],
                        env={**os.environ, "PYTHONPATH": "/root"}, check=True,
                    )  # fmt: skip
                    d = work / "evals" / run_name
                    results.append({
                        "evals": run_name,
                        "records": (d / "records.jsonl").read_text(),
                        "summary": (d / "summary.json").read_text(),
                    })  # fmt: skip
                    stored.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cache, stored)
                    judge_volume.commit()
                continue
            if run_id := opts.get("replay"):
                path = work / f"replay-{run_id}.json"
                subprocess.run(
                    [f"{ENV_VENV}/bin/python", "-m", "rubricrl.analysis.replay",
                     str(RUNS / run_id / "rewards.jsonl"),
                     "--data", str(HB / "healthbench_train.jsonl"), "--judge", model,
                     "--base-url", f"http://127.0.0.1:{JUDGE_PORT}",
                     "--effort", opts["effort"],
                     "--concurrency", str(opts["concurrency"]),
                     "--verdicts", str(STEP_VERDICTS), "--out", str(path)],
                    env={**os.environ, "PYTHONPATH": "/root"}, check=True,
                )  # fmt: skip
                results.append(json.loads(path.read_text()) | {"replay": run_id})
                continue
            mode = "constrained" if opts.get("constrained", True) else "free"
            bench = opts.get("bench", False)
            stored = JUDGE / "self-hosted" / f"{model}-{mode}.sqlite"
            cache = work / ("bench.sqlite" if bench else stored.name)
            if bench:
                cache.unlink(missing_ok=True)
            elif stored.exists() and not cache.exists():
                shutil.copy(stored, cache)
            serving = {
                "engine": "sglang", "gpus": gpus, "gpu_type": "H100",
                "tp": JUDGE_MODELS[model][2], "dp": gpus // JUDGE_MODELS[model][2],
                "usd_per_gpu_hour": GPU_USD_PER_HOUR,
                "reasoning_effort": opts["effort"] if "gpt-oss" in model else None,
            }  # fmt: skip
            out = work / ("bench" if bench else "out")
            cmd = [
                f"{ENV_VENV}/bin/python", "-m", "rubricrl.analysis.calibrate", "run",
                "--judge", model, "--base-url", f"http://127.0.0.1:{JUDGE_PORT}",
                "--n", str(opts["n"]), "--retest", str(opts["retest"]),
                "--effort", opts["effort"], "--concurrency", str(opts["concurrency"]),
                "--root", str(JUDGE / "raw"), "--out", str(out), "--cache", str(cache),
                "--serving", json.dumps(serving),
                "--constrained" if mode == "constrained" else "--unconstrained",
            ]  # fmt: skip
            if opts.get("prompt_version"):
                cmd += ["--prompt-version", opts["prompt_version"]]
            if not bench:
                cmd += ["--weave", "rl-finetune", "--wandb", "rl-finetune"]
            env = {**os.environ, **git, "PYTHONPATH": "/root"}  # rubricrl, no Miles
            subprocess.run(cmd, env=env, check=True)
            path = max(out.glob("*.json"), key=lambda p: p.stat().st_mtime)  # newest
            results.append(json.loads(path.read_text()) | {"bench": bench})
            if bench:
                continue
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cache, stored)
            (JUDGE / "calibration").mkdir(exist_ok=True)
            shutil.copy(path, JUDGE / "calibration" / path.name)
            judge_volume.commit()
    finally:
        server.terminate()
        print((work / "judge_server.log").read_text()[-2000:], flush=True)
    return results


PROBE_SCRIPT = """
import json, sys, httpx
from rubricrl.judge.prompt import VERDICT_SCHEMA
model, effort = sys.argv[1], sys.argv[2]
payload = {
    "model": model, "max_tokens": 1024, "temperature": 0,
    "messages": [{"role": "user", "content": "Is 7 a prime number? Reply in JSON: "
                  '{"explanation": "...", "criteria_met": true or false}'}],
    "response_format": {"type": "json_schema",
                        "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
}
if model.startswith("gpt-oss"):
    payload["reasoning_effort"] = effort
else:
    payload["chat_template_kwargs"] = {"enable_thinking": False}
r = httpx.post(f"{sys.argv[3]}/v1/chat/completions", json=payload, timeout=300).json()
m = r["choices"][0]["message"]
print("probe:", json.dumps({"reasoning_chars": len(m.get("reasoning_content") or ""),
      "content": m.get("content"), "usage": r.get("usage")}))
"""


def probe_reasoning(model: str, effort: str) -> None:
    """One forced-JSON request, to show in the log whether the model still
    reasons under the schema (gpt-oss: `reasoning_content`) and what it returns."""
    subprocess.run(
        [f"{ENV_VENV}/bin/python", "-c", PROBE_SCRIPT, model, effort,
         f"http://127.0.0.1:{JUDGE_PORT}"],
        env={**os.environ, "PYTHONPATH": "/root"}, check=False,
    )  # fmt: skip


# Grading one training step: 16 prompts x 8 replies x ~13 training criteria.
STEP_VERDICTS = 1700


@app.local_entrypoint()
def calibrate_judge(
    model: str = "gpt-oss-120b",
    n: int = 1000,
    retest: int = 200,
    prompts: str = "simple-evals-healthbench-v1,gepa-v1",
    effort: str = "low",
    concurrency: int = JUDGE_CONCURRENCY,
    constrained: bool = True,
    step_bench: bool = False,
    gpus: int = 0,
    replay: str = "",
    evals: str = "",
):
    """Calibrate a self-hosted judge against physicians, once per prompt in
    `--prompts`, on the same 1,000 meta-eval judgments as the Haiku runs.
    Writes analysis/out/calibration/<label>-n<n>.json and prints the gate:
    balanced F1, "met" on the longest quartile, seconds per training step.
    `--step-bench`: instead, time one step's STEP_VERDICTS verdicts per prompt, alone
    (to analysis/out/calibration/bench/; the calibration's own rate includes
    Weave tracing). `--replay RUN_ID`: instead, re-grade that run's first step
    with this judge (`rubricrl.analysis.replay`: time, agreement with the run's
    judge). `--evals NAME,...`: instead, grade runs/samples/NAME.jsonl on the test
    set with this judge and the first of `--prompts` (to runs/evals/NAME-MODEL;
    `rubricrl.evals.run merge` then pairs it with the Claude grades). `--gpus`:
    more than the model needs, as data parallelism.

        modal run modal/miles_rft.py::calibrate_judge --model gpt-oss-120b
        modal run modal/miles_rft.py::calibrate_judge --model gpt-oss-20b --n 20 \\
            --retest 4 --prompts gepa-v1          # the quick check
    """
    if model not in JUDGE_MODELS:
        raise SystemExit(f"--model one of {sorted(JUDGE_MODELS)}")
    from rubricrl.judge.prompt import PROMPT_VERSION

    runs = [
        {
            "n": n, "retest": retest, "effort": effort, "concurrency": concurrency,
            "constrained": constrained,
            "prompt_version": None if p == PROMPT_VERSION else p,
        }
        for p in prompts.split(",")
    ]  # fmt: skip
    if step_bench:
        runs = [r | {"n": STEP_VERDICTS, "retest": 0, "bench": True} for r in runs]
    if replay:
        runs = [runs[0] | {"replay": replay}]
    if evals:
        names = evals.split(",")
        runs = [runs[0] | {"evals": names}]
        _upload_eval_inputs(names)
    download.remote([model])
    gpus = gpus or JUDGE_MODELS[model][2]
    fn = judge_calibration.with_options(
        gpu=f"{GPU}:{gpus}", cpu=8 * gpus, memory=64 * 1024 * gpus
    )
    out = ROOT / "analysis" / "out" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench").mkdir(exist_ok=True)
    (out / "replay").mkdir(exist_ok=True)
    for r in fn.remote(model, runs, _git(), gpus):
        if r.get("evals"):
            d = ROOT / "runs" / "evals" / r["evals"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "records.jsonl").write_text(r["records"])
            (d / "summary.json").write_text(r["summary"])
            print(f"wrote {d}")
            continue
        if r.get("replay"):
            path = out / "replay" / f"{model}-{gpus}gpu-{r['replay']}.json"
            path.write_text(json.dumps(r, indent=1))
            print(
                f"{model} on {gpus} GPU(s), replaying {r['replay']}: {r['verdicts']}"
                f" verdicts in {r['wall_s']} s, failed {r['failed']}, agreement"
                f" {r['agreement']:.3f} (kappa {r['kappa']:.2f}), met"
                f" {r['rate_met']['replay']:.0%} vs {r['rate_met']['run']:.0%},"
                f" score r={r['score_pearson']:.2f} -> {path}"
            )
            continue
        name = f"{r['label']}-n{r['n']}" + (f"-{gpus}gpu" if r["bench"] else "")
        path = out / ("bench" if r["bench"] else "") / f"{name}.json"
        path.write_text(json.dumps(r, indent=1))
        t, longest = r["throughput"], r["by_length"][-1]
        step_s = STEP_VERDICTS / t["verdicts_per_s"] if t["verdicts_per_s"] else None
        if r["bench"]:
            print(
                f"{r['label']} step bench: {r['n']} verdicts in "
                f"{t['first_pass_s']} s, failed {r['failed']} -> {path}"
            )
            continue
        f1, kappa = r["judge"]["overall"]["f1_balanced"], r["retest"]["kappa"]
        print(
            f"{r['label']} n={r['n']}: balanced F1 {f1:.3f}, failed {r['failed']}, "
            f"met {r['judge_rate_met']:.0%} (longest quartile "
            f"{longest['judge_rate_met']:.0%} vs physicians "
            f"{longest['physician_rate_met']:.0%}), retest kappa {kappa}, "
            f"{t['verdicts_per_s']} verdicts/s (~{step_s:.0f} s per step), "
            f"GPU ${t['gpu_usd']} -> {path}"
        )
        print(f"  weave: {r.get('weave_url')}\n  wandb: {r.get('wandb_url')}")


def _upload_eval_inputs(names: list[str]) -> None:
    """The sample files, and HealthBench's conversations (the judge Volume has only
    the meta-eval), for `--evals`."""
    from rubricrl.data import healthbench

    raw = healthbench.SOURCES["oss_eval"].file
    have = {e.path for e in judge_volume.listdir("/raw")}
    with judge_volume.batch_upload(force=True) as batch:
        if f"raw/{raw}" not in have:
            batch.put_file(ROOT / healthbench.RAW / raw, f"/raw/{raw}")
        for name in names:
            local = ROOT / "runs" / "samples" / f"{name}.jsonl"
            batch.put_file(local, f"/evals/samples/{name}.jsonl")


def _git() -> dict:
    """This checkout's commit, for the W&B config (no git inside the container)."""

    def out(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout

    return {
        "RUBRICRL_GIT_SHA": out("git", "rev-parse", "HEAD").strip(),
        "RUBRICRL_GIT_DIRTY": "1"
        if out("git", "status", "--porcelain").strip()
        else "0",
    }
