"""Ray clusters on Modal: one node with 1-8 GPUs, or two clustered 8-GPU nodes.

Whatever runs on the cluster is an *entry*: a function `main(cfg: dict) -> dict`
that finds Ray already initialised and connected to every node. The code is the
same on a Mac (`ray.init()` locally) and here; this file only provides
infrastructure.

- 1 node (`--nodes 1 --gpus N`): one container, a local Ray instance sized to the
  container's reservation (the mila-logo-segmentation pattern).
- 2 nodes (`--nodes 2`): a Clustered Function with RDMA, after Modal's Miles GRPO
  example (https://modal.com/docs/examples/miles_grpo). Rank 0 starts the Ray head,
  rank 1 joins it and waits on a Queue until rank 0 is done. Rank 0 waits for all
  16 GPUs, then runs the entry. Clustered Functions must take whole nodes (8 GPUs).

Once: the Modal secrets `wandb` (WANDB_API_KEY) and `huggingface-secret`
(HF_TOKEN). Then (prefix with `uv run`):

    modal run modal/ray_cluster.py --entry check                 # 1 GPU
    modal run modal/ray_cluster.py --entry check --gpus 8        # NVLink
    modal run --detach modal/ray_cluster.py --entry check --nodes 2   # RDMA
    modal run --detach modal/ray_cluster.py --entry train \
        --config configs/<config>.yaml --gpus 1             # after M1

`--entry` is `check` (src/nanorft/cluster_check.py), `train`
(src/nanorft/driver.py), or any `module:function`. `--dashboard` opens a public
tunnel to the Ray dashboard.
"""

import importlib
import os
import socket
import subprocess
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).parent.parent
APP_NAME = "rl-finetune-ray"
# "!" pins the GPU type: plain "H100" may be upgraded to an H200, and every
# number in the journal must say which hardware produced it.
GPU = "H100!"
CPU_PER_GPU = 4
MEM_PER_GPU_MB = 32 * 1024
N_NODES, GPUS_PER_NODE = 2, 8  # the clustered path: whole nodes only
TIMEOUT_S = 12 * 3600
RAY_PORT, DASHBOARD_PORT = 6379, 8265
RUNS = Path("/runs")

ENTRIES = {"check": "nanorft.cluster_check:main", "train": "nanorft.driver:main"}

app = modal.App(APP_NAME)

# The exact local interpreter (patch included) from .python-version. Read only
# locally: containers re-import this file but use the already-built image.
PYTHON_VERSION = (
    (ROOT / ".python-version").read_text().strip() if modal.is_local() else None
)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    # RDMA for NCCL between nodes: the verbs library and the mlx5 provider. The
    # CUDA runtime and NCCL come with the torch wheel.
    .apt_install("libibverbs1", "ibverbs-providers")
    # Exactly what uv.lock pins (frozen), the same versions as the Mac. uv_sync
    # skips the project itself, so our packages are added below.
    .uv_sync(str(ROOT), extras=["train"], extra_options="--no-dev", uv_version="0.13.5")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("nanorft")
)

hf_vol = modal.Volume.from_name("rl-finetune-hf-cache", create_if_missing=True)
# Volume v2: both nodes write checkpoint shards to the same run directory.
runs_vol = modal.Volume.from_name("rl-finetune-runs", create_if_missing=True, version=2)
VOLUMES = {"/cache": hf_vol, str(RUNS): runs_vol}
SECRETS = [
    modal.Secret.from_name("wandb"),
    modal.Secret.from_name("huggingface-secret"),
]
# Rank 0 tells the other node when the run is over (one partition per run).
done_queue = modal.Queue.from_name("rl-finetune-ray-done", create_if_missing=True)


def _run_entry(entry: str, cfg: dict[str, Any]) -> dict[str, Any]:
    module, _, fn = ENTRIES.get(entry, entry).partition(":")
    return getattr(importlib.import_module(module), fn or "main")(cfg)


def _open_dashboard(stack: ExitStack) -> None:
    """Public tunnel to the Ray dashboard. Opens before Ray starts.

    No dashboard token: anyone with the URL can submit jobs, i.e. run code here
    and read this container's secrets (W&B, HF), while the run is alive. Keep it
    private, and leave it off unless you are watching the run.
    """
    os.environ["RAY_AUTH_MODE"] = "disabled"
    tunnel = stack.enter_context(modal.forward(DASHBOARD_PORT))
    print(f"[ray] dashboard: {tunnel.url}", flush=True)


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU_PER_GPU,
    memory=MEM_PER_GPU_MB,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=TIMEOUT_S,
    scaledown_window=2,  # Modal's minimum: no idle GPU after the run returns
)
def run_node(entry: str, cfg: dict[str, Any], dashboard: bool = False) -> dict:
    """One container, one local Ray instance with this container's GPUs."""
    import ray
    import torch

    n_gpus = torch.cuda.device_count()
    with ExitStack() as stack:
        if dashboard:
            _open_dashboard(stack)
        # Explicit resources: Ray's auto-detection in a container can see the
        # host's CPUs and RAM, not this container's reservation.
        ray.init(
            num_cpus=CPU_PER_GPU * n_gpus,
            num_gpus=n_gpus,
            include_dashboard=dashboard,
            dashboard_host="0.0.0.0",
            dashboard_port=DASHBOARD_PORT,
        )
        try:
            return _run_entry(entry, cfg)
        finally:
            ray.shutdown()
            runs_vol.commit()  # keep checkpoints even if the entry crashed


def _wait_for_port(host: str, port: int, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as err:
            if time.monotonic() > deadline:
                msg = f"Ray head {host}:{port} not up in {timeout_s}s"
                raise TimeoutError(msg) from err
            time.sleep(1)


def _wait_for_cluster(timeout_s: float = 180) -> dict:
    import ray

    deadline = time.monotonic() + timeout_s
    want = N_NODES * GPUS_PER_NODE
    while (gpus := ray.cluster_resources().get("GPU", 0)) < want:
        if time.monotonic() > deadline:
            raise TimeoutError(f"{gpus:.0f}/{want} GPUs joined in {timeout_s}s")
        time.sleep(1)
    return ray.cluster_resources()


@app.function(
    image=image,
    gpu=f"{GPU}:{GPUS_PER_NODE}",
    cpu=CPU_PER_GPU * GPUS_PER_NODE,
    memory=MEM_PER_GPU_MB * GPUS_PER_NODE,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=TIMEOUT_S,
    single_use_containers=True,  # Ray processes die with the container
)
@modal.clustered(size=N_NODES, rdma=True)
def run_cluster(
    entry: str, cfg: dict[str, Any], run_id: str, dashboard: bool = False
) -> dict | None:
    """Two containers, one Ray cluster. Every container runs this function."""
    import ray

    cluster = modal.Cluster.from_context()
    rank = cluster.container_rank()
    ips = cluster.container_ips(family="ipv4")  # Ray needs IPv4
    head = ips[0]
    start = [
        "ray",
        "start",
        f"--node-ip-address={ips[rank]}",
        f"--num-gpus={GPUS_PER_NODE}",
        f"--num-cpus={CPU_PER_GPU * GPUS_PER_NODE}",
        "--disable-usage-stats",
    ]
    if rank != 0:
        _wait_for_port(head, RAY_PORT)
        subprocess.run([*start, f"--address={head}:{RAY_PORT}"], check=True)
        # Stay alive (and in the cluster) until rank 0 reports the run is over.
        ok = done_queue.get(partition=run_id, timeout=TIMEOUT_S)
        runs_vol.commit()
        if not ok:
            raise RuntimeError("the entry failed on rank 0; see its logs")
        return None

    succeeded = False
    with ExitStack() as stack:
        if dashboard:
            _open_dashboard(stack)
        try:
            head_args = ["--head", f"--port={RAY_PORT}"]
            if dashboard:
                head_args += [
                    "--include-dashboard=true",
                    "--dashboard-host=0.0.0.0",
                    f"--dashboard-port={DASHBOARD_PORT}",
                ]
            subprocess.run([*start, *head_args], check=True)
            ray.init(address=f"{head}:{RAY_PORT}")
            print(f"[ray] cluster ready: {_wait_for_cluster()}", flush=True)
            result = _run_entry(entry, cfg)
            succeeded = True
            return result
        finally:
            done_queue.put(succeeded, partition=run_id)
            ray.shutdown()
            runs_vol.commit()


@app.local_entrypoint()
def main(
    entry: str = "check",
    config: str = "",
    nodes: int = 1,
    gpus: int = 1,
    dashboard: bool = False,
) -> None:
    """Run one entry on a fresh Ray cluster.

    --entry      check | train | module:function
    --config     a YAML file, read here and passed to the entry as a dict
    --nodes      1 (one container, --gpus of them) or 2 (two 8-GPU nodes)
    --gpus       GPUs on the single node (1-8); ignored when --nodes 2
    --dashboard  public tunnel to the Ray dashboard (see _open_dashboard)
    """
    import yaml

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    cfg = yaml.safe_load(Path(config).read_text()) if config else {}
    cfg |= {"run_id": run_id, "run_dir": str(RUNS / run_id)}
    print(f"[run] {run_id}: {entry} on {nodes} node(s)")
    if nodes == 1:
        fn = run_node.with_options(
            gpu=f"{GPU}:{gpus}", cpu=CPU_PER_GPU * gpus, memory=MEM_PER_GPU_MB * gpus
        )
        result = fn.spawn(entry, cfg, dashboard).get()
    elif nodes == N_NODES:
        result = run_cluster.spawn(entry, cfg, run_id, dashboard).get()
    else:
        raise SystemExit(f"--nodes must be 1 or {N_NODES}")
    print(f"[run] {run_id} done: {result}")
