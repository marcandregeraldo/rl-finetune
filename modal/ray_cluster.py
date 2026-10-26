"""Ray clusters on Modal for RL experiments: one node (1-8 GPUs) or two 8-GPU nodes.

Follows Modal's multi-node training guide, Ray section
(https://modal.com/docs/guide/multi-node-training): rank 0 starts the Ray head and
submits the experiment as a **Ray Job**, then forwards its logs; other ranks start
Ray workers pointed at the head and stay alive while it runs. One node is the
same head with no workers (Clustered Functions must take whole 8-GPU nodes, so
1-GPU dev runs can't be clustered).

An experiment is a module run as `python -m <module> --config <yaml>` that calls
`ray.init()`, which joins this cluster. `--entry check` is
`nanorft.cluster_check` (GPUs, versions, NCCL bandwidth), `--entry train` is
`nanorft.driver` (M1), and any other value is taken as a module name.

Once: the Modal secrets `wandb` (WANDB_API_KEY) and `huggingface-secret`
(HF_TOKEN). Then (prefix with `uv run`):

    modal run modal/ray_cluster.py --entry check                      # 1 GPU
    modal run modal/ray_cluster.py --entry check --gpus 8             # NVLink
    modal run --detach modal/ray_cluster.py --entry check --nodes 2   # RDMA
    modal run --detach modal/ray_cluster.py --entry train \
        --config configs/<config>.yaml --gpus 1                       # after M1

`--dashboard` opens a public tunnel to the Ray dashboard (see `_head`).
"""

import asyncio
import shlex
import socket
import subprocess
import sys
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).parent.parent
APP_NAME = "rl-finetune-ray"
# "!" pins the GPU type: plain "H100" may be upgraded to an H200, and every
# number must say which hardware produced it.
GPU = "H100!"
CPU_PER_GPU = 4
MEM_PER_GPU_MB = 32 * 1024
N_NODES, GPUS_PER_NODE = 2, 8  # the clustered path: whole nodes only
TIMEOUT_S = 12 * 3600
RAY_PORT, DASHBOARD_PORT = 6379, 8265
RUNS = Path("/runs")
# The torch wheels in uv.lock are built for CUDA 13; the base image must not be
# newer than the host's CUDA. "devel" has the toolkit, for building kernels such
# as flash-linear-attention / causal-conv1d (Qwen3.5) later.
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"

ENTRIES = {"check": "nanorft.cluster_check", "train": "nanorft.driver"}

app = modal.App(APP_NAME)

# The Python series of .python-version (add_python takes "3.14", not a patch).
# Read only locally: containers re-import this file but use the built image.
PYTHON_SERIES = (
    ".".join((ROOT / ".python-version").read_text().strip().split(".")[:2])
    if modal.is_local()
    else None
)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_SERIES)
    # RDMA for NCCL between nodes: the verbs library and the mlx5 provider.
    .apt_install("libibverbs1", "ibverbs-providers")
    # Exactly what uv.lock pins (frozen), the same versions as the Mac. uv_sync
    # skips the project itself, so the package is added below.
    .uv_sync(str(ROOT), extras=["train"], extra_options="--no-dev", uv_version="0.13.5")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "PYTHONUNBUFFERED": "1",
            # Ray Jobs and their workers don't run from /root, where the package
            # is mounted: put it on the path for every process on every node.
            "PYTHONPATH": "/root",
            # No dashboard token: see the tunnel note in _head.
            "RAY_AUTH_MODE": "disabled",
        }
    )
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


def _ray(*args: str) -> None:
    """The `ray` CLI from this interpreter's environment (the uv venv)."""
    subprocess.run([str(Path(sys.executable).parent / "ray"), *args], check=True)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


async def _head(
    my_ip: str,
    gpus_per_node: int,
    n_nodes: int,
    entry: str,
    cfg: dict[str, Any],
    dashboard: bool,
) -> dict[str, Any]:
    """Start the Ray head, wait for every node, run the entry as a Ray Job."""
    import ray
    import yaml
    from ray.job_submission import JobStatus, JobSubmissionClient

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"  # recorded with the run's outputs
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    _ray(
        "start",
        "--head",
        f"--node-ip-address={my_ip}",
        f"--port={RAY_PORT}",
        "--dashboard-host=0.0.0.0",
        f"--dashboard-port={DASHBOARD_PORT}",
        f"--num-gpus={gpus_per_node}",
        # Explicit resources: Ray's auto-detection in a container can see the
        # host's CPUs, not this container's reservation.
        f"--num-cpus={CPU_PER_GPU * gpus_per_node}",
        "--disable-usage-stats",
    )
    try:
        ray.init(address="auto")
        deadline = time.monotonic() + 180
        want = gpus_per_node * n_nodes
        while (gpus := ray.cluster_resources().get("GPU", 0)) < want:
            if time.monotonic() > deadline:
                raise TimeoutError(f"only {gpus:.0f}/{want} GPUs joined in 180s")
            await asyncio.sleep(1)
        print(f"[ray] cluster ready: {ray.cluster_resources()}", flush=True)
        ray.shutdown()  # the Job is the driver; this process only watches it

        module = ENTRIES.get(entry, entry)
        cmd = shlex.join([sys.executable, "-m", module, "--config", str(config_path)])
        env = {"no_proxy": f"127.0.0.1,{my_ip}", "MASTER_ADDR": my_ip}
        client = JobSubmissionClient(f"http://127.0.0.1:{DASHBOARD_PORT}")
        job_id = client.submit_job(entrypoint=cmd, runtime_env={"env_vars": env})
        print(f"[ray] job {job_id}: {cmd}", flush=True)

        async with AsyncExitStack() as stack:
            if dashboard:
                # No dashboard token (RAY_AUTH_MODE=disabled in the image):
                # anyone with this URL can submit jobs, i.e. run code here and
                # read the W&B and HF secrets, while the run is alive.
                forward = modal.forward(DASHBOARD_PORT)
                tunnel = await stack.enter_async_context(forward)
                print(f"[ray] dashboard: {tunnel.url}", flush=True)
            async for line in client.tail_job_logs(job_id):
                print(line, end="", flush=True)

        status = client.get_job_status(job_id)
        if status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"Ray job {job_id} ended {status}")
        return {"run_id": cfg["run_id"], "job": job_id, "status": str(status)}
    finally:
        _ray("stop")  # closes the head's port: the workers see it and exit
        runs_vol.commit()  # keep outputs even if the job failed


async def _worker(my_ip: str, head_ip: str) -> None:
    """Join the head, then stay alive while it is up."""
    deadline = time.monotonic() + 180
    while not _port_open(head_ip, RAY_PORT):
        if time.monotonic() > deadline:
            raise TimeoutError(f"Ray head {head_ip}:{RAY_PORT} not up in 180s")
        await asyncio.sleep(1)
    _ray(
        "start",
        f"--node-ip-address={my_ip}",
        f"--address={head_ip}:{RAY_PORT}",
        f"--num-gpus={GPUS_PER_NODE}",
        f"--num-cpus={CPU_PER_GPU * GPUS_PER_NODE}",
        "--disable-usage-stats",
    )
    misses = 0
    while misses < 3:  # ~30 s without the head: the job is over
        await asyncio.sleep(10)
        misses = 0 if _port_open(head_ip, RAY_PORT) else misses + 1
    runs_vol.commit()


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
async def run_node(entry: str, cfg: dict[str, Any], dashboard: bool = False) -> dict:
    """One container: a Ray head with this container's GPUs and no workers."""
    import torch

    ip = socket.gethostbyname(socket.gethostname())
    return await _head(ip, torch.cuda.device_count(), 1, entry, cfg, dashboard)


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
async def run_cluster(
    entry: str, cfg: dict[str, Any], dashboard: bool = False
) -> dict | None:
    """Every container runs this: rank 0 is the Ray head, the others workers."""
    cluster = modal.Cluster.from_context()
    ips = cluster.container_ips(family="ipv4")  # Ray only works with IPv4
    rank = cluster.container_rank()
    if rank == 0:
        return await _head(ips[0], GPUS_PER_NODE, N_NODES, entry, cfg, dashboard)
    await _worker(ips[rank], ips[0])
    return None


@app.local_entrypoint()
def main(
    entry: str = "check",
    config: str = "",
    nodes: int = 1,
    gpus: int = 1,
    dashboard: bool = False,
) -> None:
    """Run one experiment on a fresh Ray cluster.

    --entry      check | train | a module name (run as `python -m`)
    --config     a YAML file, read here; the run gets a copy in its run_dir
    --nodes      1 (one container, --gpus of them) or 2 (two 8-GPU nodes)
    --gpus       GPUs on the single node (1-8); ignored when --nodes 2
    --dashboard  public tunnel to the Ray dashboard (see _head)
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
        result = run_cluster.spawn(entry, cfg, dashboard).get()
    else:
        raise SystemExit(f"--nodes must be 1 or {N_NODES}")
    print(f"[run] {run_id} done: {result}")
