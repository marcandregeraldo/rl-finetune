"""Check a Ray cluster before training on it: GPUs, versions, NCCL bandwidth.

One Ray actor per GPU in the cluster. The actors form one NCCL process group and
time an all-reduce, so a 2-node run shows whether traffic between nodes goes over
RDMA: look for `NET/IB` (RDMA) vs `NET/Socket` (TCP) in the NCCL INFO lines, and at
the bus bandwidth (same definition as nccl-tests). It also prints the Python, torch
and Ray versions the cluster runs, to compare with the Mac's (same uv.lock).

    python -m rubricrl.cluster_check [--config run.yaml]   # on any Ray cluster

Submitted as a Ray Job by `modal/ray_cluster.py --entry check`. Config keys
(optional): `mib` (all-reduce size, default 1024), `iters` (default 10).
"""

import platform
import socket
import time
from typing import Any

import click
import ray
import yaml


@ray.remote(num_gpus=1)
class _Rank:
    def address(self) -> tuple[str, int]:
        """This node's IP and a free port, for the process group's TCP store."""
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return ray.util.get_node_ip_address(), port

    def all_reduce(
        self, rank: int, world: int, addr: str, port: int, mib: int, iters: int
    ) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        dist.init_process_group(
            "nccl", init_method=f"tcp://{addr}:{port}", rank=rank, world_size=world
        )
        dev = torch.device("cuda", 0)  # Ray sets CUDA_VISIBLE_DEVICES per actor
        x = torch.ones(mib * 2**20 // 2, dtype=torch.bfloat16, device=dev)
        for _ in range(3):  # warm-up: NCCL builds its rings/trees on the first call
            dist.all_reduce(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        dist.destroy_process_group()
        size = x.numel() * x.element_size()
        return {
            "rank": rank,
            "node": ray.util.get_node_ip_address(),
            "gpu": torch.cuda.get_device_name(dev),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "ray": ray.__version__,
            },
            "ms": round(dt * 1e3, 2),
            # All-reduce moves 2 (n - 1) / n of the buffer per rank.
            "busbw_GBps": round(size / dt * 2 * (world - 1) / world / 1e9, 1),
        }


def main(cfg: dict[str, Any]) -> dict[str, Any]:
    world = int(ray.cluster_resources().get("GPU", 0))
    if world == 0:
        raise RuntimeError("no GPUs in the Ray cluster")
    env = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}
    ranks = [
        _Rank.options(runtime_env={"env_vars": env}).remote() for _ in range(world)
    ]
    addr, port = ray.get(ranks[0].address.remote())
    mib, iters = cfg.get("mib", 1024), cfg.get("iters", 10)
    rows = ray.get(
        [
            r.all_reduce.remote(i, world, addr, port, mib, iters)
            for i, r in enumerate(ranks)
        ]
    )
    nodes = sorted({r["node"] for r in rows})
    for r in rows:
        print(r)
    summary = {
        "nodes": len(nodes),
        "gpus": world,
        "gpu": rows[0]["gpu"],
        "versions": rows[0]["versions"],
        "all_reduce_MiB": mib,
        "busbw_GBps_min": min(r["busbw_GBps"] for r in rows),
    }
    print(summary)
    return summary


@click.command(help=__doc__.split("\n\n")[0])
@click.option(
    "--config", type=click.File(), help="YAML file (optional keys: mib, iters)"
)
def cli(config) -> None:
    cfg = yaml.safe_load(config) if config else {}
    ray.init()  # inside a Ray Job: joins the cluster it was submitted to
    main(cfg or {})


if __name__ == "__main__":
    cli()
