# Training architecture

## One container per run

![One Modal container with four H100s holds Miles' driver, an SGLang engine and an FSDP trainer colocated on each GPU, Miles' router and session server, the environment sidecar, the Ray head, Prometheus and Grafana; the LLM API, W&B, the browser, two Volumes and the secrets sit outside](diagrams/training-container.svg)

`modal/miles_rft.py` runs GRPO with [Miles](https://github.com/radixark/miles) v0.1.1,
adapted from Modal's `miles_grpo` example. One Modal container holds the whole run:
- **GPUs:** 4 × H100, with SGLang and the FSDP trainer colocated on each.
- **Control:** Miles' driver and a single-node Ray head.
- **Environment:** the OpenEnv sidecar, running from its own venv.
- **Observability:** Prometheus and Grafana.

The reward is graded either in process (`--reward rm`) or through the environment
(`--reward env`). The details are in [`report.md`](report.md) §6.

**A self-hosted judge (run E):** `--judge-model gpt-oss-120b --judge-gpus 4`. The container
gets 8 H100s:
- **Miles and Ray** see GPUs 0–3.
- **The judge** runs on GPUs 4–7: an SGLang server with two TP2 replicas, reached on
  localhost.

The judge stays outside Miles' Ray cluster on purpose. Miles owns its placement groups and its
fault-tolerance controller, and the judge is a fixed-size, stateless server. Setup and gate:
[`self-hosted-judge/`](self-hosted-judge/README.md).

## Ray on Modal (cluster check)

![Ray on Modal: the Mac spawns a 2-node clustered function; rank 0 runs the Ray head and submits the experiment as a Ray Job, rank 1 joins as a worker](diagrams/ray-on-modal.svg)

[`modal/ray_cluster.py`](../modal/ray_cluster.py) brings up a Ray cluster on Modal: one node
with 1–8 H100s, or two 8-GPU nodes with RDMA. It submits
[`rubricrl.cluster_check`](../src/rubricrl/cluster_check.py) as a Ray Job, which times an
NCCL all-reduce across every GPU and prints versions. Training doesn't use this app: Miles
runs in its own image, with its own single-node Ray head.

```bash
uv run modal run modal/ray_cluster.py --entry check --gpus 8
uv run modal run --detach modal/ray_cluster.py --entry check --nodes 2
```
