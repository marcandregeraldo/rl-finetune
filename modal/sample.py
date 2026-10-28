"""Sample responses to the fixed HealthBench eval subset with SGLang, on Modal.

The same prompts, chat template (thinking off), sampling settings and K for every
model, base or trained, so `rubricrl.evals.run` compares like with like. Runs
SGLang's offline engine in the Miles image (the trainer's own inference stack).

    modal run modal/sample.py --model qwen3.5-0.8b
    modal run modal/sample.py --model qwen3.5-4b
    modal run modal/sample.py --model qwen3.6-35b-a3b                # 2 GPUs, TP2
    modal run modal/sample.py --model qwen3.5-4b \\
        --checkpoint RUN_ID/huggingface/iter_0000060 --name rft-60   # a trained policy

Writes `runs/samples/<name>.jsonl` locally: one line per response (prompt_id,
sample, response, finish_reason, completion_tokens). Defaults: K = 2 samples,
temperature 0.7, top-p 0.8, top-k 20 (Qwen's non-thinking settings), up to 2,048
new tokens (twice the training cap, so truncation shows up as a number rather
than as a silent bias; `rubricrl.evals.run` reports it).
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("rubricrl-sample")
ROOT = Path(__file__).parent.parent
DATA = Path("/data")  # Volume rubricrl-data, shared with modal/miles_rft.py
RUNS = Path("/runs")  # Volume rubricrl-runs: exported checkpoints
EVAL_PROMPTS = ROOT / "data" / "miles" / "healthbench_test.jsonl"

# name -> (HF repo, revision, GPUs). Same layout as modal/miles_rft.py.
MODELS = {
    "qwen3.5-0.8b": (
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
        1,
    ),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 1),
    "qwen3.6-35b-a3b": (
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        2,
    ),
    "qwen3-4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 1),
}

data_volume = modal.Volume.from_name("rubricrl-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("rubricrl-runs", create_if_missing=True, version=2)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==0.34.4")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
image = modal.Image.from_registry(
    "radixark/miles:v0.1.1@sha256:6355834f16bacd35d5d40c43f142e3758376f7b2e8d678bccfe870c092bd96bf"
).entrypoint([])


@app.function(
    image=download_image,
    volumes={DATA: data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=90 * 60,
)
def download(name: str) -> None:
    from huggingface_hub import snapshot_download

    repo, rev, _ = MODELS[name]
    snapshot_download(repo, revision=rev, local_dir=DATA / "models" / name)
    data_volume.commit()


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=64 * 1024,
    volumes={DATA: data_volume, RUNS: runs_volume},
    timeout=60 * 60,
)
def generate(
    name: str,
    checkpoint: str,
    prompts: list[dict],
    k: int,
    sampling: dict,
    tp: int,
) -> list[dict]:
    import sglang as sgl
    from transformers import AutoTokenizer

    base = DATA / "models" / name
    model_path = RUNS / checkpoint if checkpoint else base
    tok = AutoTokenizer.from_pretrained(base)  # the base model's chat template
    texts = [
        tok.apply_chat_template(
            p["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    engine = sgl.Engine(
        model_path=str(model_path),
        tokenizer_path=str(base),
        tp_size=tp,
        mem_fraction_static=0.85,
    )
    t0 = time.monotonic()
    jobs = [(j, i) for j in range(len(prompts)) for i in range(k)]
    outputs = engine.generate([texts[j] for j, _ in jobs], sampling)
    engine.shutdown()
    print(f"{len(jobs)} responses in {time.monotonic() - t0:.0f}s", flush=True)
    return [
        {
            "prompt_id": prompts[j]["metadata"]["prompt_id"],
            "sample": i,
            "response": out["text"],
            "finish_reason": (out["meta_info"].get("finish_reason") or {}).get("type"),
            "completion_tokens": out["meta_info"].get("completion_tokens"),
            "model": name,
            "checkpoint": checkpoint or None,
            "sampling": sampling,
        }
        for (j, i), out in zip(jobs, outputs, strict=True)
    ]


@app.local_entrypoint()
def main(
    model: str = "qwen3.5-4b",
    checkpoint: str = "",
    name: str = "",
    k: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_new_tokens: int = 2048,
):
    """--model a base model name; --checkpoint a path in the rubricrl-runs Volume."""
    if model not in MODELS:
        raise SystemExit(f"--model one of {sorted(MODELS)}")
    prompts = [json.loads(line) for line in EVAL_PROMPTS.read_text().splitlines()]
    _, _, tp = MODELS[model]
    sampling = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
    }
    download.remote(model)
    fn = generate.with_options(gpu=f"H100!:{tp}", cpu=8 * tp, memory=64 * 1024 * tp)
    rows = fn.remote(model, checkpoint, prompts, k, sampling, tp)
    out = ROOT / "runs" / "samples" / f"{name or model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    truncated = sum(r["finish_reason"] == "length" for r in rows) / len(rows)
    print(f"wrote {out}: {len(rows)} responses, {truncated:.1%} truncated")
