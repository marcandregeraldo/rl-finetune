from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


def merge_lora(checkpoint_dir: str | Path, output_dir: str | Path) -> Path:
    """Merge LoRA adapter with the base model and save as a HF model.

    Args:
        checkpoint_dir: Path to a TRL checkpoint directory (contains adapter_config.json).
        output_dir: Where to save the merged HF model.

    Returns:
        Path to the merged model directory.
    """
    from peft import AutoPeftModelForCausalLM

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merging LoRA from %s …", checkpoint_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype="bfloat16",
        device_map="cpu",  # merge on CPU to avoid MPS OOM
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))

    for f in checkpoint_dir.parent.glob("tokenizer*"):
        shutil.copy(f, output_dir)
    for f in checkpoint_dir.parent.glob("special_tokens_map.json"):
        shutil.copy(f, output_dir)

    logger.info("Merged model saved to %s", output_dir)
    return output_dir


def convert_to_gguf(
    hf_model_dir: str | Path,
    output_path: str | Path,
    quantize: str = "q4_k_m",
) -> Path:
    """Convert a HF model to GGUF using llama.cpp's convert script.

    Requires llama.cpp to be installed (brew install llama.cpp or from source).

    Args:
        hf_model_dir: Path to the merged HF model directory.
        output_path: Destination GGUF file path.
        quantize: Quantization type (e.g. "q4_k_m", "q8_0", "f16").

    Returns:
        Path to the output GGUF file.
    """
    hf_model_dir = Path(hf_model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f16_path = output_path.with_suffix(".f16.gguf")

    convert_script = _find_convert_script()
    if not convert_script:
        raise RuntimeError(
            "Could not find llama.cpp's convert_hf_to_gguf.py. "
            "Install llama.cpp: brew install llama.cpp"
        )

    logger.info("Converting to GGUF (f16) …")
    subprocess.run(
        [sys.executable, str(convert_script), str(hf_model_dir), "--outfile", str(f16_path)],
        check=True,
    )

    if quantize.lower() == "f16":
        f16_path.rename(output_path)
    else:
        logger.info("Quantizing to %s …", quantize)
        subprocess.run(
            ["llama-quantize", str(f16_path), str(output_path), quantize.upper()],
            check=True,
        )
        f16_path.unlink(missing_ok=True)

    logger.info("GGUF saved to %s", output_path)
    return output_path


def reload_server(server_url: str, model_path: str | Path) -> bool:
    """Log the new model path and check that llama-server is still up.

    llama-server doesn't yet have a standard hot-reload endpoint.
    The WeightSyncCallback logs the path; the user should restart the server
    with `rl serve --model <path>` to pick up the new weights.

    Args:
        server_url: llama-server base URL.
        model_path: Path to the new GGUF file.

    Returns:
        True if the server responded after the sync, False otherwise.
    """
    logger.info(
        "Weight sync complete. Restart llama-server to use new model: %s",
        model_path,
    )
    try:
        resp = httpx.get(f"{server_url}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def sync_weights(checkpoint_dir: str | Path, config: dict) -> bool:
    """Full sync pipeline: merge → GGUF → reload.

    Called by WeightSyncCallback during training.

    Args:
        checkpoint_dir: TRL checkpoint directory with LoRA adapter.
        config: Training config dict (from config/default.yaml).

    Returns:
        True on success, False on any error (training continues either way).
    """
    try:
        with tempfile.TemporaryDirectory(prefix="rl-merged-") as tmp:
            merged_dir = merge_lora(checkpoint_dir, Path(tmp) / "merged")

        gguf_path = Path(config.get("output_dir", "checkpoints/grpo")) / "synced.gguf"
        convert_to_gguf(merged_dir, gguf_path, quantize=config.get("quantize", "q4_k_m"))
        reload_server(config.get("llama_server_url", "http://localhost:8080"), gguf_path)
        return True
    except Exception as e:
        logger.warning("Weight sync failed (training continues): %s", e)
        return False


class WeightSyncCallback:
    """TRL TrainerCallback that triggers LoRA→GGUF sync every N steps.

    Usage:
        trainer = GRPOTrainer(..., callbacks=[WeightSyncCallback(config, sync_every=50)()])
    """

    def __init__(self, config: dict, sync_every: int = 50):
        from transformers import TrainerCallback

        self._config = config
        self._sync_every = sync_every

        class _Callback(TrainerCallback):
            def on_step_end(inner_self, args, state, control, **kwargs):
                if self._sync_every > 0 and state.global_step % self._sync_every == 0:
                    ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    if ckpt.exists():
                        logger.info("Triggering weight sync at step %d", state.global_step)
                        sync_weights(ckpt, self._config)

        self._callback = _Callback()

    def __call__(self):
        return self._callback


def _find_convert_script() -> Path | None:
    """Locate llama.cpp's convert_hf_to_gguf.py on the system."""
    candidates = [
        Path("/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/usr/share/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin:
        share = Path(quantize_bin).parent.parent / "share" / "llama.cpp"
        candidate = share / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    import click

    @click.group()
    def _cli():
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    @_cli.command("sync")
    @click.option("--checkpoint", required=True, help="TRL checkpoint directory")
    @click.option("--config", "-c", default="config/default.yaml")
    def _sync(checkpoint, config):
        """Merge + convert + reload (used during training)."""
        import yaml

        with open(config) as f:
            cfg = yaml.safe_load(f)
        ok = sync_weights(checkpoint, cfg)
        sys.exit(0 if ok else 1)

    @_cli.command("model")
    @click.option("--checkpoint", required=True, help="TRL checkpoint directory")
    @click.option("--output", "-o", default="models/finetuned.gguf", show_default=True)
    @click.option("--quantize", "-q", default="q4_k_m", show_default=True)
    def _model(checkpoint, output, quantize):
        """Standalone export of a checkpoint to GGUF."""
        with tempfile.TemporaryDirectory(prefix="rl-merged-") as tmp:
            merged = merge_lora(checkpoint, Path(tmp) / "merged")
        convert_to_gguf(merged, output, quantize=quantize)

    _cli()
