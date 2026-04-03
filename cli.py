from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
import httpx
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

_REPO_ROOT = Path(__file__).parent
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"

# OpenEnv Docker images
_ENV_IMAGES = {
    "wordle": "registry.hf.space/openenv/wordle:latest",
    "coding": "registry.hf.space/openenv/coding_env:latest",
}
_ENV_PORTS = {
    "wordle": 8001,
    "coding": 8002,
}





@click.group()
def cli():
    """rl-finetune — local RL post-training for LLMs."""
    pass





@cli.command()
@click.option("--config", "-c", default=str(_DEFAULT_CONFIG), show_default=True, help="YAML config file")
@click.option("--max-steps", type=int, default=None, help="Override max_steps in config")
@click.option("--env", type=str, default=None, help="Override env in config (e.g. wordle)")
@click.option("--dry-run", is_flag=True, help="Print resolved config and exit")
def train(config, max_steps, env, dry_run):
    """Launch GRPO training with a config file."""
    import yaml

    # Load config
    try:
        with open(config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Config not found: {config}[/red]")
        sys.exit(1)

    # Apply CLI overrides
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    if env is not None:
        cfg["env"] = env

    if dry_run:
        console.print(Panel(yaml.dump(cfg, default_flow_style=False), title="Resolved config"))
        return

    # Pre-flight checks
    _check_llama_server(cfg.get("llama_server_url", "http://localhost:8080"))
    _check_openenv(cfg.get("env_url", "http://localhost:8001"))

    # Print training summary
    console.print(
        Panel(
            f"[bold]Model:[/bold]    {cfg.get('model', '?')}\n"
            f"[bold]Env:[/bold]      {cfg.get('env', '?')} @ {cfg.get('env_url', '?')}\n"
            f"[bold]Steps:[/bold]    {cfg.get('max_steps', 'epochs=' + str(cfg.get('num_train_epochs', 1)))}\n"
            f"[bold]Output:[/bold]   {cfg.get('output_dir', 'checkpoints/grpo')}\n"
            f"[bold]LR:[/bold]       {cfg.get('learning_rate', '?')}\n"
            f"[bold]Rollout:[/bold]  {cfg.get('llama_server_url', '?')}",
            title="[bold cyan]rl train[/bold cyan]",
            border_style="cyan",
        )
    )

    # Run training (import here to keep CLI startup fast)
    from train import main as train_main

    try:
        train_main(cfg)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted.[/yellow]")





@cli.command()
@click.option("--model", "-m", default=None, help="Path to GGUF model file")
@click.option("--port", "-p", default=8080, show_default=True, help="Server port")
@click.option("--ctx-size", default=4096, show_default=True, help="Context window size")
@click.option("--config", "-c", default=str(_DEFAULT_CONFIG), help="YAML config to read defaults from")
def serve(model, port, ctx_size, config):
    """Start llama-server for rollout generation and eval inference."""
    # Resolve model path
    if model is None:
        try:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            port = cfg.get("llama_server_port", port)
        except Exception:
            pass

        # Look for any GGUF in models/
        guesses = sorted(Path("models").glob("*.gguf")) if Path("models").exists() else []
        if guesses:
            model = str(guesses[-1])  # most recently modified
        else:
            console.print("[red]No model specified and no .gguf found in models/[/red]")
            console.print("Usage: rl serve --model models/Qwen3-1.7B-Q4_K_M.gguf")
            sys.exit(1)

    if not Path(model).exists():
        console.print(f"[red]Model file not found: {model}[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]Model:[/bold]  {model}\n"
            f"[bold]Port:[/bold]   {port}\n"
            f"[bold]URL:[/bold]    http://localhost:{port}",
            title="[bold green]llama-server[/bold green]",
            border_style="green",
        )
    )

    # Stream server output
    cmd = [
        "llama-server",
        "-m", model,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--n-gpu-layers", "99",
        "--cont-batching",
        "--log-prefix",
    ]
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            for line in proc.stdout:
                console.print(f"[dim]{line.rstrip()}[/dim]")
    except FileNotFoundError:
        console.print("[red]llama-server not found. Install: brew install llama.cpp[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")





@cli.group()
def env():
    """Manage OpenEnv Docker containers."""
    pass


@env.command("start")
@click.argument("name", type=click.Choice(list(_ENV_IMAGES.keys())))
@click.option("--port", type=int, default=None, help="Override default port")
def env_start(name, port):
    """Start an OpenEnv environment container."""
    image = _ENV_IMAGES[name]
    host_port = port or _ENV_PORTS[name]

    console.print(f"[cyan]Pulling {image} …[/cyan]")
    subprocess.run(["docker", "pull", image], check=True)

    container_name = f"openenv-{name}"

    # Remove existing container if any
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )

    console.print(f"[cyan]Starting {name} on port {host_port} …[/cyan]")
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{host_port}:8000",
            image,
        ],
        check=True,
    )
    console.print(f"[green]{name} environment running at http://localhost:{host_port}[/green]")


@env.command("stop")
@click.argument("name", type=click.Choice(list(_ENV_IMAGES.keys())))
def env_stop(name):
    """Stop and remove an OpenEnv environment container."""
    container_name = f"openenv-{name}"
    subprocess.run(["docker", "rm", "-f", container_name], check=True)
    console.print(f"[yellow]{name} environment stopped.[/yellow]")


@env.command("ls")
def env_ls():
    """List known OpenEnv environments and their status."""
    table = Table(title="OpenEnv Environments")
    table.add_column("Name", style="cyan")
    table.add_column("Port")
    table.add_column("Image", style="dim")
    table.add_column("Status")

    for name, image in _ENV_IMAGES.items():
        port = _ENV_PORTS[name]
        status = _container_status(f"openenv-{name}")
        status_str = "[green]running[/green]" if status == "running" else "[dim]stopped[/dim]"
        table.add_row(name, str(port), image, status_str)

    console.print(table)





@cli.command()
@click.option("--checkpoint", "-k", required=True, help="Path to TRL checkpoint directory")
@click.option("--output", "-o", default="models/finetuned.gguf", show_default=True)
@click.option("--quantize", "-q", default="q4_k_m", show_default=True)
def export(checkpoint, output, quantize):
    """Convert a LoRA checkpoint to GGUF for llama-server."""
    from export import convert_to_gguf, merge_lora
    import tempfile

    console.print(f"[cyan]Merging LoRA from {checkpoint} …[/cyan]")
    with console.status("[bold yellow]Merging…"):
        with tempfile.TemporaryDirectory(prefix="rl-merged-") as tmp:
            merged = merge_lora(checkpoint, Path(tmp) / "merged")

    console.print(f"[cyan]Converting to GGUF ({quantize}) → {output} …[/cyan]")
    with console.status("[bold yellow]Converting…"):
        convert_to_gguf(merged, output, quantize=quantize)

    console.print(f"[green]Export complete: {output}[/green]")





@cli.command()
@click.option("--config", "-c", default="evals/promptfooconfig.yaml", show_default=True)
@click.option("--view", is_flag=True, help="Open promptfoo web UI after eval")
def eval(config, view):
    """Run promptfoo evaluation suite against llama-server."""
    if not Path(config).exists():
        console.print(f"[red]Eval config not found: {config}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Running promptfoo eval with {config} …[/cyan]")
    try:
        subprocess.run(["promptfoo", "eval", "--config", config], check=True)
    except FileNotFoundError:
        console.print("[red]promptfoo not found. Install: npm install -g promptfoo[/red]")
        sys.exit(1)

    if view:
        subprocess.run(["promptfoo", "view"])





@cli.command()
@click.option("--config", "-c", default=str(_DEFAULT_CONFIG))
def status(config, ):
    """Show llama-server health, OpenEnv status, and latest checkpoint."""
    try:
        with open(config) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {}

    llama_url = cfg.get("llama_server_url", "http://localhost:8080")
    env_url = cfg.get("env_url", "http://localhost:8001")
    output_dir = Path(cfg.get("output_dir", "checkpoints/grpo"))

    table = Table(title="[bold]rl-finetune status[/bold]", show_header=False)
    table.add_column("", style="bold cyan", no_wrap=True)
    table.add_column()

    # llama-server health
    try:
        resp = httpx.get(f"{llama_url}/health", timeout=3.0)
        llama_status = "[green]healthy[/green]" if resp.status_code == 200 else f"[red]HTTP {resp.status_code}[/red]"
    except Exception:
        llama_status = "[red]unreachable[/red]"
    table.add_row("llama-server", f"{llama_url}  {llama_status}")

    # OpenEnv health
    try:
        resp = httpx.get(f"{env_url}/health", timeout=3.0)
        env_status = "[green]healthy[/green]" if resp.status_code == 200 else f"[red]HTTP {resp.status_code}[/red]"
    except Exception:
        env_status = "[red]unreachable[/red]"
    table.add_row("OpenEnv", f"{env_url}  {env_status}")

    # Latest checkpoint
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            latest = checkpoints[-1]
            table.add_row("Latest checkpoint", str(latest))
        else:
            table.add_row("Latest checkpoint", "[dim]none yet[/dim]")
        # Latest log entry
        log_file = output_dir / "trainer_log.jsonl"
        if log_file.exists():
            last_line = _last_jsonl_line(log_file)
            if last_line:
                loss = last_line.get("loss", "—")
                reward = last_line.get("rewards/mean", "—")
                step = last_line.get("step", "—")
                table.add_row("Last log", f"step={step}  loss={loss}  reward={reward}")
    else:
        table.add_row("Checkpoints", "[dim]no output_dir yet[/dim]")

    console.print(table)





def _check_llama_server(url: str):
    """Warn if llama-server is not reachable before training."""
    try:
        httpx.get(f"{url}/health", timeout=3.0)
    except Exception:
        console.print(
            f"[yellow]Warning: llama-server not reachable at {url}.[/yellow]\n"
            "[yellow]Start it first: rl serve --model <path>[/yellow]"
        )
        if not click.confirm("Continue anyway?", default=False):
            sys.exit(1)


def _check_openenv(url: str):
    """Warn if the OpenEnv container is not reachable before training."""
    try:
        httpx.get(f"{url}/health", timeout=3.0)
    except Exception:
        console.print(
            f"[yellow]Warning: OpenEnv not reachable at {url}.[/yellow]\n"
            "[yellow]Start it first: rl env start wordle[/yellow]"
        )
        if not click.confirm("Continue anyway?", default=False):
            sys.exit(1)


def _container_status(name: str) -> str:
    """Return Docker container status string or 'stopped'."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "stopped"


def _last_jsonl_line(path: Path) -> dict | None:
    """Return the last non-empty JSON line from a .jsonl file."""
    try:
        lines = path.read_text().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line)
    except Exception:
        pass
    return None


if __name__ == "__main__":
    cli()
