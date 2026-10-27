"""Smoke: packages import, the Modal apps load, the cluster check runs on CPU Ray,
and what runs inside the Miles image (Python 3.12) parses as Python 3.10."""

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MODULES = [
    "rubricrl.analysis.calibrate",
    "rubricrl.cluster_check",
    "rubricrl.data.healthbench",
    "rubricrl.data.schema",
    "rubricrl.evals.run",
    "rubricrl.judge.client",
    "rubricrl.judge.prompt",
    "rubricrl.rewards.miles_rm",
    "rubricrl.rewards.rubric",
]
# Imported by Miles' reward hook, or by Modal inside the Miles image.
IN_MILES_IMAGE = [
    "src/rubricrl/data/schema.py",
    "src/rubricrl/judge/client.py",
    "src/rubricrl/judge/prompt.py",
    "src/rubricrl/rewards/miles_rm.py",
    "src/rubricrl/rewards/rubric.py",
    "modal/miles_rft.py",
    "modal/sample.py",
]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("path", IN_MILES_IMAGE)
def test_parses_as_python_310(path):
    ast.parse((ROOT / path).read_text(), feature_version=(3, 10))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_apps_load():
    pytest.importorskip("modal")
    ray_cluster = _load(ROOT / "modal" / "ray_cluster.py")
    local = (ROOT / ".python-version").read_text().strip()
    assert local.startswith(ray_cluster.PYTHON_SERIES + ".")
    assert ray_cluster.ENTRIES == {"check": "rubricrl.cluster_check"}
    rft = _load(ROOT / "modal" / "miles_rft.py")
    sample = _load(ROOT / "modal" / "sample.py")
    # The two apps share a Volume layout: same names, same pinned revisions.
    for name, (repo, rev) in rft.MODELS.items():
        assert sample.MODELS[name][:2] == (repo, rev)


def test_miles_args_are_well_formed(tmp_path):
    pytest.importorskip("modal")
    rft = _load(ROOT / "modal" / "miles_rft.py")
    args = rft.training_args("healthbench", "qwen3.5-4b", tmp_path / "run", 60, False)
    flags = [a for a in args if a.startswith("--")]
    assert len(flags) == len(set(flags)), "a flag is set twice"
    i = args.index("--custom-rm-path")
    assert args[i + 1] == "rubricrl.rewards.miles_rm.healthbench_reward"
    assert int(args[args.index("--global-batch-size") + 1]) == 16 * 8  # on-policy
    assert (
        args[args.index("--apply-chat-template-kwargs") + 1]
        == '{"enable_thinking": false}'
    )
    smoke = rft.training_args("healthbench", "qwen3.5-4b", tmp_path / "run", 2, True)
    assert any(a.endswith("healthbench_val.jsonl@[0:8]") for a in smoke)
    math = rft.training_args("math", "qwen3.5-4b", tmp_path / "run", 2, True)
    assert "--custom-rm-path" not in math and "math" in math


def test_cluster_check_needs_gpus():
    ray = pytest.importorskip("ray")
    from rubricrl import cluster_check

    ray.init(num_cpus=1, num_gpus=0, include_dashboard=False)
    try:
        with pytest.raises(RuntimeError, match="no GPUs"):
            cluster_check.main({})
    finally:
        ray.shutdown()


CLIS = {
    "rubricrl.data.healthbench": {"download", "stats", "export-miles"},
    "rubricrl.analysis.calibrate": {"estimate", "run", "report"},
    "rubricrl.evals.run": {"score", "table", "compare", "physician"},
}


@pytest.mark.parametrize("name", CLIS)
def test_cli_help(name):
    from click.testing import CliRunner

    group = importlib.import_module(name).main
    assert set(group.commands) == CLIS[name]
    for cmd in [[], *([c] for c in CLIS[name])]:
        result = CliRunner().invoke(group, [*cmd, "--help"])
        assert result.exit_code == 0, result.output


def test_cli_options_match_the_functions_they_forward_to():
    """`score` and `run` pass their options through as **kwargs."""
    import inspect

    from rubricrl.analysis import calibrate
    from rubricrl.evals import run

    for cmd, fn in [
        (run.main.commands["score"], run.score_file),
        (calibrate.main.commands["run"], calibrate.run),
    ]:
        assert {p.name for p in cmd.params} == set(inspect.signature(fn).parameters)
