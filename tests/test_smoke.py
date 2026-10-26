"""Smoke: packages import, the Modal app loads, the cluster check runs on CPU Ray."""

import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MODULES = [
    "nanorft.algos",
    "nanorft.cluster_check",
    "nanorft.driver",
    "nanorft.generator",
    "nanorft.learner",
    "nanorft.protocol",
    "nanorft.reference",
    "nanorft.reward_worker",
    "nanorft.tracking",
]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)


def test_modal_app_loads():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location(
        "ray_cluster", ROOT / "modal" / "ray_cluster.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PYTHON_VERSION == (ROOT / ".python-version").read_text().strip()
    assert set(module.ENTRIES) == {"check", "train"}


def test_cluster_check_needs_gpus():
    ray = pytest.importorskip("ray")
    from nanorft import cluster_check

    ray.init(num_cpus=1, num_gpus=0, include_dashboard=False)
    try:
        with pytest.raises(RuntimeError, match="no GPUs"):
            cluster_check.main({})
    finally:
        ray.shutdown()
