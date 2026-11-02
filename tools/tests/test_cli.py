"""Exercise repository commands from outside the checkout."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_command_runs_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "proact.py"), "benchmark", "validate"],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sessions"] == 40


def test_training_preview_does_not_start_torch(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "proact.py"), "--dry-run", "--processes", "2", "train", "controlnet"],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "torch.distributed.run" in result.stdout
    assert "--nproc_per_node=2" in result.stdout
    assert str(ROOT / "configs/training/controlnet.yaml") in result.stdout
    assert "--controlnet" in result.stdout
