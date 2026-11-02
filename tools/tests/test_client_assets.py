"""Verify preprocessing exports survive the cleaned package layout."""
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]


def test_exported_pymo_pipeline_loads_in_client(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src/motion"))
    from data_process.pymo.data import MocapData
    from data_process.pymo.preprocessing import Numpyfier

    track = MocapData()
    track.values = pd.DataFrame(np.arange(62).reshape(2, 31),
                                columns=[f"joint_{i}" for i in range(31)])
    pipeline = Pipeline([("numpy", Numpyfier())]).fit([track])
    scaler = StandardScaler().fit(track.values.to_numpy())
    pipe_path, scaler_path = tmp_path / "pipe.sav", tmp_path / "scaler.sav"
    joblib.dump(pipeline, pipe_path)
    joblib.dump(scaler, scaler_path)
    env = dict(os.environ, PROACT_DATA_PIPE=str(pipe_path),
               PROACT_TEXT_DATA_PIPE=str(pipe_path),
               PROACT_OUTPUT_SCALER=str(scaler_path),
               PROACT_TEXT_OUTPUT_SCALER=str(scaler_path))
    # A fresh interpreter must discover the new data_process package itself.
    code = """
import sys
sys.path.insert(0, 'src/client')
import numpy as np
import realtime_client as client
client._load_motion_assets()
values, mocap = client.visualize_and_write_no_gt(np.zeros((2, 31)), '.', 1)
assert values.shape == (2, 31)
assert np.allclose(values.to_numpy(), client.body_scaler.mean_)
assert mocap.values.shape == (2, 31)
print('export roundtrip passed')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "export roundtrip passed" in result.stdout


def test_legacy_top_level_pymo_pipeline_loads_in_client(tmp_path):
    """A historical export referring to top-level ``pymo`` remains usable."""
    data_process_dir = ROOT / "src/motion" / "data_process"
    pipe_path, scaler_path = tmp_path / "legacy_pipe.sav", tmp_path / "legacy_scaler.sav"
    build_code = f"""
import joblib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, {str(data_process_dir)!r})
from pymo.data import MocapData
from pymo.preprocessing import Numpyfier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

track = MocapData()
track.values = pd.DataFrame(np.arange(62).reshape(2, 31),
                            columns=[f"joint_{{i}}" for i in range(31)])
pipeline = Pipeline([("numpy", Numpyfier())]).fit([track])
scaler = StandardScaler().fit(track.values.to_numpy())
joblib.dump(pipeline, {str(pipe_path)!r})
joblib.dump(scaler, {str(scaler_path)!r})
print("legacy export created")
"""
    created = subprocess.run(
        [sys.executable, "-c", build_code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert created.returncode == 0, created.stderr
    assert "legacy export created" in created.stdout

    output_dir = tmp_path / "output"
    env = dict(
        os.environ,
        PROACT_DATA_PIPE=str(pipe_path),
        PROACT_TEXT_DATA_PIPE=str(pipe_path),
        PROACT_OUTPUT_SCALER=str(scaler_path),
        PROACT_TEXT_OUTPUT_SCALER=str(scaler_path),
    )
    load_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, 'src/client')
import numpy as np
import realtime_client as client

output_dir = Path({str(output_dir)!r})
output_dir.mkdir()
client._load_motion_assets()
values, mocap = client.visualize_and_write_no_gt(np.zeros((2, 31)), str(output_dir), 1)
assert values.shape == (2, 31)
assert np.allclose(values.to_numpy(), client.body_scaler.mean_)
assert mocap.values.shape == (2, 31)
print('legacy export roundtrip passed')
"""
    loaded = subprocess.run(
        [sys.executable, "-c", load_code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert "legacy export roundtrip passed" in loaded.stdout
