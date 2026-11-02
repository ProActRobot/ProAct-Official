from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from retarget.bvh import Bvh
from retarget.__main__ import convert

ROOT = Path(__file__).resolve().parents[1]


def synthetic_bvh():
    lines = [
        "HIERARCHY",
        "ROOT b_root",
        "{",
        "OFFSET 0 80 0",
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation",
    ]
    count = 6
    for side, sign in [("l", 1), ("r", -1)]:

        def joint(name, offset, children=()):
            nonlocal count
            lines.extend(
                [
                    f"JOINT b_{side}_{name}",
                    "{",
                    f"OFFSET {offset}",
                    "CHANNELS 3 Zrotation Yrotation Xrotation",
                ]
            )
            count += 3
            for child in children:
                child()
            lines.append("}")

        def fingers():
            for finger in ["thumb", "middle", "index"]:
                joint(
                    finger + "1",
                    "0 0 3",
                    [lambda finger=finger: joint(finger + "2", "0 0 3")],
                )

        joint(
            "arm",
            f"{sign * 18} 20 0",
            [
                lambda: joint(
                    "forearm",
                    f"{sign * 25} 0 0",
                    [lambda: joint("wrist", f"{sign * 36} 0 0", [fingers])],
                )
            ],
        )
    lines.extend(
        [
            "}",
            "MOTION",
            "Frames: 2",
            "Frame Time: 0.04",
            " ".join(["0"] * count),
            " ".join(["0"] * count),
        ]
    )
    return "\n".join(lines) + "\n"


def test_complete_headless_conversion(tmp_path):
    source = tmp_path / "synthetic.bvh"
    source.write_text(synthetic_bvh())
    model = Bvh()
    model.parse_file(source)
    assert model.frames == 2 and model.fps == 25
    csv_path = Path(
        convert(
            (
                source,
                tmp_path,
                ROOT / "assets/unitree_g1/scene_empty_coll.xml",
                ROOT / "hierarchy_g1.bvh",
                False,
                0,
                1000,
            )
        )
    )
    result = pd.read_csv(csv_path)
    assert result.frame_idx.tolist() == [0, 1]
    assert result.shape == (2, 51)
    assert np.isfinite(result.to_numpy()).all()
    assert "Frame Time: 0.04" in csv_path.with_suffix(".bvh").read_text()


def test_parallel_cli_and_failure_exit(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ["a", "b"]:
        (inputs / f"{name}.bvh").write_text(synthetic_bvh())
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "retarget",
            str(inputs),
            "--output-dir",
            str(output),
            "--workers",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    for name in ["a", "b"]:
        assert pd.read_csv(output / f"{name}.csv").frame_idx.tolist() == [0, 1]
        assert (output / f"{name}.bvh").exists()
    failure = subprocess.run(
        [
            sys.executable,
            "-m",
            "retarget",
            str(inputs / "missing.bvh"),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert failure.returncode != 0
