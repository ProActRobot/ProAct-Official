# Motion retargeting for Unitree G1

This component converts human motion in BVH format into motion for the Unitree G1
robot model. It reads the supported source skeleton, maps body and hand motion
to G1 targets, and uses inverse kinematics with MuJoCo and Mink to solve the
robot's joint configurations frame by frame.

Each input produces a CSV file containing the robot's floating-base pose and
joint positions, plus a BVH file using the G1 hierarchy. The pipeline supports
optional Gaussian smoothing, parallel conversion of multiple files, and a
MuJoCo preview for inspecting a single motion.

To use it, install the dependencies, prepare a BVH with the source skeleton
described below, and run the conversion command. Start with one file and use
`--visual` to inspect the motion, or pass a directory and `--workers` for batch
processing. The model XML, meshes, and output hierarchy are included; provide
the source motion files.

## Dependencies

Use Python 3.10 or later. A display and supported graphics driver are required only
for `--visual`. Run these commands from `tools/retargeting/` in the integrated
repository.

| Dependency | Version constraint |
| --- | --- |
| `numpy` | `>=1.26,<2` |
| `scipy` | `>=1.11,<2` |
| `pandas` | `>=2.2,<3` |
| `mujoco` | `==3.3.0` |
| `mink` | `==0.0.5` |
| `qpsolvers` | `>=4.8,<5` |
| `osqp` | `>=1,<2` |
| `loop-rate-limiters` | `>=1.1,<2` |
| `transforms3d` | `>=0.4,<1` |

Local checks additionally use `pytest>=8,<10` and `ruff>=0.14,<1`.

## Installation

```console
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in Bash, then install:

```console
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment variables

No API keys are required. Supply input, output, model, and hierarchy paths through
the command-line arguments below.

## Run conversion

The G1 model XML, its referenced meshes, and `hierarchy_g1.bvh` are included.
Provide the source motion BVH using the input argument. Keep the model XML and
mesh directory together. With a custom `--model`, also provide every XML include
and mesh file referenced by that model; with `--hierarchy`, provide the matching
BVH hierarchy.

```console
python -m retarget /path/to/motion.bvh --output-dir outputs/single
python -m retarget /path/to/motions --output-dir outputs/batch --workers 4
```

Input is a BVH file or a directory of BVH files. Each input produces CSV and BVH
files in the output directory. Use a source skeleton with `b_root`, `b_l_arm`,
`b_l_forearm`, `b_l_wrist`, corresponding right-arm joints, and hand `thumb1/2`,
`middle1/2`, `index1/2` joints. Source offsets are in centimetres.

| Argument | Use |
| --- | --- |
| `--visual` | Preview one input; use one worker. |
| `--smooth-sigma 1` | Apply Gaussian smoothing; default 0 disables it. |
| `--frequency 1000` | Set IK integration frequency. |
| `--model /path/to/scene.xml` | Select the supported 50-qpos G1 scene. |
| `--hierarchy /path/to/hierarchy.bvh` | Select the matching G1 BVH hierarchy. |

## Run local checks

```console
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m retarget --help
```
