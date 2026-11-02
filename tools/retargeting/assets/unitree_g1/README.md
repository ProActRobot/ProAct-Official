# Unitree G1 model

Model: [G1 Humanoid Robot](https://www.unitree.com/g1/) by
[Unitree Robotics](https://www.unitree.com/).
MJCF reference: [g1_description](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description).

## Requirements

Use MuJoCo 2.3.4 or later. For the retargeting pipeline, install the versions in
the repository's `requirements.txt`.

## Run

Run from the repository root:

```console
python -m retarget /path/to/motion.bvh --output-dir outputs/motion --model assets/unitree_g1/scene_empty_coll.xml
```
