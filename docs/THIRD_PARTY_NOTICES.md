# Third party notices

This file records the upstream attribution and license sources for third party
code and robot assets retained in this tree. The full license text is kept in
licenses/ when the upstream repository provides a license file. Dataset and
model-download terms are listed separately from code licenses.

The homepage uses unmodified original project figures. Their source links and
attribution are recorded in [docs/images](images/README.md); the website's
figure terms are separate from the source-code MIT license.

## PyMO

The copy under src/motion/data_process/pymo/ is from
[PyMO](https://github.com/omimo/PyMO), created by Omid Alemi. PyMO is
distributed under the MIT License; the exact
[upstream master license text](https://raw.githubusercontent.com/omimo/PyMO/master/LICENSE)
is retained in [licenses/pymo-MIT.txt](licenses/pymo-MIT.txt).

## Mink

The optional `mink==0.0.6` dependency is installed separately from
[Mink](https://github.com/kevinzakka/mink), authored and maintained by Kevin
Zakka. The upstream repository publishes the Apache License, Version 2.0; its
[upstream main license text](https://raw.githubusercontent.com/kevinzakka/mink/main/LICENSE)
is retained in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

The standalone retargeting component under `tools/retargeting/` uses Mink 0.0.5
in a separate environment. Its bundled G1 assets retain their own attribution and
license in [tools/retargeting/NOTICE.md](../tools/retargeting/NOTICE.md) and
[tools/retargeting/assets/unitree_g1/LICENSE](../tools/retargeting/assets/unitree_g1/LICENSE).

## Unitree G1 robot description

The URDF, MJCF, and mesh assets under src/assets/g1/ are attributed by
the retained G1 package documentation to Unitree Robotics' public
[unitree_ros G1 description](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description).
The upstream repository publishes the BSD 3-Clause License, including:

> Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")

The exact [upstream license text](https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/LICENSE)
is retained in
[licenses/unitree-ros-BSD-3-Clause.txt](licenses/unitree-ros-BSD-3-Clause.txt).

## Unitree SDK2 Python

The retained src/third_party/unitree_sdk2py/ package is from
[unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python).
The upstream BSD 3-Clause text carries:

> Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")

The exact [upstream license text](https://raw.githubusercontent.com/unitreerobotics/unitree_sdk2_python/master/LICENSE)
is retained in
[licenses/unitree-sdk2-python-BSD-3-Clause.txt](licenses/unitree-sdk2-python-BSD-3-Clause.txt).

## Semantic Gesticulator code

The retained src/motion/ code is a reimplementation and retained portion of
[Semantic Gesticulator](https://github.com/LuMen-ze/Semantic-Gesticulator-Official).
Its [upstream README](https://raw.githubusercontent.com/LuMen-ze/Semantic-Gesticulator-Official/main/README.md)
states that the project is licensed under the MIT License, but the upstream
repository has no LICENSE file. The repository therefore records the upstream
statement and its provenance in
[licenses/semantic-gesticulator-NOTICE.txt](licenses/semantic-gesticulator-NOTICE.txt)
without asserting an exact MIT text that the upstream project did not publish.

The upstream paper attribution is:

> Zeyi Zhang, Tenglong Ao, Yuyao Zhang, Qingzhe Gao, Chuan Lin, Baoquan Chen,
> and Libin Liu, "Semantic Gesticulator: Semantics-Aware Co-Speech Gesture
> Synthesis," ACM Transactions on Graphics, July 2024,
> [doi:10.1145/3658134](https://doi.org/10.1145/3658134).

## MOCCA Semantic Gesture Dataset

The MOCCA Semantic Gesture Dataset is not bundled in this release. Its
upstream [license agreement](https://github.com/LuMen-ze/Semantic-Gesticulator-Official/blob/main/SeG_dataset/License%20Agreement%20for%20MOCCA%20Semantic%20Gesture%20Dataset.pdf)
identifies the Motion Control and Character Animation Group (MOCCA) as the
owner and grants a non-commercial, non-exclusive, non-transferable,
non-sublicensable, royalty-free research license. It prohibits sharing or
distribution to third parties without prior written consent, commercial use,
commercial model training, and other uses specified in the agreement; it
allows one archive copy. Publications using the dataset must cite the
Semantic Gesticulator paper above.

The upstream README links the motion archive at
<https://drive.google.com/file/d/1qVQ4-XwAzhfoHRM6EzR9BrGI1lD1vqVW/view?usp=sharing>.
That link is a source reference only. It does not change the upstream
agreement or grant redistribution rights.

The Semantic Gesticulator generator checkpoints, retriever weights, and
zeroeggs data are external downloads described by the upstream README. No
license text for those downloads is bundled here, so this notice does not
relicense or grant rights to them.

## ProAct checkpoint and processed-data releases

The matched ProAct backbone and ControlNet checkpoints are external downloads
from [UsanoCoCr/ProAct](https://huggingface.co/UsanoCoCr/ProAct), revision
`b246d61db9f41502b6ed00b142e85461b2a2938c`. That repository publishes CC BY-NC
4.0 terms for its artifacts. The source MIT license does not relicense them.

The processed SeG export and retargeted Photoreal subset are external downloads
from [UsanoCoCr/ProAct-Data](https://huggingface.co/datasets/UsanoCoCr/ProAct-Data),
revision `09e82a76c12931503e83f1026a40bb4fbc5687d1`. Consult that repository's
data card, license and source-specific terms before using its archives. The data
card records project-owner authorization for the processed SeG export; it does
not grant redistribution rights to other upstream data.

The complete mixed audio training dataset includes ZEGGS-derived material and is
not redistributed in that release. This source repository bundles the matching
audio-domain inference scalers and motion pipeline under
[src/assets/preprocessing/audio](../src/assets/preprocessing/audio/README.md), following
the project owner's instruction. The three exports contain fitted statistics
and a motion template with zero motion frames; they do not include the mixed
training dataset. Model and original dataset terms remain separate from the
source code license.
