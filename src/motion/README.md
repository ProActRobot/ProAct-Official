# Motion generation

This directory contains the audio Flow Matching backbone, semantic ControlNet,
motion preprocessing and streaming samplers. Start with the
[training guide](../../docs/guides/training.md) or [inference guide](../../docs/guides/inference.md).
Configuration files live under [configs](../../configs/README.md).

| Path | Role |
| --- | --- |
| `train.py` | Distributed training entry point |
| `Gesture_trainer.py` | Backbone and ControlNet training/inference orchestration |
| `generate_multimodal_gestures.py` | Main streaming generation used by the server |
| `model/model_for_interaction_7_cfg_mfcc.py` | Released model architecture |
| `ddpm/flow_matching_new.py` | Main loss and Euler samplers |
| `data_process/` | Audio/motion features and per-clip SeG preparation |
| `dataset/` | Audio and text dataset loaders |
| `utils/` | Streaming features and scheduler utilities |

SeG captions are retained independently for each `file_name`. Caches are written
under `text/by_clip/`, and training windows use their own clip's captions.
See [SeG preparation](../../docs/guides/training.md#seg-annotations).

Legacy triplet generation and alternate samplers support existing experiments;
their optional config lives in `configs/legacy/`. The `cal_fid*.py` utilities use
diagnostic feature definitions and do not implement the complete paper protocol.

PyMO remains under `data_process/pymo/` so historical joblib exports can load
through both `data_process.pymo` and top-level `pymo` module names. Attribution
is recorded in [THIRD_PARTY_NOTICES.md](../../docs/THIRD_PARTY_NOTICES.md).
