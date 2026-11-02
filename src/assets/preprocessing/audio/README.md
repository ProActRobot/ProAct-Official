# Audio motion preprocessing exports

These three small files are bundled with the source so the released audio
backbone and ControlNet can use their matching preprocessing without another
audio-asset download. Together they occupy 25,109 bytes.

| File | Purpose |
| --- | --- |
| `input_scaler.sav` | Fitted StandardScaler for 181-dimensional audio features |
| `output_scaler.sav` | Fitted StandardScaler for 31-dimensional motion features |
| `data_pipe_30fps.sav` | Fitted motion pipeline, skeleton, channel layout and inverse transform |

The pipeline's saved motion template contains zero motion frames. These are
inference preprocessing exports; they do not include training examples, audio
recordings or neural network weights. Additional training data is obtained
separately under its original dataset terms.

The files are paired with the epoch-600 audio backbone and epoch-800 ControlNet in
[UsanoCoCr/ProAct](https://huggingface.co/UsanoCoCr/ProAct), revision
`b246d61db9f41502b6ed00b142e85461b2a2938c`.
[manifest.json](manifest.json) records their sizes and SHA-256 hashes.

Serialization uses joblib, scikit-learn and the retained PyMO classes. The
validated environment uses scikit-learn 1.6.1, NumPy 1.26.4 and joblib 1.6.0.
The client configures both historical PyMO import paths when loading the files.
Text-conditioned motion uses the separate SeG exports from
[UsanoCoCr/ProAct-Data](https://huggingface.co/datasets/UsanoCoCr/ProAct-Data).

The default inference config and example environment use this directory for
`MOTION_AUDIO_DATA_ROOT`. Client audio inverse transforms use these bundled
files by default; environment variables can select another matching export.
Training outputs use the separate `MOTION_BACKBONE_DATA_ROOT` and
`MOTION_AUDIO_PROCESSED_DIR` paths, keeping generated files outside this directory.
