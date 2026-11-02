# Setup

Use Python 3.12. Motion inference needs Linux, a CUDA GPU and a compatible
PyTorch 2.5.1 / torchaudio 2.5.1 installation. The stock audio client uses Linux
PulseAudio and needs a microphone, speaker and camera for live interaction.

```bash
git clone https://github.com/ProActRobot/ProAct-Official.git
cd ProAct-Official
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

On Ubuntu, install the audio system packages:

```bash
sudo apt install libportaudio2 ffmpeg pulseaudio-utils
```

MuJoCo playback additionally uses `mink==0.0.6`, `mujoco==3.3.0`,
`loop-rate-limiters`, `qpsolvers` and `quadprog`. Physical G1 deployment uses the
bundled Unitree SDK plus its CycloneDDS setup; see [client notes](../../src/client/README.md).
Standalone BVH retargeting uses a [separate environment](../../tools/retargeting/README.md).

## Downloaded assets

| Asset | Source | Local variable |
| --- | --- | --- |
| Backbone epoch 600 and ControlNet epoch 800 | [ProAct model release](https://huggingface.co/UsanoCoCr/ProAct) | `MOTION_BASE_CHECKPOINT`, `MOTION_CHECKPOINT` |
| Text-side pipeline and scalers | [Processed SeG archive](https://huggingface.co/datasets/UsanoCoCr/ProAct-Data) | `MOTION_CONTROLNET_DATA_ROOT` |
| Audio-side pipeline and scalers | [Bundled files](../../src/assets/preprocessing/audio/README.md) | Already selected by defaults |
| CLIP ViT-B/32 | [CLIP model](https://huggingface.co/openai/clip-vit-base-patch32) | `MOTION_CLIP_MODEL` |
| Streaming ASR | [Sherpa-ONNX Zipformer](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2) | `SHERPA_MODEL_DIR` |

For the released motion assets, run from the repository root:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(
    "UsanoCoCr/ProAct", revision="b246d61db9f41502b6ed00b142e85461b2a2938c",
    local_dir="checkpoints/ProAct",
)
hf_hub_download(
    "UsanoCoCr/ProAct-Data", "processed_seg.tar.gz", repo_type="dataset",
    revision="09e82a76c12931503e83f1026a40bb4fbc5687d1",
    local_dir="data/ProAct-Data",
)
PY
tar -xzf data/ProAct-Data/processed_seg.tar.gz -C data/ProAct-Data
```

The processed archive includes training features as well as the small text-side
exports required for inference. For offline execution, download CLIP in advance
and set `MOTION_CLIP_MODEL` to its local directory. The GPU acceptance runner
checks the released checkpoint and preprocessing hashes.

## Local configuration

```bash
cp .env.example .env
set -a
. ./.env
set +a
```

Set `OPENAI_API_KEY` for live Realtime and cognition requests. Entry points read
exported environment variables; they do not automatically load `.env`.
`PROACT_SERVER_URL` identifies the motion service. The example provides the
checkpoint and preprocessing paths for the layout above. ASR needs its separately
downloaded model directory; the expected files are listed in the client notes.

The code license, checkpoint CC BY-NC 4.0 terms and dataset terms are separate.
Raw training data and generated recordings remain outside the tracked source.
Continue with [inference](inference.md), [training](training.md) or [evaluation](evaluation.md).
