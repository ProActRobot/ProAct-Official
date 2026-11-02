# Data preparation and training

Use the main [environment](setup.md). Run commands from the repository root.
Preprocessing and training settings live in `configs/preprocessing/` and
`configs/training/`. Generated files belong under ignored `data/` and `experiments/` paths.

## SeG annotations

Set `MOTION_SEG_RAW_ROOT` to a directory with `bvh/` and `text.json`.
Every annotation record identifies one clip and contains all three captions:

```json
{
  "file_name": "HAND_WAG-2.bvh",
  "label": "HAND_WAG",
  "annotation_intent": "The person rejects a suggestion.",
  "annotation_structure_pose": "The right hand is raised near the shoulder.",
  "annotation_structure_motion": "The right hand moves from side to side."
}
```

`file_name` matches the BVH basename. Clips in the same category keep independent
captions. Text caches are stored at `text/by_clip/<clip>.npz`, including the
caption strings and feature names. Unannotated motions are reported and skipped;
conflicting duplicates and incomplete triples are rejected.

```bash
python proact.py prepare seg
```

Use `python proact.py prepare seg --config /path/to/seg.yaml` for another YAML.
For the default 30 FPS / 5-second standardized layout, outputs are placed under
`$MOTION_SEG_PROCESSED_DIR/feat_30fps_5s_TEXT_WithStd`. Point
`MOTION_CONTROLNET_DATA_ROOT` at that directory when training on regenerated data.
Use the matching published preprocessing exports with the released checkpoints.

The retained [gesture annotation prompt](../../src/prompts/gesture_annotation.txt)
supports the paper's annotation process. Rendering and Gemini annotation need
the source clips and the corresponding external service setup.

## Audio and motion preparation

Set `MOTION_RAW_DATA_ROOT` to the expected retargeted audio/motion/state inputs,
then run:

```bash
python proact.py prepare audio
```

The configuration selects 16 kHz audio, 30 FPS motion and 5-second windows.
Default outputs are written under
`$MOTION_AUDIO_PROCESSED_DIR/feat_30fps_5s_WithExp_MFCC_WithStd`.
Set `MOTION_BACKBONE_DATA_ROOT` to that feature directory for backbone training.
BVH-to-G1 conversion is described in the [retargeting guide](../../tools/retargeting/README.md).

The public data repository includes processed SeG and a retargeted Photoreal
subset. Prepare any additional audio-motion training data in the input format
expected by the preprocessing configuration. Convert raw Photoreal data to the
retargeted BVH, audio and state layout before preprocessing.

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1 python proact.py --processes 2 train backbone
```

After choosing the resulting backbone checkpoint:

```bash
export MOTION_BASE_CHECKPOINT=/path/to/backbone/weights/model.pt
CUDA_VISIBLE_DEVICES=0,1 python proact.py --processes 2 train controlnet
```

ControlNet training uses SeG caption variants, Gaussian audio conditions and a
frozen base model. The model-level audio conditional dropout is configured in
the YAML.

Inference uses the pipeline/scalers paired with its selected weights. Newly
trained models need their own matching exports; the bundled audio files belong
to the released epoch-600/800 pair.
