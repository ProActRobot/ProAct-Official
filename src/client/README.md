# Client runtime

Run the client from the repository root. Python 3.12 and the packages in
`requirements.txt` provide the common audio, networking, ASR, and OpenAI
runtime. The stock microphone capture uses Linux PulseAudio; install `pulseaudio-utils` for `parec` and ensure a working audio session. The client writes recordings, generated motion, and logs below
`runtime/client_media/` by default; set `PROACT_MEDIA_DIR` to keep those files
elsewhere. Runtime output is ignored by Git.

## Entry point

The canonical entry point is `src/client/realtime_client.py`:

```bash
python src/client/realtime_client.py \
  --need_slow_thinking \
  --asr \
  --allow-interrupt \
  --render_target mujoco
```

Use `--render_target none` for audio and conversation without a renderer, or
`--render_target blender` for the Blender TCP client. Add `--real` only on a
configured Unitree G1 host. Omit `--need_slow_thinking` for the reactive
baseline and `--allow-interrupt` to disable user interruption of agent speech.
The server is expected at `http://localhost:32558` by default and must expose
`/api/v1/time`, `/api/v1/set_zero_time`, `/api/v1/stream`, and `/ws/motion`.

Set credentials in the shell or a private environment file; entry points do
not load `.env` automatically:

```bash
export OPENAI_API_KEY="..."
export PROACT_SERVER_URL="http://localhost:32558"
export PROACT_SCENARIO="Describe the current task and interaction context."
export PROACT_COGNITIVE_MODEL="gpt-4.1-mini-2025-04-14"
python src/client/realtime_client.py --need_slow_thinking --render_target none
```

`SLOW_THINKING_API_KEY` and `SLOW_THINKING_BASE_URL` override the cognitive
credentials and endpoint. Otherwise the cognitive loop uses
`OPENAI_API_KEY` and `OPENAI_BASE_URL`. `PROACT_SCENARIO` is shared by the
realtime client and `src/client/cognition.py`. Without an override, both use the
Appendix D integrated scenario in `src/prompts/scenario.txt`. Realtime uses that
scenario alone as its base instructions; cognition loads Appendix F.1/F.2
without appending instructions, and dialogue interventions pass through the
Planner's output directly.

## External assets

Matching audio preprocessing exports are bundled in `src/assets/preprocessing/audio/`.
The text-side exports are provided by the public processed SeG archive. The
motion listener loads four joblib files lazily when the first motion packet
arrives:

```bash
export PROACT_DATA_PIPE="$PWD/assets/preprocessing/audio/data_pipe_30fps.sav"
export PROACT_OUTPUT_SCALER="$PWD/assets/preprocessing/audio/output_scaler.sav"
export PROACT_TEXT_DATA_PIPE="$PWD/data/ProAct-Data/processed_seg/data_pipe_30fps.sav"
export PROACT_TEXT_OUTPUT_SCALER="$PWD/data/ProAct-Data/processed_seg/output_scaler.sav"
```

These paths are the client defaults relative to the checkout. The bundled
audio exports match the released checkpoint pair and include a SHA-256 manifest.
Each `*_DATA_PIPE` must be a joblib/scikit-learn pipeline with
`inverse_transform`; when root continuity is required it should expose the
root step as `named_steps["root"]`. Each `*_OUTPUT_SCALER` must provide
`inverse_transform` for the server's 31-dimensional G1 motion features. The
inverse pipeline must return the joint columns consumed by
`save_to_csv()` in `realtime_client.py` (for example,
`waist_yaw_Zrotation` and `left_shoulder_pitch_Yrotation`). For newly trained
models, generate matching exports with `src/motion/data_process/` and
override the four variables. Keep generated datasets and interaction recordings
outside Git; the three named inference exports are explicitly tracked.

Both historical exports referring to `pymo.*` and newer exports referring to
`data_process.pymo.*` load from the retained preprocessing package. Exported
feature dimensions and joint layouts must still match the selected checkpoint.

For `--asr`, download a Sherpa-ONNX Zipformer model and set
`SHERPA_MODEL_DIR`. The directory must contain `tokens.txt`,
`encoder-epoch-34-avg-19.onnx`, `decoder-epoch-34-avg-19.onnx`, and
`joiner-epoch-34-avg-19.onnx`. Without this directory the client still starts,
but the ASR worker remains disabled.

The MuJoCo renderer uses the tracked model at
`src/assets/g1/scene_empty_coll.xml` and its accompanying meshes; no
neutral BVH is required. Blender uses
`src/client/blender_conjugate_config.json` by default. Set
`BLENDER_CONJUGATE_REF_BVH` only when generating a new conjugation config,
and set `BLENDER_STREAM_HOST`, `BLENDER_STREAM_PORT`,
`BLENDER_FRAME_INTERVAL`, or `BLENDER_AUDIO_CHUNK_SECONDS` to match the local
Blender receiver.

## Optional environments

The common requirements do not include platform-specific renderer and robot
packages. Install the simulation and retargeting extras before using MuJoCo
or `src/client/retarget.py`:

```bash
pip install "mink==0.0.6" mujoco loop-rate-limiters qpsolvers quadprog
```

The client uses the public `mink` 0.0.6 API; pin this version because newer
releases have changed the Lie-group and task APIs used by the retained
retargeter.

`pygame` enables the optional Gantt/state visualizers and `pynput` enables the
keyboard listener; both are optional for `--render_target none`. The physical
G1 path uses the patched runtime under `src/third_party/unitree_sdk2py`.
That runtime differs from the current official SDK in its channel lifecycle,
G1 locomotion service, audio streaming call, and RPC binary helper, so do not
replace it with the latest official package without porting those changes.
Install the official SDK's CycloneDDS 0.10.x prerequisite following
the [Unitree SDK2 setup](https://github.com/unitreerobotics/unitree_sdk2_python), then set
`PROACT_UNITREE_INTERFACE` to the configured network interface. Set
`PROACT_AUDIO_DIR` and `PROACT_MOTION_DIR` when running either renderer
module directly. The hardware camera path is enabled with
`PROACT_ROBOT_SSH_HOST` and optional `PROACT_ROBOT_SSH_USER`.

Useful tuning variables include `OPENAI_REALTIME_MODEL`,
`OPENAI_REALTIME_VOICE`, `PROACT_CLIENT_ID`, `PROACT_IMAGE_CAPTURE_INTERVAL`,
`PROACT_IMAGE_SEND_INTERVAL`, `PROACT_USER_SPEECH_GRACE_SEC`,
`PROACT_USER_SPEECH_RMS_THRESHOLD`, `PROACT_REPLY_COOLDOWN_SEC`, and
`PROACT_RESPONSE_TIMEOUT_SEC`.
