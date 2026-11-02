# Inference

Complete [setup](setup.md) and export `.env` first. Run commands from the repository root.

## Motion server

```bash
CUDA_VISIBLE_DEVICES=0 python proact.py serve --host 127.0.0.1 --port 32558
```

The launcher starts one distributed GPU process using the canonical
`src/server/motion_server.py`, ControlNet mode, one-second audio uploads and two-second
turn-completion padding. Choose a bind address appropriate for the client host.
Server options such as `--port`, `--config` and `--mogen_padding` are forwarded.
Use `python proact.py --dry-run serve` to inspect the full command.

The model reads `configs/inference/controlnet.yaml`. Bundled audio scalers are
selected by default; checkpoint, CLIP and SeG paths come from the environment.
The [configuration map](../../configs/README.md) describes the remaining files.

## Interaction client

In another terminal with the same environment:

```bash
python proact.py interact --need_slow_thinking --asr --allow-interrupt --render_target mujoco
```

Set `PROACT_SERVER_URL` when the server is on another host. `--render_target none`
uses audio interaction without a simulator. Omit `--need_slow_thinking` for
reactive interaction and omit `--allow-interrupt` to disable user interruption.
Physical control uses `--real` on the configured robot host; deployment details
are in [src/client/README.md](../../src/client/README.md).

Both systems use the shared scenario in `src/prompts/scenario.txt` unless overridden
by `PROACT_SCENARIO`. The cognitive prompt files are loaded without appended
behavioral instructions. Encoder and Planner run in parallel; the runtime
preserves their three action channels and queues non-urgent dialogue separately
from urgent interruption.

## Validate the trained motion path

```bash
python proact.py validate-gpu --gpu 0 --with-text --output runtime/gpu-validation
```

This starts a dedicated loopback server, uses actual trained weights, checks
audio/text motion and client inverse transforms, exercises interruption/reset,
and stops the server. It needs the model/CLIP/SeG paths but no live API or devices.
Use `--asset-config /path/to/assets.json` for a local asset-path configuration.
Results and generated artifacts are retained under the ignored output directory.

For a server protocol check without motion weights:

```bash
python proact.py benchmark protocol --output runtime/proactbench-protocol.json
```

See [checks and evaluation](evaluation.md) for the available test commands.
