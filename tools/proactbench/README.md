# ProActBench

This component provides 40 scenarios for evaluating when an embodied social agent
should initiate proactive behavior: 30 positive scenarios and 10 negative
scenarios. The JSON files contain dialogue transcripts, stage directions, and
trigger annotations. The scripts validate the dataset, export transcripts,
score recorded model predictions, and check HTTP/WebSocket compatibility with
the inference server.

To evaluate predictions, record your model's decisions for each scenario and
use the scoring commands below. The HTTP/WebSocket checks use synthetic silence
to test the transport protocol.

## Test-environment photos

The original photos captured in the test environment are not publicly available
to protect participants' privacy. Researchers interested in testing with the
original visual observations should contact the authors to discuss testing
arrangements.

## Dependencies

Use Python 3.10 or later. Validation, transcript export, prediction evaluation, and
unit tests use the Python standard library. HTTP/WebSocket checks additionally need:

| Dependency | Version constraint |
| --- | --- |
| `websockets` | `>=13,<17` |

The canonical ProAct server needs Python 3.12 and the dependencies in the main
repository's `requirements.txt`. Dataset validation and scoring do not need that
server environment. Run the commands below from this benchmark directory.

## Installation

Create and activate a virtual environment:

```console
python -m venv .venv
```

Use `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate`
in Bash, then run:

```console
python -m pip install --upgrade pip
python -m pip install -r requirements-smoke.txt
```

## Environment variables

These scripts require no API key. For a separate microphone client, set
`OPENAI_API_KEY` in that client's environment. Set `HF_TOKEN` in the server
environment only when private/gated model downloads require it.

## Validate and export

Run commands from this repository root. The expanded dataset is
`data/proactbench_40.json`; the transcript projection is
`data/proactbench_40_transcripts.json`.
Both files are included. Validation and transcript export require no model,
media, or preprocessing downloads.

```console
python scripts/validate_benchmark.py
python -m unittest discover -s tests -v
python scripts/export_transcripts.py --output transcripts.json
```

To convert a local source script:

```console
python scripts/import_screenplay.py /path/to/source.md --output-dir /path/to/generated
python scripts/validate_benchmark.py --data /path/to/generated/proactbench_40.json
```

## Run HTTP/WebSocket checks

The check uses synthetic silence and the bundled JSON; no model checkpoints,
decoding artifacts, or recorded media are required.

From the integrated repository, start a temporary protocol-only ProAct server:

```console
python scripts/run_protocol_smoke.py --output protocol-smoke.json
```

For another canonical checkout, add `--runtime-root /path/to/ProAct`.
If the server uses a separate Python environment, add `--python /path/to/python`.
To use a running server:

```console
python scripts/run_protocol_smoke.py --server http://127.0.0.1:32558 --output protocol-smoke.json
```

Use `--data /path/to/data.json` to select a dataset and `--audio-seconds` to choose
0.5 through 4 seconds of synthetic silence per spoken turn. The check uses HTTP
turn-completion acknowledgements and nested WebSocket interruption events. Sessions
run sequentially, resetting the global server state before and after each case.
Use a dedicated server. A server started by the script is stopped afterward.

## Evaluate saved predictions

Supply a predictions JSON file recorded from your inference runs. Keep the
bundled `data/proactbench_40.json` available for scoring. The evaluator reads
these JSON files without loading models, camera frames, or audio recordings.

Provide a JSON object with `schema_version: "1.0.0"`, `kind: "cognitive_predictions"`,
and a `runs` array. Each run needs `session_id`, `trial` (1, 2, or 3), and nonempty
`cycles`. Each cycle needs five integer scores (1 through 5) named `visual_scene`,
`user_intent`, `conversation_state`, `social_protocol`, and `emotional_response`,
plus string action fields `DialogueContext`, `GestureIntention`, and
`LocomotionControl`. Use an empty string for an absent action. To include interruption
checks, add a boolean `should_interrupt` to each cycle; it defaults to `false`.

```console
python scripts/evaluate_predictions.py predictions.json --output decision-metrics.json
```

Supply all 40 sessions and three trials each. Use `--allow-partial` for an incomplete
input file. Protocol-check output is not an input to this evaluator.
