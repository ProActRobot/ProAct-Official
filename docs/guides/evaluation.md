# Checks and evaluation

Run commands from the repository root in the main environment.

## Offline checks

```bash
python proact.py check
```

This runs main-system tests, Benchmark tests and dataset
validation. It needs the installed Python dependencies but no GPU, API key,
microphone or robot. Retargeting tests run in their separate environment:

```bash
cd tools/retargeting
python -m pytest -q
python -m ruff check .
```

## ProActBench

The benchmark includes 40 sessions, transcript projections and trigger labels.

```bash
python proact.py benchmark validate
python proact.py benchmark protocol --output runtime/proactbench-protocol.json
python proact.py benchmark evaluate predictions.json --output runtime/decision-metrics.json
```

`protocol` starts a dedicated no-motion server and sends synthetic silence to
check service contracts. `evaluate` reads actual saved cognitive predictions;
protocol output and ground-truth stage directions cannot substitute for them.
The default scoring protocol requires all 40 sessions and three trials each.

Prediction schema, partial diagnostics and external-server options are documented
in [tools/proactbench/README.md](../../tools/proactbench/README.md).
Original environment images are not publicly distributed. Matched multimodal
replay requires the corresponding permitted inputs and a replay runner.

## Trained GPU acceptance

```bash
python proact.py validate-gpu --gpu 0 --with-text
```

This checks the released model and preprocessing hashes, motion packets,
inverse-transformed CSVs, interruption and reset. It does not call a live
reasoning API or compute cognitive accuracy. Timings measure the motion
generation function, excluding the full sensor/API/network/playback chain.

## Motion diagnostics

`src/motion/cal_fid.py` and `src/motion/cal_fid_w_time.py` provide diagnostic
Fréchet distance and diversity calculations using temporal-mean and flattened
motion features, respectively. Use the same feature definition for generated
and reference motion when interpreting these diagnostics.
