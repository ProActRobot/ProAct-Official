# Configuration map

Run commands from the repository root. YAML files read the environment variables
in [.env.example](../.env.example); entry points do not load `.env` automatically.

| File | Purpose | Command |
| --- | --- | --- |
| `inference/controlnet.yaml` | Released streaming motion model | `python proact.py serve` |
| `training/backbone.yaml` | Audio backbone | `python proact.py train backbone` |
| `training/controlnet.yaml` | Semantic ControlNet | `python proact.py train controlnet` |
| `preprocessing/seg.yaml` | Per-clip src/motion/text features | `python proact.py prepare seg` |
| `preprocessing/audio.yaml` | Audio/motion features | `python proact.py prepare audio` |
| `legacy/triplet.yaml` | Optional legacy mixed-modality model | Explicit `--mode mix` use |

The released model uses 31 motion channels, 181 audio features per stream,
512-dimensional CLIP conditions and 50 Euler sampling steps. Architecture settings
must match the weights. Preparation outputs belong in separate ignored directories.
