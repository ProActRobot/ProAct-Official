# ProAct source

| Module | Responsibility |
| --- | --- |
| `client/` | Realtime interaction, cognition, interruption and playback |
| `server/` | Streaming HTTP/WebSocket motion service |
| `motion/` | Model, training, feature processing and samplers |
| `prompts/` | Shared scenario and cognitive/annotation prompts |
| `assets/` | G1 geometry and matching audio preprocessing exports |
| `third_party/` | Bundled Unitree SDK |

Use the root `proact.py` launcher for standard commands. It sets the source
paths while keeping downloaded data and generated output at the repository root.
See the [code map](../docs/guides/structure.md) for implementation details.
