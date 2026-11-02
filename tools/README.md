# Tools and checks

| Directory | Contents |
| --- | --- |
| `proactbench/` | 40-session dataset, saved-prediction scoring and protocol checks |
| `retargeting/` | BVH-to-G1 conversion with a separate environment |
| `checks/` | Transport smoke and trained-GPU acceptance |
| `tests/` | Main-system regression tests |

Common commands from the repository root:

```bash
python proact.py check
python proact.py benchmark validate
python proact.py benchmark evaluate predictions.json --output runtime/metrics.json
python proact.py validate-gpu --gpu 0 --with-text
```

Retargeting installation and usage are documented in [its guide](retargeting/README.md).
