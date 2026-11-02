"""Evaluate external Cognitive System predictions, never reference stage actions."""

import argparse
import json
from pathlib import Path
from benchmark import DEFAULT_DATA, DIMENSIONS, load, write_json

CHANNELS = ("DialogueContext", "GestureIntention", "LocomotionControl")


def evaluate(dataset, predictions, allow_partial=False):
    if (
        predictions.get("kind") != "cognitive_predictions"
        or predictions.get("schema_version") != "1.0.0"
    ):
        raise ValueError(
            "Expected cognitive_predictions format; protocol smoke reports are not predictions"
        )
    truth = {
        s["session_id"]: s["trigger_annotation"]["should_trigger"]
        for s in dataset["sessions"]
    }
    required = {(sid, trial) for sid in truth for trial in (1, 2, 3)}
    seen, counts, violations = set(), dict(TP=0, FP=0, FN=0, TN=0), []
    decisions = []
    for record in predictions["runs"]:
        key = (record["session_id"], record["trial"])
        if type(record["trial"]) is not int or key not in required or key in seen:
            raise ValueError(f"Unknown, duplicate or invalid run: {key}")
        seen.add(key)
        cycles = record["cycles"]
        # No cycles means missing observations, not an observed negative decision.
        if not isinstance(cycles, list) or not cycles:
            raise ValueError(
                f"{key}: At least one observed cognitive cycle is required"
            )
        triggered = False
        for index, cycle in enumerate(cycles):
            scores = cycle["scores"]
            if (
                not isinstance(scores, dict)
                or set(scores) != set(DIMENSIONS)
                or any(type(v) is not int or not 1 <= v <= 5 for v in scores.values())
            ):
                raise ValueError(f"{key}: Five integer scores in [1, 5] are required")
            actions = cycle["actions"]
            if (
                not isinstance(actions, dict)
                or set(actions) != set(CHANNELS)
                or any(not isinstance(v, str) for v in actions.values())
            ):
                raise ValueError(
                    f"{key}: Three action channels must be explicit strings; empty means no action"
                )
            maximum = max(scores.values())
            nonempty = any(value.strip() for value in actions.values())
            triggered |= maximum >= 4 and nonempty
            should_interrupt = cycle.get("should_interrupt", False)
            if type(should_interrupt) is not bool:
                raise ValueError(f"{key}: should_interrupt must be a boolean")
            if should_interrupt and maximum < 5:
                violations.append(
                    {
                        "session_id": key[0],
                        "trial": key[1],
                        "cycle": index,
                        "policy": "interruption_requires_score_5",
                    }
                )
            if nonempty and maximum < 4:
                violations.append(
                    {
                        "session_id": key[0],
                        "trial": key[1],
                        "cycle": index,
                        "policy": "action_below_trigger_threshold",
                    }
                )
        actual = truth[key[0]]
        counts[
            "TP"
            if triggered and actual
            else "FP"
            if triggered
            else "FN"
            if actual
            else "TN"
        ] += 1
        decisions.append(
            {
                "session_id": key[0],
                "trial": key[1],
                "triggered": triggered,
                "expected": actual,
            }
        )
    if not seen:
        raise ValueError("No observed runs supplied")
    missing = sorted(required - seen)
    if missing and not allow_partial:
        raise ValueError(
            f"Incomplete coverage: missing {len(missing)} of 120 runs; use --allow-partial for diagnostics only"
        )
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    return {
        "kind": "cognitive_evaluation",
        "dataset_id": dataset["dataset_id"],
        "dataset_source_sha256": dataset["source"]["sha256"],
        "complete_coverage": not missing,
        "observed_runs": len(seen),
        "required_runs": 120,
        "missing_runs": [list(k) for k in missing],
        "counts": counts,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "policy_violations": violations,
        "decisions": decisions,
        "scope": "Decision counts only. Latency and human ratings are outside this output.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    write_json(args.output, evaluate(load(args.data), predictions, args.allow_partial))
