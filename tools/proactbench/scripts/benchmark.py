"""Shared dataset checks and transcript projection; standard library only."""

import json
from pathlib import Path
import re

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "proactbench_40.json"
DIMENSIONS = (
    "visual_scene",
    "user_intent",
    "conversation_state",
    "social_protocol",
    "emotional_response",
)


def transcripts(dataset):
    result = []
    for session in dataset["sessions"]:
        turns = {}
        for event in session["events"]:
            if event["type"] == "speech":
                turn = turns.setdefault(
                    event["turn_id"], {"speaker": event["speaker"], "parts": []}
                )
                turn["parts"].append(event["text"])
        result.append(
            {
                "session_id": session["session_id"],
                "transcript": [
                    {"speaker": turn["speaker"], "text": " ".join(turn["parts"])}
                    for turn in turns.values()
                ],
            }
        )
    return result


def validate(data):
    """Reject malformed fixtures and changes that invalidate this 40-case release."""

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(data.get("schema_version") == "1.0.0", "Unsupported schema_version")
    sessions = data.get("sessions", [])
    ids = [session["session_id"] for session in sessions]
    require(
        ids == [f"SES-{i:02d}" for i in range(1, 41)],
        "Expected ordered, unique SES-01 through SES-40",
    )
    require(data["source"]["session_count"] == 40, "Incorrect source session_count")
    require(
        bool(re.fullmatch("[a-f0-9]{64}", data["source"]["sha256"])),
        "Invalid source hash",
    )
    positives = 0
    for session in sessions:
        sid = session["session_id"]
        annotation = session["trigger_annotation"]
        require(
            type(annotation["should_trigger"]) is bool,
            f"{sid}: Boolean ground truth required",
        )
        require(
            session["design_label"] in {"CD", "MN", "FA-trap"},
            f"{sid}: Invalid design label",
        )
        require(
            annotation["should_trigger"] == (session["design_label"] != "FA-trap"),
            f"{sid}: Inconsistent label",
        )
        positives += annotation["should_trigger"]
        require(
            bool(session["setting"] and session["initial_state"]),
            f"{sid}: Missing scene context",
        )
        require(
            bool(annotation["decision_point"]["raw"]), f"{sid}: Missing decision point"
        )
        require(
            annotation["decision_point"]["kind"]
            in {"absolute_time", "script_turn", "event_condition"},
            f"{sid}: Invalid anchor",
        )
        require(
            set(annotation["dimensions"]) <= set(DIMENSIONS),
            f"{sid}: Unknown dimension",
        )
        require(
            len(set(annotation["dimensions"])) == len(annotation["dimensions"]),
            f"{sid}: Repeated dimension",
        )
        if annotation["should_trigger"]:
            require(
                bool(annotation["dimensions"]), f"{sid}: Positive missing dimensions"
            )
            require(
                type(annotation["urgency"]) is int and 1 <= annotation["urgency"] <= 5,
                f"{sid}: Invalid urgency",
            )
            require(annotation["trap"] is None, f"{sid}: Unexpected trap")
        else:
            require(
                all(
                    annotation["trap"][k]
                    for k in ("surface_cue", "true_context", "correct_behavior")
                ),
                f"{sid}: Incomplete trap",
            )
        require(
            set(annotation["paired_session_ids"]) <= set(ids) - {sid},
            f"{sid}: Invalid pair reference",
        )
        require(
            not annotation["unparsed_lines"],
            f"{sid}: Unparsed annotations require review",
        )
        require(
            bool(annotation["source_fields"]), f"{sid}: Missing original annotations"
        )
        events = session["events"]
        require(
            [e["event_id"] for e in events]
            == [f"{sid}-E{i:03d}" for i in range(1, len(events) + 1)],
            f"{sid}: Invalid event IDs",
        )
        require(
            any(e["type"] == "stage_direction" for e in events),
            f"{sid}: Missing stage directions",
        )
        require(
            bool(
                session["source"]["script_lines"]
                and session["source"]["annotation_lines"]
            ),
            f"{sid}: Missing source preservation",
        )
        prior_line = 0
        speakers = {}
        for event in events:
            require(
                event["type"] in {"speech", "stage_direction", "speech_placeholder"},
                f"{sid}: Invalid event type",
            )
            require(bool(event["text"].strip()), f"{sid}: Empty event")
            require(
                type(event["source_line"]) is int
                and event["source_line"] >= prior_line,
                f"{sid}: Nonmonotonic source lines",
            )
            prior_line = event["source_line"]
            if event["turn_id"]:
                require(
                    bool(re.fullmatch(re.escape(sid) + r"-T\d{2}", event["turn_id"])),
                    f"{sid}: Invalid turn ID",
                )
                require(
                    event["speaker"] in {"agent", "user"}, f"{sid}: Invalid speaker"
                )
                require(
                    speakers.setdefault(event["turn_id"], event["speaker"])
                    == event["speaker"],
                    f"{sid}: Turn speaker changed",
                )
            else:
                require(
                    event["type"] == "stage_direction" and event["speaker"] is None,
                    f"{sid}: Unattributed speech",
                )
            if event["type"] == "speech":
                require(
                    not any(c in event["text"] for c in "()（）"),
                    f"{sid}: Stage direction leaked into speech",
                )
            if event["type"] == "stage_direction":
                require(
                    type(event["proactive_marker"]) is bool
                    and type(event["annotation_only"]) is bool,
                    f"{sid}: Missing stage classification",
                )
                require(
                    isinstance(event["gesture_labels"], list),
                    f"{sid}: Missing gesture labels",
                )
                if event["time"] is not None:
                    require(event["time"]["seconds"] >= 0, f"{sid}: Negative timestamp")
    require(positives == 30, "Expected 30 positive and 10 negative scenarios")
    projected = transcripts(data)
    require(
        sum(len(s["transcript"]) for s in projected) == 152, "Expected 152 spoken turns"
    )
    return {
        "sessions": len(sessions),
        "positive": positives,
        "negative": len(sessions) - positives,
        "events": sum(len(s["events"]) for s in sessions),
        "stage_directions": sum(
            e["type"] == "stage_direction" for s in sessions for e in s["events"]
        ),
        "spoken_turns": 152,
    }


def load(path=DEFAULT_DATA):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(data)
    return data


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
