"""One-time, local conversion of the supplied screenplay to release fixtures."""

from pathlib import Path
import hashlib
import json
import re

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
SOURCE = args.source
DEST = args.output_dir
DIMENSIONS = [
    "visual_scene",
    "user_intent",
    "conversation_state",
    "social_protocol",
    "emotional_response",
]
GESTURES = [
    "Arms Welcome",
    "HAND SHAKE",
    "FOREFINGER POINT",
    "PALMS UP-HIGH",
    "FINGERS BECKON",
    "HANDS CROSS",
    "NECK SCRATCH",
    "MOUTH CLASP",
    "HEAD NOD",
]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def paired_ids(value):
    # The source uses both SES-05, SES-15 and compressed SES-05/15 notation.
    found = []
    for match in re.finditer(r"SES-(\d{2})(/\d{2})*", value):
        for number in re.findall(r"\d{2}", match.group()):
            sid = "SES-" + number
            if sid not in found:
                found.append(sid)
    return found


def timestamp(value):
    match = re.search(r"\bt\s*([=≈])\s*(\d+(?:\.\d+)?)s\b", value)
    return (
        None
        if match is None
        else {"seconds": float(match[2]), "approximate": match[1] == "≈"}
    )


def split_parts(value):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(value):
        if char in "(（":
            if depth == 0:
                if value[start:index].strip():
                    parts.append(("speech", value[start:index].strip()))
                start = index
            depth += 1
        elif char in ")）" and depth:
            depth -= 1
            if depth == 0:
                parts.append(("stage_direction", value[start : index + 1]))
                start = index + 1
    assert depth == 0, value
    if value[start:].strip():
        parts.append(("speech", value[start:].strip()))
    return parts


text = SOURCE.read_text(encoding="utf-8")
headings = list(re.finditer(r"^### SES-(\d{2})\b.*$", text, re.M))
sessions = []
for number, heading in enumerate(headings):
    end = headings[number + 1].start() if number + 1 < len(headings) else len(text)
    section = text[heading.end() : end]
    code_match = re.search(r"```text\s*\n(.*?)```", section, re.S)
    code_start = heading.end() + code_match.start(1)
    first_line = text[:code_start].count("\n") + 1
    code = code_match[1]
    before, script_rest = code.split("—— 台本 ——", 1)
    annotation_match = re.search(
        r"^——.*(?:ProAct Trigger|GT: NO TRIGGER).*$", script_rest, re.M
    )
    script = script_rest[: annotation_match.start()]
    annotation_raw = script_rest[annotation_match.start() :]
    script_start = code_start + code.index("—— 台本 ——") + len("—— 台本 ——")
    script_first_line = text[:script_start].count("\n") + 1
    sid = "SES-" + heading[1]
    label = (
        "FA-trap"
        if "FA-trap" in heading[0]
        else ("MN" if re.search(r"\bMN\b", heading[0]) else "CD")
    )
    expected = label != "FA-trap"
    fields, unparsed = [], []
    trigger_raw = None
    for line in annotation_raw.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Trigger @"):
            trigger_raw = stripped.removeprefix("Trigger @").strip()
            continue
        match = re.match(r"(?:·\s*)?([^:：]+)[:：]\s*(.*)", stripped)
        if match:
            fields.append({"key": match[1].strip(), "value": match[2].strip()})
        elif line[:1].isspace() and fields:
            fields[-1]["value"] += "\n" + stripped
        else:
            unparsed.append(stripped)
    field_map = {field["key"]: field["value"] for field in fields}
    if "GT_should_trigger" in field_map:
        assert (field_map["GT_should_trigger"] == "True") == expected
    decision_raw = trigger_raw or field_map.get("Primary trap window")
    # Earlier negative cases only identify the cue in surface_signal.
    decision_raw = decision_raw or field_map.get("surface_signal")
    assert decision_raw, sid
    turn_match = re.match(r"turn\s+(\d+)\b", decision_raw)
    time = timestamp(decision_raw)
    anchor = {
        "raw": decision_raw,
        "kind": "absolute_time"
        if time
        else ("script_turn" if turn_match else "event_condition"),
        "time": time,
        "script_turn": int(turn_match[1]) if turn_match else None,
    }
    events, active, turn = [], None, 0
    for row, raw in enumerate(script.splitlines()):
        value = raw.strip()
        if not value:
            continue
        match = re.match(r"^(Agent|User|Visitor)\s*[:：]\s*(.*)", value)
        if match:
            turn += 1
            active = {
                "speaker": "agent" if match[1] == "Agent" else "user",
                "source_speaker": match[1],
                "turn_id": f"{sid}-T{turn:02d}",
            }
            value = match[2]
        elif not (raw[:1].isspace() and active):
            active = None
        if active:
            parts = split_parts(value)
        else:
            parts = [("stage_direction", value)]
        for kind, part in parts:
            if kind == "speech" and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", part):
                kind = "speech_placeholder"
            event = {
                "event_id": f"{sid}-E{len(events) + 1:03d}",
                "type": kind,
                "text": part,
                "source_line": script_first_line + row,
            }
            event.update(
                active or {"speaker": None, "source_speaker": None, "turn_id": None}
            )
            if kind == "stage_direction":
                event["time"] = timestamp(part)
                event["gesture_labels"] = [
                    gesture for gesture in GESTURES if gesture in part
                ]
                event["proactive_marker"] = "ProAct" in part
                event["annotation_only"] = (
                    "baseline" in part or "快系统" in part or "漏判" in part
                )
            events.append(event)
    urgency_raw = field_map.get("urgency")
    urgency = int(re.match(r"\d", urgency_raw)[0]) if urgency_raw else None
    pair_text = "\n".join(
        field["value"]
        for field in fields
        if field["key"] in {"paired_with", "Paired positive"}
    )
    pair_heading = heading[0].split("配对", 1)[1] if "配对" in heading[0] else ""
    pairs = paired_ids(pair_text + "\n" + pair_heading)
    sessions.append(
        {
            "session_id": sid,
            "title_source": heading[0].removeprefix("### "),
            "design_label": label,
            "setting": re.search(r"^Setting:\s*(.*)", before, re.M)[1],
            "initial_state": re.search(r"^Initial State:\s*(.*)", before, re.M)[1],
            "events": events,
            "trigger_annotation": {
                "should_trigger": expected,
                "decision_point": anchor,
                "dimensions": [
                    dim for dim in DIMENSIONS if dim in field_map.get("dimension", "")
                ],
                "urgency": urgency,
                "urgency_source": urgency_raw,
                "signal": field_map.get(
                    "signal",
                    field_map.get("surface_signal", field_map.get("Surface cue")),
                ),
                "reference_actions": {
                    name: field_map.get(name)
                    for name in ["gesture", "locomotion", "dialogue_intent"]
                },
                "trap": None
                if expected
                else {
                    "surface_cue": field_map.get(
                        "surface_signal", field_map.get("Surface cue")
                    ),
                    "true_context": field_map.get(
                        "true_context", field_map.get("True context")
                    ),
                    "correct_behavior": field_map.get(
                        "正确行为", field_map.get("Correct behavior")
                    ),
                    "fooled_setting": field_map.get("fooled_setting"),
                    "hint": field_map.get("trap_hint"),
                },
                "paired_session_ids": pairs,
                "unresolved_pair_reference": pair_heading.strip()
                if pair_heading and not paired_ids(pair_heading)
                else None,
                "source_fields": fields,
                "unparsed_lines": unparsed,
            },
            "source": {
                "heading_line": text[: heading.start()].count("\n") + 1,
                "block_first_line": first_line,
                "script_lines": script.splitlines(),
                "annotation_lines": annotation_raw.splitlines(),
            },
        }
    )
dataset = {
    "schema_version": "1.0.0",
    "dataset_id": "proactbench-scripted-40",
    "description": "Scripted scenarios, reference dialogue, stage directions and decision annotations; not recorded multimodal sessions or model predictions.",
    "source": {
        "filename": SOURCE.name,
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "languages": ["en", "zh"],
        "session_count": 40,
    },
    "sessions": sessions,
}
assert len(sessions) == 40
assert sum(s["trigger_annotation"]["should_trigger"] for s in sessions) == 30
dump(DEST / "proactbench_40.json", dataset)
print(
    json.dumps(
        {
            "sessions": len(sessions),
            "events": sum(len(s["events"]) for s in sessions),
            "positive": 30,
            "negative": 10,
            "unparsed": {
                s["session_id"]: s["trigger_annotation"]["unparsed_lines"]
                for s in sessions
                if s["trigger_annotation"]["unparsed_lines"]
            },
        }
    )
)
