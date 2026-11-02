"""Check model request boundaries and paper-to-runtime conversion offline."""
import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from client.cognition import (
    CURRENT_SCENARIO_SLOW, MemorySystemResult, PaperDecisionResponse, PaperLocomotionAction,
    PaperMotivationAssessment, SlowThinkingConfig, SlowThinkingSystem,
)

ROOT = Path(__file__).resolve().parents[2]


def paper_plan(action="turn", direction="left", magnitude=45):
    return PaperDecisionResponse(
        motivation_eval=PaperMotivationAssessment(
            reasoning="History check: no previous turn toward the poster.",
            tracking_target="poster", locomotion_reasoning="Face the requested poster.",
            visual_scene=4, user_intent=1, conversation_state=1,
            social_protocol=1, emotional_response=1,
        ),
        should_interrupt=True, motion_intent="", dialogue_prefill="The poster is here.",
        locomotion=[PaperLocomotionAction(
            action=action, direction=direction, magnitude=magnitude,
            tracking_target="poster", reasoning="Face the requested poster.",
        )],
    )


def test_cognitive_api_receives_only_paper_instructions(monkeypatch):
    system = SlowThinkingSystem(lambda: None, lambda result: None,
                                config=SlowThinkingConfig(api_key="offline-test"))
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        parsed = (MemorySystemResult(**{
            key: "Memory sentence." for key in MemorySystemResult.model_fields
        }) if len(requests) == 1 else paper_plan())
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    monkeypatch.setattr(system._client.beta.chat.completions, "parse", parse)
    try:
        system._call_memory_system("DIALOGUE_DATA", [], "MEMORY_DATA")
        result = system._call_decision_system("NEW_DIALOGUE_DATA", "MEMORY_DATA", [])
        for request, filename in zip(requests, ["context_encoder.txt", "behavior_planner.txt"]):
            expected = (ROOT / "src/prompts" / filename).read_text().replace(
                "{scenario_prompt}", CURRENT_SCENARIO_SLOW
            )
            assert request["messages"][0] == {"role": "system", "content": expected}
            assert '"description":' not in json.dumps(request["response_format"].model_json_schema())
        assert requests[0]["messages"][1]["content"] == [{
            "type": "text",
            "text": "## PREVIOUS MEMORY (t-1)\nMEMORY_DATA\n\n## DIALOGUE TRANSCRIPTS\nDIALOGUE_DATA",
        }]
        assert requests[1]["messages"][1]["content"] == [{
            "type": "text", "text": "MEMORY_DATA\n## RECENT DIALOGUE TRANSCRIPT\nNEW_DIALOGUE_DATA",
        }]
        assert requests[1]["response_format"] is PaperDecisionResponse
        assert result.locomotion[0].action == "turn_left"
        assert result.motivation_eval.motivation_score == 4
        assert not result.should_interrupt
    finally:
        system.stop(join_timeout=1)


@pytest.mark.parametrize("action,direction,magnitude,expected", [
    ("move", "forward", 0.5, "move_forward"),
    ("move", "backward", 1.5, "move_backward"),
    ("turn", "left", 45, "turn_left"),
    ("turn", "right", 180, "turn_right"),
])
def test_paper_locomotion_converts_without_prompt_adapter(action, direction, magnitude, expected):
    result = paper_plan(action, direction, magnitude).to_runtime()
    assert result.locomotion[0].action == expected
    assert result.locomotion[0].parameter == magnitude


@pytest.mark.parametrize("action,direction,magnitude", [
    ("move", "left", 0.5), ("turn", "forward", 45),
    ("move", "forward", 1.6), ("turn", "left", 181),
])
def test_invalid_paper_movement_is_rejected(action, direction, magnitude):
    with pytest.raises(ValidationError):
        paper_plan(action, direction, magnitude)


def test_legacy_metadata_is_not_model_output_or_previous_decision():
    from client.cognition import MemoryBank

    schema = PaperDecisionResponse.model_json_schema()
    assert set(schema["properties"]) == {
        "motivation_eval", "should_interrupt", "motion_intent", "locomotion", "dialogue_prefill",
    }
    assert set(schema["$defs"]["PaperMotivationAssessment"]["properties"]) == {
        "reasoning", "tracking_target", "locomotion_reasoning", "visual_scene",
        "user_intent", "conversation_state", "social_protocol", "emotional_response",
    }
    bank = MemoryBank()
    bank.update_from_decision_system(paper_plan().to_runtime())
    assert set(bank.previous_decision) == set(schema["properties"])
    assert bank.previous_decision["locomotion"][0]["action"] == "turn"


@pytest.fixture
def realtime_client(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src/client"))
    return importlib.import_module("realtime_client")


def test_realtime_instructions_and_prefill_have_no_extra_prompt(realtime_client):
    client = realtime_client
    scenario = "SCENARIO_DATA"
    assert client.get_system_instructions(scenario) == scenario
    assert client.REALTIME_SETTINGS["instructions"] == client.CURRENT_SCENARIO
    assert client.agent.instructions == client.CURRENT_SCENARIO
    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(model=SimpleNamespace(send_event=send_event))
    prefill = 'Offer encouragement about the user\'s "next step".'
    asyncio.run(client.GPTSession._apply_prefill(None, session, prefill))
    item = events[0].message["other_data"]["item"]
    assert item["role"] == "system"
    assert item["content"] == [{"type": "input_text", "text": prefill}]
