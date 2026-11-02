"""Offline regression checks for Appendix F policy and runtime adaptation."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from client.cognition import (
    DecisionSystemResult, EmbodiedMotivationEval, LocoAction, MemoryBank,
    MemorySystemResult, PaperDecisionResponse, SlowThinkingConfig, SlowThinkingSystem,
    get_decision_system_prompt, get_memory_system_prompt,
)


def decision(score=4, **overrides):
    motivation = EmbodiedMotivationEval(
        reasoning="History check: no prior matching action.",
        tracking_target="poster", target_position_explanation="On the left.",
        locomotion_reasoning="Orient toward the requested poster.",
        visual_scene=score, user_intent=1, conversation_state=1,
        social_protocol=1, emotional_response=1, motivation_score=1,
        intention_score=1, proactive_trigger="User requests poster orientation.",
    )
    payload = dict(motivation_eval=motivation, should_interrupt=True,
                   motion_intent="Raise right hand and wave.",
                   locomotion=[LocoAction(action="turn_left", parameter=45)],
                   dialogue_prefill="The poster is on the left.")
    payload.update(overrides)
    return DecisionSystemResult(**payload)


def memory():
    return MemorySystemResult(
        user_profile="The user wears glasses and carries a bag.",
        user_mental_state="The user wants to learn about the poster.",
        event_timeline="The user entered and asked about the poster.",
        workspace_status="The poster is on the left and the bag is on a chair.",
        unresolved_tasks="Explain the poster and remind the user about the bag.",
        robot_actions="The robot greeted the user.",
    )


@pytest.mark.parametrize("score", [1, 2, 3])
def test_below_threshold_empties_all_channels(score):
    result = decision(score)
    assert result.motivation_eval.motivation_score == score
    assert not result.should_interrupt
    assert result.motion_intent == result.dialogue_prefill == ""
    assert result.locomotion == []


def test_score_four_acts_without_interrupting():
    result = decision(4)
    assert result.motion_intent and result.dialogue_prefill and result.locomotion
    assert not result.should_interrupt
    assert result.motivation_eval.motivation_score == 4


def test_urgent_score_allows_but_does_not_force_interruption():
    assert decision(5).should_interrupt
    assert not decision(5, should_interrupt=False).should_interrupt


@pytest.mark.parametrize("action,parameter", [
    ("move_forward", 1.6), ("turn_left", 181), ("turn_right", -1), ("none", 1),
])
def test_out_of_range_locomotion_rejected(action, parameter):
    with pytest.raises(ValidationError):
        LocoAction(action=action, parameter=parameter)


def test_memory_bank_preserves_six_fields_and_prior_decision():
    bank = MemoryBank()
    bank.update_from_memory_system(memory())
    bank.update_from_decision_system(decision())
    context = bank.to_decision_context_string()
    for field, value in memory().model_dump().items():
        assert field in context and value in context
    assert '"motivation_score": 4' in context
    assert "turn_left(45.0)" in context
    assert set(MemorySystemResult.model_fields) == {
        "user_profile", "user_mental_state", "event_timeline", "workspace_status",
        "unresolved_tasks", "robot_actions",
    }


def test_paper_prompts_are_loaded_and_scenario_is_injected():
    encoder = get_memory_system_prompt("SCENARIO_SENTINEL")
    planner = get_decision_system_prompt("SCENARIO_SENTINEL")
    assert "SCENARIO_SENTINEL" in encoder and "SCENARIO_SENTINEL" in planner
    assert "{scenario_prompt}" not in encoder + planner
    assert "EXACTLY one sentence" in encoder
    assert "If ANY\ndimension >= 4" in planner
    assert "ONLY if motivation_score >= 5" in planner
    assert "Milestone 1" not in encoder + planner
    # Both output schemas can be serialized for strict structured responses.
    json.dumps(MemorySystemResult.model_json_schema())
    json.dumps(PaperDecisionResponse.model_json_schema())


def test_parallel_cycle_adapts_paper_output_without_network(tmp_path):
    output = []
    system = SlowThinkingSystem(
        context_provider=lambda: ("User-turn_1: Explain the poster.", ["frame.jpg"], 7),
        result_callback=output.append,
        config=SlowThinkingConfig(api_key="offline-test", log_dir=str(tmp_path)),
    )
    system._call_memory_system = lambda *args: memory()
    system._call_decision_system = lambda *args: decision(4)
    try:
        result = system.run_once()
        assert result is output[0]
        assert result.run_id == 7
        assert result.memory_summary == memory().event_timeline
        assert result.clip_texts.intent == "Raise right hand and wave."
        assert not result.dialogue_instruction.should_interrupt_user
        assert len(list(tmp_path.glob("*.json"))) == 3
    finally:
        system.stop(join_timeout=1)


def test_memory_receives_three_images_and_decision_receives_latest(monkeypatch):
    import client.cognition as cognition
    monkeypatch.setattr(cognition, "encode_image", lambda path, quality: path)
    system = SlowThinkingSystem(lambda: None, lambda result: None,
                                config=SlowThinkingConfig(api_key="offline-test"))
    try:
        frames = ["1", "2", "3", "4"]
        mem = system._build_memory_segments("hello", frames, "previous")
        assert [x["image_url"]["url"] for x in mem[1:]] == [
            "data:image/jpeg;base64,2", "data:image/jpeg;base64,3", "data:image/jpeg;base64,4",
        ]
        assert "previous" in mem[0]["text"]
    finally:
        system.stop(join_timeout=1)


def test_repetition_suppresses_channels_independently():
    system = SlowThinkingSystem(lambda: None, lambda result: None,
                                config=SlowThinkingConfig(api_key="offline-test"))
    try:
        system.memory_bank.update_from_decision_system(decision(4))
        repeat = decision(5)
        system._suppress_repeated_actions(repeat, "Agent: The poster is on the left.")
        assert repeat.motion_intent == "idle attentive"
        assert repeat.dialogue_prefill == ""
        assert repeat.locomotion == []
        assert not repeat.should_interrupt
        partial = decision(5, motion_intent="Nod in agreement.")
        system._suppress_repeated_actions(partial, "")
        assert partial.motion_intent == "Nod in agreement."
        assert partial.dialogue_prefill
        assert partial.locomotion == []
    finally:
        system.stop(join_timeout=1)


def test_encoder_window_and_planner_delta():
    system = SlowThinkingSystem(lambda: None, lambda result: None,
                                config=SlowThinkingConfig(api_key="offline-test"))
    try:
        assert system._cycle_transcripts("hello") == ("hello", "hello")
        memory_text, delta = system._cycle_transcripts("hello world")
        assert delta == "world" and memory_text == "hello\n\nworld"
        _, delta = system._cycle_transcripts("hello world")
        assert delta == "[No new dialogue since last cycle]"
        for index in range(10):
            system._cycle_transcripts(f"cycle {index}")
        assert len(system._recent_cycle_transcripts) == 5
    finally:
        system.stop(join_timeout=1)
