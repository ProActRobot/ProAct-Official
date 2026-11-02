"""Context Encoder and Behavior Planner from paper Appendix F."""

import base64
import json
import logging
import os
import threading
import time
import io
import concurrent.futures
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Dict, Any

from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

LOGGER = logging.getLogger("slow_thinking")

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
CURRENT_SCENARIO_SLOW = os.getenv(
    "PROACT_SCENARIO", (PROMPT_DIR / "scenario.txt").read_text(encoding="utf-8").strip()
)


def get_memory_system_prompt(scenario: str) -> str:
    return (PROMPT_DIR / "context_encoder.txt").read_text(encoding="utf-8").replace(
        "{scenario_prompt}", scenario
    )


def get_decision_system_prompt(scenario: str) -> str:
    return (PROMPT_DIR / "behavior_planner.txt").read_text(encoding="utf-8").replace(
        "{scenario_prompt}", scenario
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def encode_image(image_path: str, quality_mode: str = "ultra_low") -> str:
    """Resize and encode image with configurable quality presets."""
    presets = {
        "ultra_low": {"size": (320, 180), "quality": 50},
        "low":       {"size": (480, 270), "quality": 70},
        "medium":    {"size": (768, 432), "quality": 85},
        "high":      {"size": (1280, 720), "quality": 95},
    }

    settings = presets.get(quality_mode, presets["ultra_low"])
    target_size = settings["size"]
    jpeg_quality = settings["quality"]

    try:
        with Image.open(image_path, "r") as img:
            target_aspect = 16 / 9
            width, height = img.size
            current_aspect = width / height if height else target_aspect
            if current_aspect > target_aspect:
                new_width = int(height * target_aspect)
                left = (width - new_width) // 2
                box = (left, 0, left + new_width, height)
                img = img.crop(box)
            elif current_aspect < target_aspect:
                new_height = int(width / target_aspect)
                top = (height - new_height) // 2
                box = (0, top, width, top + new_height)
                img = img.crop(box)
            img = img.resize(target_size, Image.LANCZOS)

            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality)

            encoded_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded_str
    except Exception as e:
        LOGGER.error(f"Error encoding image {image_path}: {e}")
        return ""


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class LocoAction(BaseModel):
    action: Literal["none", "turn_left", "turn_right", "move_forward", "move_backward"]
    parameter: float = Field(..., ge=0, description="Meters for move (0-1.5), degrees for turn (0-180)")

    @model_validator(mode="after")
    def check_magnitude(self):
        limit = 180 if self.action.startswith("turn_") else 1.5
        if self.parameter > limit:
            raise ValueError(f"{self.action} exceeds {limit}")
        if self.action == "none" and self.parameter != 0:
            raise ValueError("Idle action must have zero magnitude")
        return self


class MotionClipData(BaseModel):
    intent: str = Field(..., description="Gesture intent")


class TheoryOfMindState(BaseModel):
    user_belief: str = Field(..., max_length=800)
    user_desire: Literal["wants_help", "wants_company", "wants_space", "wants_info", "unclear"]
    user_intention: str = Field(..., max_length=800)
    knowledge_gap: Optional[str] = Field(None, max_length=200)


# Appendix F.1 memory fields; the API schema adds no instruction descriptions.
class MemorySystemResult(BaseModel):
    user_profile: str = Field(..., max_length=800)
    user_mental_state: str = Field(..., max_length=800)
    event_timeline: str = Field(..., max_length=1600)
    workspace_status: str = Field(..., max_length=800)
    unresolved_tasks: str = Field(..., max_length=800)
    robot_actions: str = Field(..., max_length=1600)

    # Compatibility views for the realtime display; these are not LLM output fields.
    @property
    def context_summary(self):
        return self.event_timeline

    @property
    def user_profile_notes(self):
        return self.user_profile

    @property
    def workspace_notes(self):
        return self.workspace_status

    @property
    def user_intention(self):
        return self.user_mental_state

    user_belief = property(lambda self: self.user_mental_state)
    user_desire = property(lambda self: "unclear")
    engagement_level = property(lambda self: "focused")
    conversation_phase = property(lambda self: "mid_conversation")
    empathy_requirement = property(lambda self: "none")
    agent_emotion = property(lambda self: "neutral")
    reasoning = property(lambda self: "")


class EmbodiedMotivationEval(BaseModel):
    """Runtime assessment with compatibility fields for the UI and server."""
    reasoning: str = Field(..., description="Analyze 5 dims (visual_scene, user_intent, conversation_state, social_protocol, emotional_response). Score each 1-5 in your reasoning.")

    # Explicit spatial and decision analysis
    tracking_target: str = Field(..., description="The specific object or person identified as the current focus of tracking (e.g., 'user', 'whiteboard', 'bottle')")
    target_position_explanation: str = Field(..., description="Explicitly describe the tracking_target's position in the visual frame (e.g. 'user is off-center to the left', 'whiteboard is centered but small')")
    locomotion_reasoning: str = Field(..., description="Explain the task-oriented locomotion, or why no movement is needed; never track or center automatically.")

    visual_scene: int = Field(..., ge=1, le=5)
    user_intent: int = Field(..., ge=1, le=5)
    conversation_state: int = Field(..., ge=1, le=5)
    social_protocol: int = Field(..., ge=1, le=5)
    emotional_response: int = Field(..., ge=1, le=5)
    motivation_score: int = Field(default=1, ge=1, le=5)
    intention_score: int = Field(default=1, ge=1, le=5, description="Compatibility metadata; no extra action threshold")
    proactive_level: Literal["L0_passive", "L1_reactive", "L2_proactive", "L3_interrupt"] = "L0_passive"
    proactive_trigger: str = Field(..., max_length=200)

    @model_validator(mode="after")
    def derive_motivation(self):
        self.motivation_score = max(self.visual_scene, self.user_intent,
                                    self.conversation_state, self.social_protocol,
                                    self.emotional_response)
        self.intention_score = self.motivation_score
        self.proactive_level = ("L3_interrupt" if self.motivation_score == 5 else
                                "L2_proactive" if self.motivation_score >= 4 else "L0_passive")
        return self


class DecisionSystemResult(BaseModel):
    """Internal decision representation; not used as the API response schema."""
    motivation_eval: EmbodiedMotivationEval

    # Core decisions (must be extracted)
    should_interrupt: bool = Field(default=False, description="True if motivation_score >= 5")
    motion_intent: str = Field(..., max_length=200, description="Gesture intent, e.g. 'wave greeting', 'supportive nod', 'idle'")
    locomotion: List[LocoAction] = Field(default_factory=list)
    dialogue_prefill: str = Field(default="", max_length=300, description="Text to prefill agent's response")
    emotion: Literal["neutral", "happy", "concerned", "curious", "supportive"] = "neutral"
    confidence: Literal["low", "medium", "high"] = "medium"
    insertion_point: Literal["before_agent_reply", "before_user_reply"] = "before_agent_reply"


    @model_validator(mode="after")
    def enforce_paper_thresholds(self):
        score = self.motivation_eval.motivation_score
        if score < 4:
            self.motion_intent = ""
            self.locomotion = []
            self.dialogue_prefill = ""
        if score < 5:
            self.should_interrupt = False
        return self


# API response types contain only the assessment and action channels in F.2.
# Legacy UI/server metadata is added locally, outside the model request.
class PaperMotivationAssessment(BaseModel):
    reasoning: str
    tracking_target: str
    locomotion_reasoning: str
    visual_scene: int = Field(..., ge=1, le=5)
    user_intent: int = Field(..., ge=1, le=5)
    conversation_state: int = Field(..., ge=1, le=5)
    social_protocol: int = Field(..., ge=1, le=5)
    emotional_response: int = Field(..., ge=1, le=5)


class PaperLocomotionAction(BaseModel):
    action: Literal["move", "turn"]
    direction: Literal["forward", "backward", "left", "right"]
    magnitude: float = Field(..., ge=0)
    tracking_target: str
    reasoning: str

    @model_validator(mode="after")
    def check_movement(self):
        directions = {"move": {"forward", "backward"}, "turn": {"left", "right"}}
        if self.direction not in directions[self.action]:
            raise ValueError("Direction does not match movement type")
        limit = 1.5 if self.action == "move" else 180
        if self.magnitude > limit:
            raise ValueError(f"{self.action} exceeds {limit}")
        return self


class PaperDecisionResponse(BaseModel):
    motivation_eval: PaperMotivationAssessment
    should_interrupt: bool
    motion_intent: str = Field(..., max_length=200)
    locomotion: List[PaperLocomotionAction]
    dialogue_prefill: str = Field(..., max_length=300)

    def to_runtime(self) -> DecisionSystemResult:
        assessment = self.motivation_eval.model_dump()
        assessment.update(target_position_explanation="", proactive_trigger="")
        actions = [
            LocoAction(action=f"{item.action}_{item.direction}", parameter=item.magnitude)
            for item in self.locomotion
        ]
        return DecisionSystemResult(
            motivation_eval=EmbodiedMotivationEval(**assessment),
            should_interrupt=self.should_interrupt,
            motion_intent=self.motion_intent,
            locomotion=actions,
            dialogue_prefill=self.dialogue_prefill,
        )


def paper_decision_data(result: DecisionSystemResult) -> Dict[str, Any]:
    """Keep compatibility metadata out of the next model input."""
    motivation = result.motivation_eval
    assessment = {
        key: getattr(motivation, key)
        for key in PaperMotivationAssessment.model_fields
    }
    assessment["motivation_score"] = motivation.motivation_score
    actions = []
    for item in result.locomotion:
        if item.action == "none":
            continue
        action, direction = item.action.split("_", 1)
        actions.append(dict(action=action, direction=direction, magnitude=item.parameter,
                            tracking_target=motivation.tracking_target,
                            reasoning=motivation.locomotion_reasoning))
    return dict(motivation_eval=assessment, should_interrupt=result.should_interrupt,
                motion_intent=result.motion_intent, locomotion=actions,
                dialogue_prefill=result.dialogue_prefill)


class DialogueInstruction(BaseModel):
    insertion_point: Literal["before_agent_reply", "before_user_reply"]
    prefill_text: str = Field(..., max_length=300)
    should_interrupt_user: bool = False


class SlowThinkingAnalysis(BaseModel):
    """Analysis summary for frontend display (Backward Compatibility)"""
    my_intention: str = ""
    engagement_level: str = "focused"
    social_distance: str = "personal"
    conversation_phase: str = "opening"
    atmosphere: str = "casual"
    empathy_requirement: str = "none"


class SlowThinkingResult(BaseModel):
    """Final result (Appendix F)"""
    memory_summary: Optional[str] = None
    user_mental_state: Optional[TheoryOfMindState] = None
    embodied_motivation: Optional[EmbodiedMotivationEval] = None
    analysis: Optional["SlowThinkingAnalysis"] = None  # BACKWARD COMPATIBILITY
    motivation_score: int = Field(default=3, ge=1, le=5)
    intention_score: int = Field(default=3, ge=1, le=5)
    proactive_level: Literal["L0_passive", "L1_reactive", "L2_proactive", "L3_interrupt"] = "L1_reactive"
    proactive_trigger: str = "none"
    clip_texts: MotionClipData
    locomotion: List[LocoAction]
    emotion: Literal["neutral", "happy", "concerned", "curious", "supportive"]
    confidence: Literal["low", "medium", "high"]
    should_interrupt_user: bool
    dialogue_instruction: DialogueInstruction
    run_id: int
    timings: Optional[Dict[str, float]] = None


# =============================================================================
# MEMORY BANK - Shared State Between Systems
# =============================================================================

@dataclass
class MemoryBank:
    """Shared memory between Memory System and Decision System."""

    compressed_context: str = ""
    paper_memory: Dict[str, str] = field(default_factory=dict)
    previous_decision: Dict[str, Any] = field(default_factory=dict)
    engagement_level: str = "focused"
    conversation_phase: str = "opening"
    atmosphere: str = "casual"
    empathy_requirement: str = "none"
    agent_emotion: str = "neutral"

    user_belief: str = ""
    user_desire: str = "unclear"
    user_intention: str = ""
    user_profile_notes: str = ""
    workspace_notes: str = ""

    motion_intent_history: List[str] = field(default_factory=lambda: ["idle", "idle", "idle"])
    locomotion_history: List[str] = field(default_factory=lambda: ["none", "none", "none"])
    last_tracking_target: str = "user"
    last_target_position: str = "unknown"
    last_decision_emotion: str = "neutral"

    memory_last_updated: float = 0.0
    decision_last_run: float = 0.0

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def get_for_decision(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "compressed_context": self.compressed_context,
                "engagement_level": self.engagement_level,
                "conversation_phase": self.conversation_phase,
                "atmosphere": self.atmosphere,
                "empathy_requirement": self.empathy_requirement,
                "agent_emotion": self.agent_emotion,
                "user_belief": self.user_belief,
                "user_desire": self.user_desire,
                "user_intention": self.user_intention,
                "motion_intent_history": self.motion_intent_history,
                "locomotion_history": self.locomotion_history,
                "last_tracking_target": self.last_tracking_target,
            }

    def to_decision_context_string(self) -> str:
        """Format memory for Decision System prompt - Appendix F (shorter)"""
        with self._lock:
            loco_str = " -> ".join([x if x != 'none' else '' for x in self.locomotion_history])
            motion_str = " -> ".join([x if x != 'idle' else '' for x in self.motion_intent_history])
            return (
                "## MEMORY BANK\n"
                f"{json.dumps(self.paper_memory, ensure_ascii=False)}\n"
                f"Previous Decision: {json.dumps(self.previous_decision, ensure_ascii=False)}\n"
                f"Applied Locomotion History (recent): {loco_str}\n"
                f"Applied Motion Intents History(recent): {motion_str}\n"
            )

    def update_from_memory_system(self, result: MemorySystemResult) -> None:
        with self._lock:
            self.compressed_context = result.context_summary
            self.paper_memory = result.model_dump()
            self.engagement_level = result.engagement_level
            self.conversation_phase = result.conversation_phase
            # atmosphere removed from simplified schema
            self.empathy_requirement = result.empathy_requirement
            self.agent_emotion = result.agent_emotion
            # Direct access (no longer nested in user_mental_state)
            self.user_belief = result.user_belief
            self.user_desire = result.user_desire
            self.user_intention = result.user_intention
            self.user_profile_notes = result.user_profile_notes
            self.workspace_notes = result.workspace_notes
            self.memory_last_updated = time.time()

    def update_from_decision_system(self, result: DecisionSystemResult) -> None:
        with self._lock:
            self.motion_intent_history.append(result.motion_intent)
            new_loco = str([f"{l.action}({l.parameter})" for l in result.locomotion]) if result.locomotion else "none"
            self.locomotion_history.append(new_loco)
            # Keep at least last 10 for safety
            if len(self.motion_intent_history) > 10:
                self.motion_intent_history = self.motion_intent_history[-10:]
            if len(self.locomotion_history) > 10:
                self.locomotion_history = self.locomotion_history[-10:]
            self.last_tracking_target = result.motivation_eval.tracking_target
            self.last_target_position = result.motivation_eval.target_position_explanation
            self.last_decision_emotion = result.emotion
            self.decision_last_run = time.time()
            self.previous_decision = paper_decision_data(result)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SlowThinkingConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    model: str = field(default_factory=lambda: os.getenv("PROACT_COGNITIVE_MODEL", "gpt-4.1-mini-2025-04-14"))
    temperature: float = 0.3
    min_transcript_chars: int = 10
    max_images_memory: int = 3
    max_images_decision: int = 1
    image_quality: Literal["ultra_low", "low", "medium", "high"] = "ultra_low"
    log_dir: Optional[str] = None


# =============================================================================
# DUAL-SYSTEM SLOW THINKING - Appendix F
# =============================================================================

class SlowThinkingSystem:
    """
    Dual-System Slow Thinking Architecture - Appendix F.

    Optimized for lower latency with streamlined prompts.
    """

    def __init__(
        self,
        context_provider: Callable[[], Optional[Tuple[str, Sequence[str], int]]],
        result_callback: Callable[[SlowThinkingResult], None],
        interval_seconds: float = 6.0,
        run_continuously: bool = True,
        config: Optional[SlowThinkingConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._context_provider = context_provider
        self._result_callback = result_callback
        self._interval = max(0.0, interval_seconds)
        self._run_continuously = run_continuously
        self._config = config or SlowThinkingConfig()
        self._logger = logger or LOGGER

        self.memory_bank = MemoryBank()
        self._recent_cycle_transcripts = deque(maxlen=5)
        self._last_transcript = ""

        self._client = OpenAI(
            api_key=self._config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self._config.base_url,
            timeout=8.0,
            max_retries=0,
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._run_counter = 0
        self._run_counter_lock = threading.Lock()

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="CognitiveSystem"
        )

        if self._config.log_dir:
            os.makedirs(self._config.log_dir, exist_ok=True)

    def start(self, daemon: bool = True) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=daemon,
            name="CognitiveLoop"
        )
        self._thread.start()
        self._logger.info("SlowThinkingSystem (Appendix F) started.")

    def stop(self, join_timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        self._wake_event.set()
        self._executor.shutdown(wait=False)
        if self._thread:
            self._thread.join(timeout=join_timeout)
        self._logger.info("SlowThinkingSystem (Appendix F) stopped.")

    def request_run_soon(self, reason: str = "context_update") -> None:
        if not self._stop_event.is_set():
            self._logger.debug("Wake requested: %s", reason)
            self._wake_event.set()

    def _next_run_id(self) -> int:
        with self._run_counter_lock:
            self._run_counter += 1
            return self._run_counter

    def _timed_call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - start

    def _truncate_transcript(self, transcript: str, max_blocks: int = 10) -> str:
        """Truncate transcript to the last N utterances/blocks."""
        clean_transcript = transcript.strip()
        if not clean_transcript or clean_transcript == "[No transcript yet]":
            return clean_transcript

        blocks = []
        current_block = []
        for line in clean_transcript.splitlines():
            # Robust detection for various turn formats
            is_turn_start = (
                line.startswith("User-turn_") or
                line.startswith("Agent-turn_") or
                line.startswith("## User") or
                line.startswith("## Assistant")
            )
            if is_turn_start:
                if current_block:
                    blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                # Ignore the generic "## History transcripts" header
                if line.startswith("## History transcripts") and not current_block:
                    continue
                current_block.append(line)
        if current_block:
            blocks.append("\n".join(current_block))

        if len(blocks) > max_blocks:
            return "(...history omitted...)\n\n" + "\n\n".join(blocks[-max_blocks:])
        return clean_transcript

    def _cycle_transcripts(self, transcript: str) -> Tuple[str, str]:
        """Encoder gets five cycles; planner gets only new transcript content."""
        current = transcript.strip()
        previous = self._last_transcript
        if current == previous:
            delta = "[No new dialogue since last cycle]"
        elif previous and current.startswith(previous):
            delta = current[len(previous):].strip()
        else:
            # Transcript providers may use a sliding window; retain its new suffix.
            overlap = 0
            for length in range(min(len(previous), len(current)), 0, -1):
                if previous[-length:] == current[:length]:
                    overlap = length
                    break
            delta = current[overlap:].strip()
        self._last_transcript = current
        self._recent_cycle_transcripts.append(delta)
        return "\n\n".join(self._recent_cycle_transcripts), delta

    def _suppress_repeated_actions(self, result: DecisionSystemResult, transcript: str) -> None:
        """Cross-check each channel separately, preserving unfinished components."""
        with self.memory_bank._lock:
            if result.motion_intent and result.motion_intent in self.memory_bank.motion_intent_history:
                result.motion_intent = "idle attentive"
            if result.dialogue_prefill and result.dialogue_prefill in transcript:
                result.dialogue_prefill = ""
                result.should_interrupt = False
            issued = self.memory_bank.locomotion_history
            result.locomotion = [
                action for action in result.locomotion
                if not any(f"{action.action}({action.parameter})" in item for item in issued)
            ]

    def run_once(self) -> Optional[SlowThinkingResult]:
        """Execute one cycle of both systems in parallel"""
        ctx = self._context_provider()
        if not ctx:
            return None

        if isinstance(ctx, tuple) and len(ctx) >= 3:
            transcript, image_paths, run_id_from_ctx = ctx[0], ctx[1], ctx[2]
        else:
            transcript, image_paths = ctx[:2]
            run_id_from_ctx = None

        if not image_paths or len(image_paths) == 0:
            return None

        # Truncate transcript to save tokens in both systems and logs
        transcript = self._truncate_transcript(transcript, max_blocks=10)
        memory_transcript, delta_transcript = self._cycle_transcripts(transcript)

        run_id = run_id_from_ctx if run_id_from_ctx is not None else self._next_run_id()
        overall_start = time.perf_counter()
        all_images = list(image_paths)

        memory_context = self.memory_bank.to_decision_context_string()

        try:
            # PARALLEL EXECUTION with timeout to avoid blocking
            future_memory = self._executor.submit(
                self._timed_call,
                self._call_memory_system,
                memory_transcript,
                all_images,
                memory_context,  # Pass previous context for incremental updates
            )

            future_decision = self._executor.submit(
                self._timed_call,
                self._call_decision_system,
                delta_transcript,
                memory_context,
                all_images[-self._config.max_images_decision:] if all_images else [],
            )

            # Wait for both with timeout to avoid blocking the main loop
            timeout_val = 8.0
            done, not_done = concurrent.futures.wait([future_memory, future_decision], timeout=timeout_val)

            for pending in not_done:
                pending.cancel()

            memory_result, memory_duration = (None, 0.0)
            decision_result, decision_duration = (None, 0.0)

            if future_memory in done:
                memory_result, memory_duration = future_memory.result()
            else:
                self._logger.warning(f"Memory System timed out ({timeout_val}s limit reached), discarding result.")

            if future_decision in done:
                decision_result, decision_duration = future_decision.result()
            else:
                self._logger.warning(f"Decision System timed out ({timeout_val}s limit reached), discarding result.")

            # Record timings and update bank only if not timed out
            timings = {
                "scene2text_s": 0.0,  # Reserved display timing
                "scene2memory_s": memory_duration,
                "scene2proact_s": decision_duration,
                "total_s": time.perf_counter() - overall_start,
            }

            if memory_result:
                self.memory_bank.update_from_memory_system(memory_result)
            if decision_result:
                self._suppress_repeated_actions(decision_result, transcript)
                self.memory_bank.update_from_decision_system(decision_result)

            if decision_result:
                dialogue_instruction = DialogueInstruction(
                    insertion_point=decision_result.insertion_point,
                    prefill_text=decision_result.dialogue_prefill,
                    should_interrupt_user=decision_result.should_interrupt,
                )

                final_result = SlowThinkingResult(
                    memory_summary=memory_result.context_summary if memory_result else None,
                    user_mental_state=TheoryOfMindState(
                        user_belief=memory_result.user_belief,
                        user_desire=memory_result.user_desire,
                        user_intention=memory_result.user_intention,
                        knowledge_gap=None,
                    ) if memory_result else None,
                    embodied_motivation=decision_result.motivation_eval,
                    analysis=SlowThinkingAnalysis(
                        my_intention=decision_result.motion_intent,
                        engagement_level=memory_result.engagement_level if memory_result else "focused",
                        social_distance="personal",
                        conversation_phase=memory_result.conversation_phase if memory_result else "opening",
                        atmosphere="casual",  # Simplified: no longer in schema
                        empathy_requirement=memory_result.empathy_requirement if memory_result else "none",
                    ) if memory_result else None,
                    motivation_score=decision_result.motivation_eval.motivation_score,
                    intention_score=decision_result.motivation_eval.intention_score,
                    proactive_level=decision_result.motivation_eval.proactive_level,
                    proactive_trigger=decision_result.motivation_eval.proactive_trigger,
                    clip_texts=MotionClipData(intent=decision_result.motion_intent),
                    locomotion=decision_result.locomotion,
                    emotion=decision_result.emotion,
                    confidence=decision_result.confidence,
                    should_interrupt_user=decision_result.should_interrupt,
                    dialogue_instruction=dialogue_instruction,
                    run_id=run_id,
                    timings=timings,
                )

                if self._config.log_dir:
                    self._write_log(
                        transcript,
                        all_images,
                        final_result,
                        memory_result,
                        memory_context=memory_context,
                        memory_duration=memory_duration,
                        decision_duration=decision_duration,
                    )

                self._result_callback(final_result)
                return final_result

        except Exception as exc:
            self._logger.exception("Dual-system execution failed: %s", exc)
            return None

        return None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            self.run_once()

            if not self._run_continuously:
                continue

            elapsed = time.time() - start
            self._logger.info(f"Cognitive cycle: {elapsed:.2f}s")

            sleep_for = max(0.5, self._interval - elapsed)
            woke_early = self._wake_event.wait(timeout=sleep_for)

            if self._stop_event.is_set():
                break
            if woke_early:
                self._wake_event.clear()

    # -------------------------------------------------------------------------
    # MEMORY SYSTEM API CALL
    # -------------------------------------------------------------------------
    def _call_memory_system(self, transcript: str, image_paths: List[str], previous_context: str = "") -> Optional[MemorySystemResult]:
        try:
            segments = self._build_memory_segments(transcript, image_paths, previous_context)

            response = self._client.beta.chat.completions.parse(
                model=self._config.model,
                temperature=self._config.temperature,
                messages=[
                    {"role": "system", "content": get_memory_system_prompt(CURRENT_SCENARIO_SLOW)},
                    {"role": "user", "content": segments},
                ],
                response_format=MemorySystemResult,
            )
            return response.choices[0].message.parsed
        except Exception as exc:
            self._logger.error("Memory System call failed: %s", exc)
            return None

    def _build_memory_segments(self, transcript: str, image_paths: List[str], previous_context: str = "") -> List[dict]:
        """Build input segments for Memory System - with previous context for incremental updates"""
        segments = []

        # Transcript is already truncated in run_once()
        clean_transcript = transcript.strip() or "[No transcript yet]"

        # Include previous memory context if available
        context_block = ""
        if previous_context:
            context_block = f"## PREVIOUS MEMORY (t-1)\n{previous_context}\n\n"

        segments.append({
            "type": "text",
            "text": (
                f"{context_block}"
                f"## DIALOGUE TRANSCRIPTS\n{clean_transcript}"
            ),
        })

        recent_images = image_paths[-self._config.max_images_memory:] if image_paths else []
        for path in recent_images:
            image_b64 = encode_image(path, self._config.image_quality)
            if image_b64:
                segments.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low",
                    },
                })

        return segments

    # -------------------------------------------------------------------------
    # DECISION SYSTEM API CALL
    # -------------------------------------------------------------------------
    def _call_decision_system(
        self,
        transcript: str,
        memory_context: str,
        image_paths: List[str]
    ) -> Optional[DecisionSystemResult]:
        try:
            segments = self._build_decision_segments(transcript, memory_context, image_paths)
            system_prompt = get_decision_system_prompt(CURRENT_SCENARIO_SLOW)

            response = self._client.beta.chat.completions.parse(
                model=self._config.model,
                temperature=self._config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": segments},
                ],
                response_format=PaperDecisionResponse,
            )
            parsed = response.choices[0].message.parsed
            return parsed.to_runtime() if parsed is not None else None
        except Exception as exc:
            self._logger.error("Decision System call failed: %s", exc)
            return None

    def _build_decision_segments(
        self,
        transcript: str,
        memory_context: str,
        image_paths: List[str]
    ) -> List[dict]:
        """Build input segments for Decision System - Appendix F"""
        segments = []

        clean_transcript = transcript.strip() or "[No new transcript]"

        segments.append({
            "type": "text",
            "text": (
                f"{memory_context}\n"
                f"## RECENT DIALOGUE TRANSCRIPT\n{clean_transcript}"
            ),
        })

        for path in image_paths:
            image_b64 = encode_image(path, self._config.image_quality)
            if image_b64:
                segments.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low",
                    },
                })

        return segments

    # -------------------------------------------------------------------------
    # LOGGING - Separate Memory and Decision logs
    # -------------------------------------------------------------------------
    def _write_memory_log(
        self,
        run_id: int,
        transcript: str,
        image_paths: List[str],
        memory_context: str,
        memory_result: Optional[MemorySystemResult],
        duration_s: float,
    ) -> None:
        """Write detailed Memory System input/output to separate file."""
        if not self._config.log_dir:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            self._config.log_dir,
            f"memory_{timestamp}_id{run_id}.json"
        )

        payload = {
            "run_id": run_id,
            "system": "memory",
            "timestamp": timestamp,
            "duration_s": duration_s,
            "input": {
                "transcript": transcript,
                "images_count": len(image_paths),
                "image_paths": image_paths[-self._config.max_images_memory:] if image_paths else [],
                "previous_memory_context": memory_context,
            },
            "output": None if not memory_result else {
                **memory_result.model_dump(),
                "reasoning": memory_result.reasoning,
                "context_summary": memory_result.context_summary,
                "user_belief": memory_result.user_belief,
                "user_desire": memory_result.user_desire,
                "user_intention": memory_result.user_intention,
                "engagement_level": memory_result.engagement_level,
                "conversation_phase": memory_result.conversation_phase,
                "empathy_requirement": memory_result.empathy_requirement,
                "agent_emotion": memory_result.agent_emotion,
                "user_profile_notes": memory_result.user_profile_notes,
                "workspace_notes": memory_result.workspace_notes,
            },
        }

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            self._logger.warning("Failed to write memory log: %s", exc)

    def _write_decision_log(
        self,
        run_id: int,
        transcript: str,
        memory_context: str,
        image_paths: List[str],
        decision_result: Optional[DecisionSystemResult],
        duration_s: float,
    ) -> None:
        """Write detailed Decision System input/output to separate file."""
        if not self._config.log_dir:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            self._config.log_dir,
            f"decision_{timestamp}_id{run_id}.json"
        )

        payload = {
            "run_id": run_id,
            "system": "decision",
            "timestamp": timestamp,
            "duration_s": duration_s,
            "input": {
                "transcript": transcript,
                "memory_context": memory_context,
                "images_count": len(image_paths),
                "image_paths": image_paths,
            },
            "output": None if not decision_result else {
                "motivation_eval": {
                    "reasoning": decision_result.motivation_eval.reasoning,  # CoT with 5-dim scores
                    "tracking_target": decision_result.motivation_eval.tracking_target,
                    "target_position": decision_result.motivation_eval.target_position_explanation,
                    "locomotion_reason": decision_result.motivation_eval.locomotion_reasoning,
                    "motivation_score": decision_result.motivation_eval.motivation_score,
                    "intention_score": decision_result.motivation_eval.intention_score,
                    "proactive_level": decision_result.motivation_eval.proactive_level,
                    "proactive_trigger": decision_result.motivation_eval.proactive_trigger,
                },
                "should_interrupt": decision_result.should_interrupt,
                "motion_intent": decision_result.motion_intent,
                "locomotion": [l.model_dump() for l in decision_result.locomotion],
                "dialogue_prefill": decision_result.dialogue_prefill,
                "insertion_point": decision_result.insertion_point,
                "confidence": decision_result.confidence,
                "emotion": decision_result.emotion,
            },
        }

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            self._logger.warning("Failed to write decision log: %s", exc)

    def _write_log(
        self,
        transcript: str,
        image_paths: List[str],
        result: SlowThinkingResult,
        memory_result: Optional[MemorySystemResult],
        memory_context: str = "",
        memory_duration: float = 0.0,
        decision_duration: float = 0.0,
    ) -> None:
        """Write combined summary log and separate detailed logs."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Write separate detailed logs
        self._write_memory_log(
            run_id=result.run_id,
            transcript=transcript,
            image_paths=image_paths,
            memory_context=memory_context,
            memory_result=memory_result,
            duration_s=memory_duration,
        )

        self._write_decision_log(
            run_id=result.run_id,
            transcript=transcript,
            memory_context=memory_context,
            image_paths=image_paths[-self._config.max_images_decision:] if image_paths else [],
            decision_result=result.embodied_motivation and DecisionSystemResult(
                motivation_eval=result.embodied_motivation,
                should_interrupt=result.should_interrupt_user,
                motion_intent=result.clip_texts.intent,
                locomotion=result.locomotion,
                dialogue_prefill=result.dialogue_instruction.prefill_text,
                insertion_point=result.dialogue_instruction.insertion_point,
                confidence=result.confidence,
                emotion=result.emotion,
            ) if result.embodied_motivation else None,
            duration_s=decision_duration,
        )

        # Write combined summary log
        log_file = os.path.join(
            self._config.log_dir,
            f"summary_{timestamp}_id{result.run_id}.json"
        )

        payload = {
            "run_id": result.run_id,
            "version": "appendix-f",
            "timestamp": timestamp,
            "transcript": transcript,
            "images_count": len(image_paths),
            "timings": result.timings,
            "memory": {
                "reasoning": memory_result.reasoning if memory_result else None,
                "context_summary": memory_result.context_summary if memory_result else None,
                "user_belief": memory_result.user_belief if memory_result else None,
                "user_desire": memory_result.user_desire if memory_result else None,
                "user_intention": memory_result.user_intention if memory_result else None,
                "engagement_level": memory_result.engagement_level if memory_result else None,
                "conversation_phase": memory_result.conversation_phase if memory_result else None,
                "empathy_requirement": memory_result.empathy_requirement if memory_result else None,
            },
            "decision": {
                "should_interrupt": result.should_interrupt_user,
                "motion_intent": result.clip_texts.intent,
                "locomotion": [l.model_dump() for l in result.locomotion],
                "emotion": result.emotion,
                "confidence": result.confidence,
                "motivation_score": result.motivation_score,
                "intention_score": result.intention_score,
                "proactive_level": result.proactive_level,
                "proactive_trigger": result.proactive_trigger,
            },
        }

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            self._logger.warning("Failed to write summary log: %s", exc)
