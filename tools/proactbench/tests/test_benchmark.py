"""Run with python -m unittest discover -s tests -v from the benchmark root."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark import DIMENSIONS, load, transcripts, validate
from evaluate_predictions import CHANNELS, evaluate
from run_protocol_smoke import silence, websocket_url
import base64
import io
import wave


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load()

    def test_release_counts(self):
        self.assertEqual(
            validate(self.data),
            {
                "sessions": 40,
                "positive": 30,
                "negative": 10,
                "events": 313,
                "stage_directions": 141,
                "spoken_turns": 152,
            },
        )

    def test_transcript_projection_matches_reviewed_export(self):
        reference = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "data/proactbench_40_transcripts.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(transcripts(self.data), reference)

    def test_inline_stage_order(self):
        events = self.data["sessions"][3]["events"]
        turn = [e for e in events if e["turn_id"] == "SES-04-T02"]
        self.assertEqual(
            [e["type"] for e in turn],
            ["speech", "stage_direction", "speech", "stage_direction"],
        )
        self.assertEqual(turn[1]["text"], "(放桌上)")

    def test_annotation_continuations_preserved(self):
        signal = self.data["sessions"][29]["trigger_annotation"]["signal"]
        self.assertIn("\n", signal)
        self.assertIn("语气平稳", signal)

    def test_negative_trap_and_unresolved_pair(self):
        item = self.data["sessions"][30]["trigger_annotation"]
        self.assertFalse(item["should_trigger"])
        self.assertTrue(item["trap"]["correct_behavior"])
        self.assertEqual(item["paired_session_ids"], [])
        self.assertIn("Storytelling", item["unresolved_pair_reference"])

    def test_compressed_pair_ids_expanded(self):
        self.assertEqual(
            self.data["sessions"][32]["trigger_annotation"]["paired_session_ids"],
            ["SES-10", "SES-17"],
        )

    def test_mn_is_positive_design_not_prediction(self):
        item = self.data["sessions"][29]
        self.assertEqual(item["design_label"], "MN")
        self.assertTrue(item["trigger_annotation"]["should_trigger"])

    def test_reject_contradictory_ground_truth(self):
        data = copy.deepcopy(self.data)
        data["sessions"][0]["trigger_annotation"]["should_trigger"] = False
        with self.assertRaisesRegex(ValueError, "Inconsistent label"):
            validate(data)

    def test_reject_duplicate_events(self):
        data = copy.deepcopy(self.data)
        data["sessions"][0]["events"][1]["event_id"] = data["sessions"][0]["events"][0][
            "event_id"
        ]
        with self.assertRaisesRegex(ValueError, "Invalid event IDs"):
            validate(data)

    def test_reject_directions_leaked_into_speech(self):
        data = copy.deepcopy(self.data)
        next(e for e in data["sessions"][0]["events"] if e["type"] == "speech")[
            "text"
        ] += " (gesture)"
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate(data)

    def test_audio_and_ws_contract(self):
        with wave.open(io.BytesIO(base64.b64decode(silence(0.5)))) as wav:
            self.assertEqual(
                (wav.getframerate(), wav.getnframes(), wav.getnchannels()),
                (15360, 7680, 1),
            )
        self.assertEqual(
            websocket_url("https://example.test", "id"),
            "wss://example.test/ws/motion?uid=id",
        )
        with self.assertRaises(ValueError):
            websocket_url("https://example.test/?token=secret", "id")


def observed(sid, score, action="", trial=1):
    # Synthetic unit-test data, not measured benchmark output.
    return {
        "session_id": sid,
        "trial": trial,
        "cycles": [
            {
                "scores": {key: score for key in DIMENSIONS},
                "actions": {
                    key: action if key == "GestureIntention" else "" for key in CHANNELS
                },
            }
        ],
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.data = load()

    def predict(self, runs):
        return {
            "kind": "cognitive_predictions",
            "schema_version": "1.0.0",
            "runs": runs,
        }

    def test_decision_requires_score_and_action(self):
        result = evaluate(
            self.data,
            self.predict([observed("SES-01", 4), observed("SES-26", 4, "wave")]),
            True,
        )
        self.assertEqual(result["counts"], {"TP": 0, "FP": 1, "FN": 1, "TN": 0})
        self.assertFalse(result["complete_coverage"])

    def test_any_cycle_and_dialogue_at_score_four(self):
        run = observed("SES-01", 2)
        run["cycles"].extend(observed("SES-01", 4, "point")["cycles"])
        run["cycles"][1]["actions"]["DialogueContext"] = "hello"
        result = evaluate(self.data, self.predict([run]), True)
        self.assertEqual(result["counts"]["TP"], 1)
        self.assertEqual(result["policy_violations"], [])

    def test_only_urgent_interruption_requires_score_five(self):
        for score, expected in ((4, 1), (5, 0)):
            run = observed("SES-01", score)
            run["cycles"][0]["actions"]["DialogueContext"] = "A relevant reminder."
            run["cycles"][0]["should_interrupt"] = True
            result = evaluate(self.data, self.predict([run]), True)
            self.assertEqual(len(result["policy_violations"]), expected)
            if expected:
                self.assertEqual(result["policy_violations"][0]["policy"], "interruption_requires_score_5")

    def test_reject_invalid_interruption_flag(self):
        run = observed("SES-01", 5)
        run["cycles"][0]["should_interrupt"] = "false"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            evaluate(self.data, self.predict([run]), True)

    def test_complete_three_trial_coverage(self):
        runs = [
            observed(s["session_id"], 1, trial=t)
            for s in self.data["sessions"]
            for t in (1, 2, 3)
        ]
        result = evaluate(self.data, self.predict(runs))
        self.assertTrue(result["complete_coverage"])
        self.assertEqual(result["counts"], {"TP": 0, "FP": 0, "FN": 90, "TN": 30})
        self.assertIsNone(result["precision"])

    def test_reject_partial_by_default(self):
        with self.assertRaisesRegex(ValueError, "Incomplete coverage"):
            evaluate(self.data, self.predict([observed("SES-01", 4, "wave")]))

    def test_reject_duplicate_runs(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(self.data, self.predict([observed("SES-01", 1)] * 2), True)

    def test_reject_empty_or_invalid_scores(self):
        for scores in (
            {},
            {key: True for key in DIMENSIONS},
            {key: 6 for key in DIMENSIONS},
        ):
            run = observed("SES-01", 1)
            run["cycles"][0]["scores"] = scores
            with self.assertRaisesRegex(ValueError, "Five integer scores"):
                evaluate(self.data, self.predict([run]), True)

    def test_reject_no_observed_cycles(self):
        run = observed("SES-01", 1)
        run["cycles"] = []
        with self.assertRaisesRegex(ValueError, "observed cognitive cycle"):
            evaluate(self.data, self.predict([run]), True)

    def test_reject_transport_as_predictions(self):
        with self.assertRaisesRegex(ValueError, "protocol smoke"):
            evaluate(self.data, {"kind": "protocol_smoke"})


if __name__ == "__main__":
    unittest.main()
