"""Regression checks for the canonical ProAct runtime transport."""

import base64
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_protocol_smoke as protocol
from benchmark import load, transcripts


class FakeSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])

    async def recv(self):
        if not self.messages:
            raise AssertionError("The canonical server does not emit WebSocket turn acknowledgements")
        return json.dumps(self.messages.pop(0))


class CanonicalProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_turn_acknowledgements_and_single_session_cleanup(self):
        dataset = load()
        dataset["sessions"] = dataset["sessions"][:2]
        sockets = {}
        requests = []

        class Connection:
            def __init__(self, address):
                self.uid = parse_qs(urlsplit(address).query)["uid"][0]
                self.socket = FakeSocket()

            async def __aenter__(self):
                sockets[self.uid] = self.socket
                return self.socket

            async def __aexit__(self, *arguments):
                sockets.pop(self.uid)

        def connect(address, *, proxy=None, **options):
            self.assertIsNone(proxy)
            return Connection(address)

        def request(server, route, uid=None, body=None, method="GET"):
            requests.append((route, method, uid, body))
            if route == "/api/v1/health":
                return {"status": "OK", "interruption_support": True}
            if route == "/api/v1/set_zero_time":
                self.assertEqual(method, "POST")
                return {"status": "ok", "zero_point": 1}
            if route == "/api/v1/stream":
                content = body["messages"][0]["content"]
                turn = content[-1]["turn_complete"]["turn"]
                self.assertEqual(content[0]["input_audio"]["turn"], turn)
                return {"id": uid, "choices": {
                    "content": "success", "finish_reason": f"turn_complete:{turn}",
                }}
            if route == "/api/v1/interrupt":
                self.assertEqual(body["type"], "manual")
                sockets[uid].messages.append({
                    "type": "interruption",
                    "data": {"id": uid, "interruption": {"sequence": body["sequence"]}},
                })
                return {"id": uid, "status": "ok", "sequence": body["sequence"]}
            raise AssertionError(f"Unsupported canonical endpoint: {method} {route}")

        with mock.patch.dict(sys.modules, {"websockets": SimpleNamespace(connect=connect)}):
            with mock.patch.object(protocol, "request", side_effect=request):
                report = await protocol.run("http://127.0.0.1:32558", dataset, 0.5)
        self.assertEqual(len(report["sessions"]), 2)
        self.assertTrue(all(session["status"] == "passed" for session in report["sessions"]))
        expected_turns = sum(len(session["transcript"]) for session in transcripts(dataset))
        self.assertEqual(sum(session["turn_acknowledgements"] for session in report["sessions"]), expected_turns)
        self.assertEqual(sum(route == "/api/v1/set_zero_time" for route, *_ in requests), 4)
        self.assertFalse(any(method == "DELETE" for _, method, *_ in requests))
        self.assertEqual(report["turn_acknowledgement"], "http_finish_reason")

    async def test_nested_interruption_acknowledgement_drains_motion(self):
        connection = FakeSocket([
            {"type": "motion", "data": {
                "id": "fixture",
                "motion_b64": base64.b64encode(b"\x93NUMPYfixture").decode(),
            }},
            {"type": "interruption", "data": {
                "id": "fixture", "interruption": {"sequence": 3},
            }},
        ])
        self.assertEqual(await protocol.consume_until(connection, "interruption", "fixture", sequence=3), 1)

    async def test_wrong_nested_interruption_sequence_is_rejected(self):
        connection = FakeSocket([{"type": "interruption", "data": {
            "id": "fixture", "interruption": {"sequence": 2},
        }}])
        with self.assertRaisesRegex(ValueError, "Interruption acknowledgement"):
            await protocol.consume_until(connection, "interruption", "fixture", sequence=3)

    async def test_wrong_session_is_rejected(self):
        connection = FakeSocket([{"type": "interruption", "data": {
            "id": "different-session", "interruption": {"sequence": 1},
        }}])
        with self.assertRaisesRegex(ValueError, "session ID mismatch"):
            await protocol.consume_until(connection, "interruption", "fixture", sequence=1)


class RuntimeLauncherTests(unittest.TestCase):
    def test_requires_canonical_motion_server(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "src/server/motion_server.py"):
                with protocol.local_server(directory, sys.executable):
                    self.fail("A missing runtime must not start")

    def test_starts_canonical_entrypoint_on_loopback_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/server").mkdir(parents=True)
            entrypoint = root / "src/server/motion_server.py"
            entrypoint.write_text("")
            process = mock.Mock()
            process.poll.return_value = None
            with mock.patch.object(protocol.subprocess, "Popen", return_value=process) as launch:
                with mock.patch.object(protocol, "request", return_value={"status": "OK"}):
                    with protocol.local_server(root, sys.executable) as origin:
                        self.assertTrue(origin.startswith("http://127.0.0.1:"))
            command = launch.call_args.args[0]
            self.assertEqual(command[:4], [sys.executable, str(entrypoint.resolve()), "--host", "127.0.0.1"])
            self.assertNotIn("--enable_motion", command)
            process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()

