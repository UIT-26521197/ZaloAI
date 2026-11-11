import json
from pathlib import Path
import sys
import tempfile
from time import monotonic
import unittest
from unittest.mock import patch
import wave

import inference


TEST_ROOT = Path(__file__).resolve().parents[1]


class FakeToolEnv:
    def __init__(self, split, sample_id, root):
        assert (Path(root) / split / f"{split}.jsonl").is_file()
        self.sample_id = sample_id
        self.calls = []
        self.on = False

    def validate(self, name, arguments):
        valid = (name, arguments) in (
            ("home.list_lights", {}),
            ("home.turn_on", {"id": "light_42"}),
        )
        return {"ok": valid, "error": "" if valid else "Expected the device ID"}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        if not self.validate(name, arguments)["ok"]:
            return {"ok": False}
        if name == "home.list_lights":
            return {"ok": True, "devices": [{"name": "đèn bếp", "id": "light_42"}]}
        self.on = True
        return {"ok": True}

    def get_tool_calls(self):
        return self.calls

    def get_state_hash(self):
        return "fixture-on" if self.on else "fixture-off"


class FakeModels:
    def __init__(self, *args):
        self.actions = iter([
            '{"name":"other.write","arguments":{}}',
            '{"name":"home.turn_on","arguments":{}}',
            '{"name":"home.list_lights","arguments":{}}',
            '{"name":"home.turn_on","arguments":{"id":"light_42"}}',
            '{"done":true}',
        ])

    def transcribe(self, audio_path, deadline):
        assert audio_path.name == "train_0001.wav"
        assert inference.remaining_seconds(deadline) > 0
        return "Bật đèn bếp"

    def next_action(self, messages, deadline):
        action = next(self.actions)
        if "light_42" in action:
            assert "light_42" in messages[-1]["content"]
        return action


class InferenceTests(unittest.TestCase):
    def test_replay_resumes_from_last_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT, prefix=".lifemate-test-") as temporary:
            root = Path(temporary)
            split = root / "train"
            split.mkdir()
            calls = [{"name": "home.turn_on", "arguments": {"id": "light_42"}}]
            rows = [{"id": f"sample_{index}", "label": {"tool_calls": calls}}
                    for index in (1, 2)]
            (split / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "predictions.jsonl"
            cli = ["inference.py", "--data-root", str(root), "--replay-gold",
                   "--checkpoint-every", "1", "--output", str(output)]

            class FailSecond(FakeToolEnv):
                def call(self, name, arguments):
                    if self.sample_id == "sample_2":
                        raise RuntimeError("interrupted")
                    return super().call(name, arguments)

            with patch.object(inference, "load_tool_env", return_value=FailSecond):
                with patch.object(sys, "argv", cli):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        inference.main()
            checkpoint = output.with_suffix(".jsonl.checkpoint")
            self.assertEqual(len(list(inference.read_rows(checkpoint))), 1)

            with patch.object(inference, "load_tool_env", return_value=FakeToolEnv):
                with patch.object(sys, "argv", cli + ["--resume"]):
                    inference.main()
            self.assertEqual([row["id"] for row in inference.read_rows(output)],
                             ["sample_1", "sample_2"])
            metrics = json.loads(output.with_suffix(".metrics.json").read_text())
            self.assertEqual(metrics["samples"], 2)

    def test_replay_and_inference_write_same_hash_and_calls(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT, prefix=".lifemate-test-") as temporary:
            root = Path(temporary)
            split = root / "train"
            package = split / "zaic_tools"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                "from test_inference import FakeToolEnv as ToolEnv\n", encoding="utf-8",
            )
            audio = split / "audio" / "train_0001" / "train_0001.wav"
            audio.parent.mkdir(parents=True)
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x00\x00")
            sample = {
                "id": "train_0001",
                "audio_path": audio.relative_to(root).as_posix(),
                "context": {"installed_apps": {"home": {"tools": ["home.list_lights", "home.turn_on"]}}},
                "label": {"tool_calls": [
                    {"name": "home.list_lights", "arguments": {}},
                    {"name": "home.turn_on", "arguments": {"id": "light_42"}},
                ]},
            }
            (split / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            catalog = {"apps": {
                "home": {"tools": [
                    {"name": "home.list_lights", "parameters": {"type": "object"}},
                    {"name": "home.turn_on", "parameters": {"type": "object"}},
                ]},
                "other": {"tools": [{"name": "other.write"}]},
            }}
            (split / "train_tools_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            gold = root / "gold.jsonl"
            predicted = root / "predicted.jsonl"
            cli = ["inference.py", "--data-root", str(root)]
            with patch.object(sys, "path", sys.path[:]), patch.dict(sys.modules):
                with patch.object(sys, "argv", cli + ["--replay-gold", "--output", str(gold)]):
                    inference.main()
                with patch.object(inference, "LocalModels", FakeModels), patch.object(sys, "argv", cli + [
                    "--asr", str(root), "--llm", str(root),
                    "--gold-replay", str(gold), "--output", str(predicted),
                ]):
                    inference.main()

            self.assertEqual(predicted.read_text(encoding="utf-8"), gold.read_text(encoding="utf-8"))
            row = json.loads(predicted.read_text(encoding="utf-8"))
            self.assertEqual(row["tool_calls"], sample["label"]["tool_calls"])
            self.assertEqual(row["state_hash"], "fixture-on")
            metrics = json.loads(predicted.with_suffix(".metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["state_hash_match_rate"], 1)
            self.assertEqual(metrics["statuses"], {"done": 1})
            with patch.object(inference, "LocalModels", side_effect=AssertionError("model reloaded")):
                with patch.object(sys, "argv", cli + [
                    "--asr", str(root), "--llm", str(root),
                    "--gold-replay", str(gold), "--output", str(predicted),
                    "--resume",
                ]):
                    inference.main()

    def test_out_of_scope_does_not_call_tool(self):
        class StopModel:
            def next_action(self, messages, deadline):
                return '{"done":true}'

        with tempfile.TemporaryDirectory(dir=TEST_ROOT, prefix=".lifemate-test-") as temporary:
            split = Path(temporary) / "train"
            split.mkdir()
            (split / "train.jsonl").touch()
            env = FakeToolEnv("train", "scope_test", temporary)
            result = inference.run_agent(
                env, StopModel(), {}, [], "Viết một bài thơ",
                [{"name": "home.turn_on"}], 1, monotonic() + 30,
            )
            self.assertEqual(result, "done")
            self.assertEqual(env.get_tool_calls(), [])
            self.assertEqual(env.get_state_hash(), "fixture-off")

    def test_audio_path_cannot_read_environment_files(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT, prefix=".lifemate-test-") as temporary:
            split = Path(temporary)
            audio_root = split / "audio"
            audio_root.mkdir()
            private = split / "train_environment"
            private.mkdir()
            (private / "state.json").touch()
            with self.assertRaises(FileNotFoundError):
                inference.audio_for({"audio_path": "../train_environment/state.json"}, audio_root)
