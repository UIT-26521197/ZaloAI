import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.config import Config
from src.inference.run import run
from src.training.dataset import CompletionDataset
from src.utils.paths import read_rows


class FixtureEnv:
    def __init__(self, split, sample_id, root):
        self.calls = []

    def validate(self, name, arguments):
        ok = name == "home.list_lights" or arguments == {"id": "lamp_1"}
        return {"ok": ok}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        return {"id": "lamp_1"} if name == "home.list_lights" else {"ok": True}

    def get_tool_calls(self):
        return self.calls

    def get_state_hash(self):
        return "on" if len(self.calls) == 2 else "off"


class PipelineTest(unittest.TestCase):
    def test_validation_observations_and_submission(self):
        with TemporaryDirectory(dir=Path.cwd()) as path:
            root = Path(path)
            split = root / "train"
            (split / "audio").mkdir(parents=True)
            (split / "audio" / "one.wav").touch()
            schemas = [{"name": f"home.{name}"} for name in ("list_lights", "turn_on")]
            (split / "train_tools_catalog.json").write_text(
                json.dumps({"apps": {"home": {"tools": schemas}}}), encoding="utf-8",
            )
            calls = [
                {"name": "home.list_lights", "arguments": {}},
                {"name": "home.turn_on", "arguments": {"id": "lamp_1"}},
            ]
            sample = {
                "id": "one", "audio_path": "train/audio/one.wav",
                "context": {"installed_apps": ["home"]},
                "label": {"tool_calls": calls, "state_hash": "on"},
            }
            (split / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            ids = root / "ids.json"
            ids.write_text('["one"]', encoding="utf-8")
            config = Config(
                mode="baseline", split="train", data_root=root, output=root / "out",
                whisper="", laya="", qwen="", qwen_adapter=None, device="cpu", laya_device="cpu",
                offline=True, seed=42, route_threshold=0.6, max_steps=5, max_seconds=28,
                context_tokens=2048, new_tokens=128, limit=None, ids=ids,
            )
            steps = iter([
                {"name": "home.turn_on", "arguments": {}}, *calls, {"stop": True},
            ])
            prompts = []

            def generate(messages, seconds):
                prompts.append(json.loads(messages[-1]["content"]))
                return json.dumps(next(steps))

            models = SimpleNamespace(
                whisper=SimpleNamespace(transcribe=lambda path: "Bật đèn bếp"),
                laya=SimpleNamespace(choose=lambda transcript, criteria: ("direct", 0.42)),
                qwen=SimpleNamespace(generate=generate),
            )
            fixture = SimpleNamespace(ToolEnv=FixtureEnv)
            with patch.dict("sys.modules", {"zaic_tools": fixture}):
                with patch("src.inference.run.load_models", return_value=models):
                    output = run(config)

            self.assertEqual(list(read_rows(output)), [
                {"id": "one", "tool_calls": calls, "state_hash": "on"},
            ])
            self.assertEqual(prompts[0]["transcript"], "Bật đèn bếp")
            self.assertEqual(prompts[1]["observations"][0]["rejected"], {"ok": False})
            self.assertEqual(prompts[2]["observations"][1]["result"], {"id": "lamp_1"})
            trace = next(read_rows(config.output / "trace.jsonl"))
            self.assertEqual(trace["route"]["kind"], "general")
            self.assertEqual(trace["status"], "done")
            metrics = json.loads((config.output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["state_hash_match_rate"], 1)

    def test_completion_loss_excludes_the_prompt(self):
        class Tokenizer:
            eos_token_id = 9

            def apply_chat_template(self, prompt, **kwargs):
                return prompt

            def __call__(self, completion, **kwargs):
                return {"input_ids": completion}

        rows = CompletionDataset([([1, 2], [3, 4]), ([1] * 10, [3])], Tokenizer(), 10, 4)
        self.assertEqual(rows[0], {"input_ids": [1, 2, 3, 4, 9], "labels": [-100, -100, 3, 4, 9]})
        self.assertEqual(rows.skipped, 1)


if __name__ == "__main__":
    unittest.main()
