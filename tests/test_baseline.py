import argparse
from pathlib import Path
import tempfile
import unittest

import baseline
from model_cache import stage_models


class BaselineEntryPointTests(unittest.TestCase):
    def test_finds_data_and_models_above_src_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ZAI"
            (root / "data").mkdir(parents=True)
            (root / "models").mkdir()
            repo = root / "src" / "ZaloAI"
            repo.mkdir(parents=True)
            self.assertEqual(baseline.find_project_root(repo), root)

    def test_limit_controls_sample_count_without_extra_model_loads(self):
        root = Path("/ZAI")
        args = argparse.Namespace(split="train", limit=10,
                                  checkpoint_every=None, resume=False)
        command = baseline.command_for(args, root)
        self.assertEqual(command[command.index("--limit") + 1], "10")
        self.assertEqual(command[command.index("--checkpoint-every") + 1], "1")
        args.limit = None
        command = baseline.command_for(args, root)
        self.assertNotIn("--limit", command)
        self.assertEqual(command[command.index("--checkpoint-every") + 1], "25")

    def test_public_test_writes_submission_beside_data(self):
        args = argparse.Namespace(split="public_test", limit=None,
                                  checkpoint_every=None, resume=True)
        command = baseline.command_for(args, Path("/ZAI"))
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--output") + 1],
                         str(Path("/ZAI/result/submission.jsonl")))

    def test_local_model_cache_resumes_partial_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models"
            destination = root / "cache"
            model = source / "Qwen3-4B-Instruct-2507"
            model.mkdir(parents=True)
            (model / "config.json").write_bytes(b"config")
            (model / "weights.bin").write_bytes(b"0123456789")
            partial = destination / model.name / "weights.bin.part"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"0123")
            self.assertEqual(stage_models(source, destination, report=lambda _: None),
                             destination)
            self.assertEqual((destination / model.name / "weights.bin").read_bytes(),
                             b"0123456789")
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
