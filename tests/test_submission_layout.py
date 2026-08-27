from pathlib import Path
import tempfile
import unittest

from acoustitrace.end2end import EvaluationError, discover_video_outputs
from acoustitrace.manifests import PromptRecord, generation_request


def prompt(prompt_id: str, task: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        task=task,
        prompt_text=f"Prompt {prompt_id}",
        evaluator_membership=(
            ("log_attack_time",) if task == "i2av" else ("motion_loudness",)
        ),
        conditioning_asset_id=(f"image-{prompt_id}" if task == "i2av" else ""),
    )


class SubmissionLayoutTests(unittest.TestCase):
    def test_generation_request_publishes_exact_mp4_name(self):
        t2av = generation_request(prompt("motion_seed_0001", "t2av"))
        i2av = generation_request(prompt("i2av_lat_0001", "i2av"))
        self.assertEqual(t2av["output_filename"], "motion_seed_0001.mp4")
        self.assertEqual(t2av["output_relpath"], "t2av/motion_seed_0001.mp4")
        self.assertEqual(i2av["output_relpath"], "i2av_lat/i2av_lat_0001.mp4")

    def test_directory_discovery_accepts_only_canonical_layout(self):
        prompts = [prompt("motion_seed_0001", "t2av"), prompt("i2av_lat_0001", "i2av")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "t2av").mkdir()
            (root / "i2av_lat").mkdir()
            (root / "t2av/motion_seed_0001.mp4").touch()
            (root / "i2av_lat/i2av_lat_0001.mp4").touch()
            outputs, missing = discover_video_outputs(prompts, root, "test-model")
            self.assertEqual(missing, [])
            self.assertEqual(len(outputs), 2)

    def test_directory_discovery_rejects_noncanonical_video(self):
        prompts = [prompt("motion_seed_0001", "t2av")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "t2av").mkdir()
            (root / "t2av/motion_seed_0001.mov").touch()
            with self.assertRaisesRegex(EvaluationError, "non-canonical"):
                discover_video_outputs(prompts, root, "test-model")

    def test_directory_discovery_rejects_unknown_extra_mp4(self):
        prompts = [prompt("motion_seed_0001", "t2av")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "t2av").mkdir()
            (root / "t2av/motion_seed_0001.mp4").touch()
            (root / "t2av/model_motion_seed_0001.mp4").touch()
            with self.assertRaisesRegex(EvaluationError, "unknown"):
                discover_video_outputs(prompts, root, "test-model")


if __name__ == "__main__":
    unittest.main()
