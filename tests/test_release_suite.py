import unittest
from pathlib import Path

from scripts.check_i2av_lat_assets import validate_assets

from acoustitrace.manifests import PromptRecord
from acoustitrace.release import load_release_profile, validate_release_suite


def release_rows():
    rows = []
    groups = [
        ("range", 90, ("range_attenuation",)),
        ("receiver", 90, ("approach_gain", "lateral_stability")),
        ("motion", 90, ("motion_loudness",)),
        ("impact", 90, ("impact_decay",)),
        ("causality", 79, ("causality_violation",)),
        ("rt60", 166, ("rt60_consistency",)),
    ]
    for prefix, count, membership in groups:
        for index in range(count):
            prompt_id = f"{prefix}-{index:03d}"
            rows.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    task="t2av",
                    prompt_text=f"Prompt {prompt_id}",
                    evaluator_membership=membership,
                    evaluator_inputs=(
                        {"receiver_observer": {"detection_targets": ["object"]}}
                        if prefix in {"range", "receiver"}
                        else {}
                    ),
                )
            )
    for index in range(143):
        prompt_id = f"lat-{index:03d}"
        rows.append(
            PromptRecord(
                prompt_id=prompt_id,
                task="i2av",
                prompt_text=f"Prompt {prompt_id}",
                evaluator_membership=("log_attack_time",),
                conditioning_asset_id=f"condition-{prompt_id}",
                evaluator_inputs={
                    "log_attack_time": {
                        "reference_audio_path": f"references/{prompt_id}.wav"
                    }
                },
            )
        )
    return rows


class PublicReleaseSuiteTests(unittest.TestCase):
    def test_bundled_i2av_lat_assets(self):
        root = Path(__file__).resolve().parents[1]
        manifest = root / "data" / "prompts" / "i2av_lat_143.jsonl"
        self.assertEqual(validate_assets(root, manifest), [])
        notice = (root / "data" / "ASSET_NOTICE.md").read_text(encoding="utf-8")
        self.assertIn("CC BY 4.0", notice)
        self.assertIn("https://andrewowens.com/vis/", notice)

    def test_rebalanced_605_plus_143_profile(self):
        rows = release_rows()
        self.assertEqual(len(rows), 748)
        self.assertEqual(
            validate_release_suite(rows, load_release_profile()),
            [],
        )

    def test_t2av_lat_shorthand_cannot_replace_lateral_stability(self):
        rows = release_rows()
        rows[90] = PromptRecord(
            prompt_id=rows[90].prompt_id,
            task="t2av",
            prompt_text=rows[90].prompt_text,
            evaluator_membership=("approach_gain",),
        )
        errors = validate_release_suite(rows, load_release_profile())
        self.assertTrue(any("unsupported release membership" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
