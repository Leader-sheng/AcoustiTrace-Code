import unittest

from acoustitrace.manifests import PromptRecord, validate_prompt_manifest


def make_record(prompt_id, task, *members, condition=True):
    return PromptRecord(
        prompt_id=prompt_id,
        task=task,
        prompt_text=f"Prompt {prompt_id}",
        evaluator_membership=tuple(members),
        conditioning_asset_id=f"image-{prompt_id}" if task == "i2av" and condition else "",
    )


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "prompt_suite": {
                "unique": {"t2av": 5, "i2av": 6},
                "assignments": {
                    "range_attenuation": 2,
                    "approach_gain": 2,
                    "lateral_stability": 2,
                    "motion_loudness": 1,
                    "impact_decay": 1,
                    "causality_violation": 1,
                    "rt60_consistency": 1,
                    "log_attack_time": 1,
                },
                "range_receiver_overlap": 1,
            }
        }

    def rows(self, task):
        core = [
            make_record("p1", task, "range_attenuation", "approach_gain", "lateral_stability"),
            make_record("p2", task, "range_attenuation"),
            make_record("p3", task, "approach_gain", "lateral_stability"),
            make_record("p4", task, "motion_loudness", "causality_violation"),
            make_record("p5", task, "impact_decay", "rt60_consistency"),
        ]
        if task == "i2av":
            core.append(make_record("p6", task, "log_attack_time"))
        return core

    def test_final_pool_relations(self):
        self.assertEqual(validate_prompt_manifest(self.rows("t2av"), self.contract), [])
        self.assertEqual(validate_prompt_manifest(self.rows("i2av"), self.contract), [])

    def test_approach_and_lateral_must_share_a_pool(self):
        rows = self.rows("t2av")
        rows[2] = make_record("p3", "t2av", "approach_gain")
        errors = validate_prompt_manifest(rows, self.contract)
        self.assertTrue(any("pools differ" in error for error in errors))

    def test_i2av_requires_conditioning_asset(self):
        rows = self.rows("i2av")
        rows[-1] = make_record("p6", "i2av", "log_attack_time", condition=False)
        errors = validate_prompt_manifest(rows, self.contract)
        self.assertTrue(any("no conditioning asset" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
