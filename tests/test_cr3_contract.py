import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments" / "evaluator_backends" / "rt60"
CONTRACT = ROOT / "models" / "rt60_evaluator_contract.json"


class RT60RuntimeContractTests(unittest.TestCase):
    def test_runtime_contract_records_released_architecture(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["method"], "sabine_guided_physics_head")
        self.assertEqual(contract["scene_representation"]["dimension"], 4096)
        self.assertEqual(contract["adapter"]["rank"], 16)
        self.assertEqual(contract["adapter"]["alpha"], 32)
        self.assertEqual(contract["head"]["rt60_seconds_range"], [0.05, 5.0])

    def test_inference_has_no_direct_sft_fallback(self):
        source = (RUNTIME / "visual_physics_runtime.py").read_text(encoding="utf-8")
        adapter = (
            ROOT / "experiments" / "evaluator_adapters" / "rt60_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("AutoModelForVision2Seq", source)
        self.assertNotIn(".generate(", source)
        self.assertIn("PeftModel.from_pretrained", source)
        self.assertIn('checkpoint / "physics_head.pt"', adapter)
        self.assertIn("load_state_dict(state)", adapter)

    def test_runtime_and_contract_artifacts_are_present(self):
        expected = [
            CONTRACT,
            RUNTIME / "audio_rt60_proxy.py",
            RUNTIME / "visual_physics_runtime.py",
            RUNTIME / "rt60_runtime.yaml",
        ]
        self.assertEqual([str(path) for path in expected if not path.is_file()], [])


if __name__ == "__main__":
    unittest.main()
