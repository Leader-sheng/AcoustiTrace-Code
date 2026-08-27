import unittest
from pathlib import Path

from acoustitrace.contract import load_contract, validate_contract


class ContractTests(unittest.TestCase):
    def test_release_contract_is_self_consistent(self):
        root = Path(__file__).resolve().parents[1]
        contract = load_contract(root / "configs" / "paper_contract.json")
        self.assertEqual(validate_contract(contract), [])


if __name__ == "__main__":
    unittest.main()

