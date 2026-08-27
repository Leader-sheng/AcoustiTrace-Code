import unittest
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SubmissionHygieneTests(unittest.TestCase):
    @staticmethod
    def submission_text_files():
        # These checks cover files that belong to the source release.  Runtime
        # environments, downloaded checkpoints, submissions, and generated
        # outputs are intentionally gitignored and may contain arbitrary
        # third-party metadata or binary strings.
        runtime_roots = {
            ".git",
            "checkpoints",
            "third_party",
            "submissions",
            "results",
            "outputs",
            "work",
        }
        binary_suffixes = {
            ".pdf",
            ".png",
            ".pyc",
            ".wav",
            ".whl",
            ".so",
            ".pt",
            ".pth",
            ".safetensors",
        }
        for path in ROOT.rglob("*"):
            if path.resolve() == Path(__file__).resolve():
                continue
            relative = path.relative_to(ROOT)
            # Development machines may keep multiple disposable environments
            # (for example .venv-source) next to the source tree.
            if relative.parts and relative.parts[0].startswith(".venv"):
                continue
            if relative.parts and relative.parts[0] in runtime_roots:
                continue
            # These folders are populated by the LAT reference downloader and
            # are runtime assets, not files in the source release.
            if relative.parts[:3] in {
                ("data", "references", "greatest_hits_zenodo"),
                ("data", "references", "greatest_hits_zenodo_archives"),
            }:
                continue
            if path.is_file() and path.suffix.lower() not in binary_suffixes:
                yield path

    def test_stale_starss_protocol_is_not_shipped(self):
        forbidden = (
            "robust_repeated_event_median",
            "fallback gray pseudo-depth",
            "only 3/26 strict audio-valid",
        )
        offenders = []
        for path in self.submission_text_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(term in text for term in forbidden):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_private_paths_and_user_identifiers_are_not_shipped(self):
        private_path_patterns = (
            re.compile(r"(?i)(?:^|[\s\"'=])[A-Z]:\\"),
            re.compile(r"(?i)/(?:home|mnt)/"),
            re.compile(r"(?i)(?:--partition=|--gres=)"),
        )
        offenders = []
        for path in self.submission_text_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in private_path_patterns):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_no_common_literal_credential_formats_are_shipped(self):
        credential_patterns = (
            re.compile(r"AKIA[0-9A-Z]{16}"),
            re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
            re.compile(r"hf_[A-Za-z0-9]{20,}"),
            re.compile(r"sk-[A-Za-z0-9]{20,}"),
        )
        offenders = []
        for path in self.submission_text_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in credential_patterns):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_runtime_environment_file_is_not_shipped(self):
        self.assertFalse((ROOT / ".env").exists())

    def test_internal_experiment_workspaces_are_not_shipped(self):
        forbidden = (
            "visual_rt60_training",
            "rt60_bras_cr3",
            "rt60_starss23",
            "controlled_perturbations",
            "data_collection",
            "range_guidance",
        )
        present = [
            name for name in forbidden if (ROOT / "experiments" / name).exists()
        ]
        self.assertEqual(present, [])


if __name__ == "__main__":
    unittest.main()
