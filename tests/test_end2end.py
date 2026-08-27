import csv
import json
from pathlib import Path
import tempfile
import unittest

from acoustitrace.end2end import _resolve_manifest_path, evaluate_release
try:
    from tests.test_release_suite import release_rows
except ModuleNotFoundError:  # unittest discovery imports test modules as top-level files
    from test_release_suite import release_rows


class EndToEndTests(unittest.TestCase):
    def test_repo_relative_asset_path_is_preferred_over_manifest_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            manifest_base = repo_root / "data" / "prompts"
            reference = repo_root / "data" / "references" / "sample.wav"
            reference.parent.mkdir(parents=True)
            reference.touch()

            resolved = _resolve_manifest_path(
                "data/references/sample.wav",
                manifest_base,
                repo_root,
            )

            self.assertEqual(Path(resolved), reference.resolve())

    def test_manifest_relative_asset_path_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            manifest_base = repo_root / "external" / "prompts"
            reference = manifest_base / "references" / "sample.wav"
            reference.parent.mkdir(parents=True)
            reference.touch()

            resolved = _resolve_manifest_path(
                "references/sample.wav",
                manifest_base,
                repo_root,
            )

            self.assertEqual(Path(resolved), reference.resolve())

    def test_full_release_routes_and_scores_with_protocol_mock(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = release_rows()
            t2av_path = root / "t2av_605.jsonl"
            i2av_path = root / "i2av_lat_143.jsonl"
            for path, task in ((t2av_path, "t2av"), (i2av_path, "i2av")):
                with path.open("w", encoding="utf-8") as handle:
                    for row in prompts:
                        if row.task != task:
                            continue
                        handle.write(
                            json.dumps(
                                {
                                    "prompt_id": row.prompt_id,
                                    "task": row.task,
                                    "prompt_text": row.prompt_text,
                                    "evaluator_membership": list(row.evaluator_membership),
                                    "conditioning_asset_id": row.conditioning_asset_id,
                                    "evaluator_inputs": row.evaluator_inputs,
                                }
                            )
                            + "\n"
                        )
            outputs_path = root / "outputs.csv"
            with outputs_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "prompt_id",
                        "task",
                        "model",
                        "video_path",
                        "status",
                    ),
                )
                writer.writeheader()
                for row in prompts:
                    writer.writerow(
                        {
                            "prompt_id": row.prompt_id,
                            "task": row.task,
                            "model": "test-model",
                            "video_path": f"videos/{row.task}/{row.prompt_id}.mp4",
                            "status": "success",
                        }
                    )
            result = evaluate_release(
                prompt_paths=(t2av_path, i2av_path),
                outputs_path=outputs_path,
                backend_config_path=repo_root / "configs/evaluator_backends.mock.json",
                output_dir=root / "results",
                allow_mock=True,
                check_files=False,
                resamples=10,
            )
            self.assertEqual(result["status"], "success")
            self.assertFalse(result["official_scores"])
            self.assertEqual(result["score_rows"], 838)
            self.assertEqual(result["valid_score_rows"], 838)
            with (root / "results/summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 8)

    def test_smoke_limit_runs_ten_assignments_per_dimension(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = release_rows()
            prompt_paths = []
            for task, filename in (("t2av", "t2av_605.jsonl"), ("i2av", "i2av_lat_143.jsonl")):
                path = root / filename
                prompt_paths.append(path)
                with path.open("w", encoding="utf-8") as handle:
                    for row in prompts:
                        if row.task == task:
                            handle.write(
                                json.dumps(
                                    {
                                        "prompt_id": row.prompt_id,
                                        "task": row.task,
                                        "prompt_text": row.prompt_text,
                                        "evaluator_membership": list(row.evaluator_membership),
                                        "conditioning_asset_id": row.conditioning_asset_id,
                                        "evaluator_inputs": row.evaluator_inputs,
                                    }
                                )
                                + "\n"
                            )
            outputs_path = root / "outputs.csv"
            with outputs_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("prompt_id", "task", "model", "video_path", "status"),
                )
                writer.writeheader()
                for row in prompts:
                    writer.writerow(
                        {
                            "prompt_id": row.prompt_id,
                            "task": row.task,
                            "model": "test-model",
                            "video_path": f"videos/{row.task}/{row.prompt_id}.mp4",
                            "status": "success",
                        }
                    )
            result = evaluate_release(
                prompt_paths=prompt_paths,
                outputs_path=outputs_path,
                backend_config_path=repo_root / "configs/evaluator_backends.mock.json",
                output_dir=root / "smoke-results",
                allow_mock=True,
                check_files=False,
                resamples=10,
                smoke_limit_per_dimension=10,
            )
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["smoke_test"])
            self.assertFalse(result["official_scores"])
            self.assertEqual(result["score_rows"], 80)
            self.assertLess(result["prompt_count"], 748)


if __name__ == "__main__":
    unittest.main()
