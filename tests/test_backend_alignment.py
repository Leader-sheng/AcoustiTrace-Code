import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "experiments" / "evaluator_backends"


class BackendAlignmentTests(unittest.TestCase):
    def test_full_evaluator_dependencies_cover_native_backends(self):
        requirements = (ROOT / "requirements-eval.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.8.0", requirements)
        self.assertIn("torchaudio==2.8.0", requirements)
        self.assertIn("torchvision==0.23.0", requirements)
        self.assertIn("addict>=2.4", requirements)
        self.assertIn("qwen-vl-utils==0.0.14", requirements)
        self.assertIn("vllm==0.11.0", requirements)

        preflight = (ROOT / "scripts" / "check_evaluator_setup.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"addict": "addict (required by Grounded-SAM)"', preflight)
        self.assertIn("def check_imports(", preflight)
        self.assertIn('"--source-python"', preflight)
        self.assertIn('"pytorchvideo": "PyTorchVideo', preflight)

    def test_receiver_recomputes_failed_dynamic_caches(self):
        gsam_source = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "02_run_grounded_sam_dynamic.py"
        ).read_text(encoding="utf-8")
        track_source = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "04_compute_depth_tracks.py"
        ).read_text(encoding="utf-8")
        self.assertIn("dynamic_cache_is_usable", gsam_source)
        self.assertIn('row.get("status", "")) != "failed"', gsam_source)
        self.assertIn("cached_track_is_usable", track_source)
        self.assertIn('source == "dynamic_keyframe_gsam"', track_source)

    def test_receiver_reuses_grounded_sam_models_across_keyframes(self):
        config = (
            BACKENDS
            / "receiver_observer"
            / "configs"
            / "receiver_observer_eval_config.yaml"
        ).read_text(encoding="utf-8")
        source = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "02_run_grounded_sam_dynamic.py"
        ).read_text(encoding="utf-8")
        runtime = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "grounded_sam_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("persistent_runtime: true", config)
        self.assertIn("_PERSISTENT_GSAM", source)
        self.assertIn("PersistentGroundedSAM(cfg)", source)
        self.assertIn("class PersistentGroundedSAM", runtime)
        self.assertIn('out_dir / "mask.json"', runtime)

    def test_receiver_uses_sign_aware_window_search(self):
        base_source = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "05_compute_receiver_observer_metrics.py"
        ).read_text(encoding="utf-8")
        range_source = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "10_compute_sign_aware_range_attenuation.py"
        ).read_text(encoding="utf-8")
        stage07 = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "07_compute_receiver_observer_unified_v2.py"
        ).read_text(encoding="utf-8")
        adapter = (
            ROOT / "experiments" / "evaluator_adapters" / "receiver_observer_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn("windowed_distance_r2_search", base_source)
        self.assertIn("if float(np.nanmedian(distance)) < 0", range_source)
        self.assertIn("for exponent in exponents", range_source)
        self.assertIn("sign_aware_windowed_inverse_square_fit_r2_proxy", adapter)
        self.assertIn('scripts / "07_compute_receiver_observer_unified_v2.py"', adapter)
        self.assertNotIn('"--existing_metrics"', adapter)
        self.assertNotIn("from_existing_metrics", stage07)
        self.assertIn('"--metrics_csv", str(unified_csv)', adapter)

    def test_receiver_empty_audio_path_uses_extracted_audio_file(self):
        source = (
            BACKENDS
            / "receiver_observer"
            / "scripts"
            / "05_compute_receiver_observer_metrics.py"
        ).read_text(encoding="utf-8")
        self.assertIn('audio_root / sample_id / "extracted_audio.wav"', source)
        self.assertIn("if event_audio_path.is_file():", source)
        self.assertNotIn("if event_audio_path.exists():", source)

    def test_causality_separates_matching_and_scoring_thresholds(self):
        config = (
            BACKENDS / "time_causality_eval" / "configs" / "time_causality_config.yaml"
        ).read_text(encoding="utf-8")
        source = (
            BACKENDS
            / "time_causality_eval"
            / "scripts"
            / "time_causality_pipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn("violation_threshold_sec: -0.001", config)
        self.assertIn("early_tolerance_sec: 0.08", config)
        self.assertIn("delay < -early_tol or delay > max_delay", source)
        self.assertIn("delay < _violation_threshold(cfg)", source)

    def test_receiver_applicability_matches_supplement(self):
        config = (
            BACKENDS
            / "receiver_observer"
            / "configs"
            / "receiver_observer_unified_v2_config.yaml"
        ).read_text(encoding="utf-8")
        for line in (
            "window_sec: 0.4",
            "slide_sec: 0.05",
            "smoothing_window_sec: 0.35",
            "min_valid_points: 6",
            "min_valid_track_ratio: 0.50",
            "distance_min_receding_duration_sec: 1.50",
            "min_motion_depth_change: 2.00",
            "approaching_min_motion_depth_change: 0.15",
            "lateral_min_bbox_center_motion: 0.03",
            "lateral_min_bbox_path_length: 0.05",
        ):
            self.assertIn(line, config)

    def test_visual_rt60_configuration_matches_paper(self):
        config = (
            ROOT
            / "experiments"
            / "evaluator_backends"
            / "rt60"
            / "rt60_runtime.yaml"
        ).read_text(encoding="utf-8")
        for line in (
            "hidden_size: 4096",
            "lora_r: 16",
            "lora_alpha: 32",
            "lora_dropout: 0.0",
            "gradient_checkpointing: false",
            "attn_implementation: sdpa",
            "rt60_min: 0.05",
            "rt60_max: 5.0",
        ):
            self.assertIn(line, config)

    def test_lat_is_separate_from_material_scoring(self):
        source = (BACKENDS / "log_attack_time" / "log_attack_time.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("SCORE_SCALE = 0.35", source)
        self.assertIn("FRAME_LENGTH = 1024", source)
        self.assertIn("HOP_LENGTH = 128", source)
        self.assertNotIn("material_score", source)

    def test_motion_uses_two_clusters_motion_only_qwen_and_fallback(self):
        clusterer = (
            BACKENDS
            / "source_mechanics"
            / "scripts"
            / "09_prepare_motion_loudness_cluster_pairs.py"
        ).read_text(encoding="utf-8")
        qwen = (
            BACKENDS
            / "source_mechanics"
            / "scripts"
            / "10_qwen3_motion_loudness_cluster_score.py"
        ).read_text(encoding="utf-8")
        fallback = (
            BACKENDS
            / "source_mechanics"
            / "scripts"
            / "11_apply_motion_loudness_frame_diff_fallback.py"
        ).read_text(encoding="utf-8")
        adapter = (
            ROOT / "experiments" / "evaluator_adapters" / "source_mechanics_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("choose_split", clusterer)
        self.assertIn("fewer_than_two_matched_av_events", clusterer)
        self.assertNotIn("audio_peak_events", clusterer)
        self.assertNotIn("audio_rms_peaks", clusterer)
        self.assertIn("ACOUSTITRACE_TRITON_PTXAS_PATH", qwen)
        self.assertIn('os.environ["TRITON_PTXAS_PATH"]', qwen)
        self.assertLess(qwen.index("configure_triton_ptxas()"), qwen.index("from vllm import"))
        self.assertIn("不能使用音频", qwen)
        self.assertIn("invalid_contact_reasoning", qwen)
        self.assertIn("sample_fps: float = 8.0", fallback)
        self.assertIn("frame_diff_fallback_rescue", fallback)
        self.assertIn("motion_error =", adapter)
        self.assertIn("motion_loudness_backend_failed", adapter)
        self.assertIn('elif evaluator == "impact_decay"', adapter)

    def test_ov_avel_anchors_imagebind_bpe_to_checkout(self):
        runner = (
            BACKENDS
            / "source_mechanics"
            / "scripts"
            / "ov_avel_batch_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("imagebind_data.BPE_PATH = str(imagebind_root / bpe_path)", runner)
        self.assertIn(
            "[f\"The sound of {label}.\" for label in classes], device\n        ).to(device)",
            runner,
        )

    def test_flexsed_uses_documented_local_clap_checkpoint(self):
        runner = (
            BACKENDS
            / "source_mechanics"
            / "scripts"
            / "flexsed_batch_runner.py"
        ).read_text(encoding="utf-8")
        adapter = (
            ROOT / "experiments" / "evaluator_adapters" / "source_mechanics_adapter.py"
        ).read_text(encoding="utf-8")
        preflight = (ROOT / "scripts" / "check_evaluator_setup.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("clap_model_path", runner)
        self.assertIn('identifier == "laion/clap-htsat-unfused"', runner)
        self.assertIn('loader_kwargs["local_files_only"] = True', runner)
        self.assertIn("checkpoints/flexsed/laion-clap-htsat-unfused", adapter)
        self.assertIn("FlexSED CLAP weights", preflight)

    def test_lat_download_has_cross_process_range_resume(self):
        source = (
            ROOT / "scripts" / "download_greatest_hits_zenodo_subset.py"
        ).read_text(encoding="utf-8")
        self.assertIn("start_position: int = 0", source)
        self.assertIn('archive_path.suffix + ".part"', source)
        self.assertIn("start_position=downloaded", source)
        self.assertIn("partial_path.replace(archive_path)", source)
        self.assertIn('"--archive-cache-dir"', source)
        self.assertNotIn('?download=1"', source)


if __name__ == "__main__":
    unittest.main()
