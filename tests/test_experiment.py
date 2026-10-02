import json
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from experiment import (
    EXPERIMENT_CONDITIONS,
    _build_output_dir,
    _save_comparison,
    _print_summary,
    _clear_vram,
    _destroy_vllm,
    run_experiment,
    main,
)
from elfarol.simulation_runner import VLLMBatchPipeline


class MockVLLMOutput:

    def __init__(self, text):
        self.outputs = [MagicMock(text=text)]


class MockVLLM:

    def __init__(self, response_fn=None):
        self.response_fn = response_fn or (lambda p: '{"move": [25, 25], "broadcast": "at bar", "proximity_speech": null, "stated_intention": "going", "actual_target": "bar"}')

    def generate(self, prompts, sampling_params):
        return [MockVLLMOutput(self.response_fn(p)) for p in prompts]


class TestExperimentHelpers:

    def test_conditions_defined(self):
        assert len(EXPERIMENT_CONDITIONS) == 2
        labels = [c["label"] for c in EXPERIMENT_CONDITIONS]
        assert "control" in labels
        assert "delta2" in labels

    def test_build_output_dir(self):
        out = _build_output_dir("base_outputs", "control")
        assert "base_outputs" in out
        assert "control_" in out

    def test_save_comparison(self, tmp_path):
        results = {
            "control": {"total_epochs": 10, "average_bar_attendance": 20.0},
            "delta2": {"total_epochs": 10, "average_bar_attendance": 35.0},
        }
        path = _save_comparison(results, str(tmp_path))
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["condition"] == "control"
        assert data[1]["condition"] == "delta2"

    def test_print_summary_runs_without_error(self, capsys):
        summaries = {
            "control": {
                "total_epochs": 5,
                "average_bar_attendance": 25.0,
                "average_deception_index": 0.1,
                "population_truthfulness": 0.9,
                "average_mean_utility": 0.45,
            },
            "delta2": {
                "total_epochs": 5,
                "average_bar_attendance": 30.0,
                "average_deception_index": 0.35,
                "population_truthfulness": 0.65,
                "average_mean_utility": 0.2,
            },
        }
        _print_summary(summaries)
        captured = capsys.readouterr().out
        assert "control" in captured
        assert "delta2" in captured
        assert "Avg Attendance" in captured

    def test_clear_vram_and_destroy_vllm_no_crash(self):
        _clear_vram()
        mock_llm = MagicMock()
        _destroy_vllm(mock_llm)


class TestRunExperiment:

    def test_run_experiment_with_dry_run_pipeline(self, tmp_path):
        mock_vllm = MockVLLM()
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)

        summaries, comparison_path = run_experiment(
            model_name="test-model",
            base_output_dir=str(tmp_path),
            num_agents=6,
            num_epochs=3,
            grid_size=20,
            seed=42,
            dry_run_pipeline=pipeline,
        )

        assert "control" in summaries
        assert "delta2" in summaries
        assert os.path.exists(comparison_path)

        with open(comparison_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        assert len(comp_data) == 2

    def test_run_experiment_multi_seed_sweep(self, tmp_path):
        mock_vllm = MockVLLM()
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)

        summaries, comparison_path = run_experiment(
            model_name="test-model",
            base_output_dir=str(tmp_path),
            num_agents=4,
            num_epochs=2,
            grid_size=20,
            seeds=[42, 100, 2026],
            dry_run_pipeline=pipeline,
        )

        assert len(summaries) == 6
        expected_keys = [
            "control_seed_42",
            "control_seed_100",
            "control_seed_2026",
            "delta2_seed_42",
            "delta2_seed_100",
            "delta2_seed_2026",
        ]
        for k in expected_keys:
            assert k in summaries
            assert (tmp_path / k).is_dir()

        with open(comparison_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        assert len(comp_data) == 6



class TestExperimentCLI:

    def test_main_cli_execution(self, tmp_path):
        test_args = [
            "experiment.py",
            "--model", "test-model",
            "--output-dir", str(tmp_path),
            "--agents", "4",
            "--epochs", "2",
            "--seed", "99",
        ]
        mock_vllm = MockVLLM()
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)

        with patch.object(sys, "argv", test_args):
            with patch("experiment._build_vllm_pipeline", return_value=(mock_vllm, None)):
                with patch("experiment._destroy_vllm"):
                    main()

        assert (tmp_path / "comparison.json").exists()
