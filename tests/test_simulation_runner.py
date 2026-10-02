import json
import os
import random
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from elfarol.simulation_runner import (
    RunConfig,
    SimulationRunner,
    VLLMBatchPipeline,
    _parse_brain_json,
    _strip_broadcasts_from_actions,
    NUM_AGENTS,
    GRID_SIZE,
    NUM_EPOCHS,
)
from elfarol.grid import SpatialGrid
from elfarol.agent import Agent, AgentPool
from elfarol.llm import MockLLMPipeline


class MockVLLMOutput:

    def __init__(self, text):
        self.outputs = [MagicMock(text=text)]


class MockVLLM:

    def __init__(self, response_fn=None):
        self.response_fn = response_fn or (lambda p: '{"move": [25, 25], "broadcast": "at bar", "proximity_speech": null, "stated_intention": "going", "actual_target": "bar"}')

    def generate(self, prompts, sampling_params):
        return [MockVLLMOutput(self.response_fn(p)) for p in prompts]


class TestRunConfig:

    def test_default_values(self):
        config = RunConfig()
        assert config.num_agents == NUM_AGENTS
        assert config.grid_size == GRID_SIZE
        assert config.num_epochs == NUM_EPOCHS
        assert config.broadcast_enabled is True
        assert config.condition_label == "delta2"
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.85
        assert config.tensor_parallel_size == 1

    def test_custom_values(self):
        config = RunConfig(
            num_agents=10,
            grid_size=20,
            num_epochs=5,
            seed=123,
            broadcast_enabled=False,
            output_dir="test_out",
            condition_label="control",
        )
        assert config.num_agents == 10
        assert config.grid_size == 20
        assert config.num_epochs == 5
        assert config.seed == 123
        assert config.broadcast_enabled is False
        assert config.condition_label == "control"


class TestParseBrainJSON:

    def test_valid_json(self):
        raw = '{"move": [10, 20], "broadcast": "hello", "proximity_speech": "hi", "stated_intention": "going", "actual_target": "bar"}'
        result = _parse_brain_json(raw)
        assert result == {
            "move": [10, 20],
            "broadcast": "hello",
            "proximity_speech": "hi",
            "stated_intention": "going",
            "actual_target": "bar",
        }

    def test_json_embedded_in_text(self):
        raw = 'Reasoning text here... ```json\n{"move": [5, 5], "broadcast": null, "proximity_speech": null, "stated_intention": "staying", "actual_target": "home"}\n``` done.'
        result = _parse_brain_json(raw)
        assert result is not None
        assert result["move"] == [5, 5]
        assert result["stated_intention"] == "staying"
        assert result["actual_target"] == "home"

    def test_missing_move_returns_none(self):
        raw = '{"broadcast": "hello", "stated_intention": "going"}'
        assert _parse_brain_json(raw) is None

    def test_invalid_move_length_returns_none(self):
        raw = '{"move": [1, 2, 3]}'
        assert _parse_brain_json(raw) is None

    def test_invalid_json_returns_none(self):
        assert _parse_brain_json("invalid json string") is None

    def test_defaults_for_missing_optional_fields(self):
        raw = '{"move": [12, 14]}'
        result = _parse_brain_json(raw)
        assert result["broadcast"] is None
        assert result["proximity_speech"] is None
        assert result["stated_intention"] == "staying"
        assert result["actual_target"] == "home"

    def test_invalid_intention_and_target_fallback(self):
        raw = '{"move": [1, 1], "stated_intention": "invalid", "actual_target": "invalid"}'
        result = _parse_brain_json(raw)
        assert result["stated_intention"] == "staying"
        assert result["actual_target"] == "home"


class TestStripBroadcasts:

    def test_strips_all_broadcasts(self):
        actions = [
            {"move": [1, 1], "broadcast": "loud message", "proximity_speech": "whisper", "stated_intention": "going", "actual_target": "bar"},
            {"move": [2, 2], "broadcast": None, "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"},
        ]
        stripped = _strip_broadcasts_from_actions(actions)
        assert stripped[0]["broadcast"] is None
        assert stripped[0]["proximity_speech"] == "whisper"
        assert stripped[0]["move"] == [1, 1]
        assert stripped[1]["broadcast"] is None


class TestVLLMBatchPipeline:

    def test_successful_batch_generation(self):
        mock_vllm = MockVLLM()
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)
        prompts = ["prompt1", "prompt2", "prompt3"]
        results = pipeline.generate_batch(prompts)
        assert len(results) == 3
        for res in results:
            assert res["move"] == [25, 25]
            assert res["stated_intention"] == "going"
            assert res["actual_target"] == "bar"

    def test_fallback_on_invalid_output(self):
        mock_vllm = MockVLLM(response_fn=lambda p: "MALFORMED_OUTPUT")
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)
        prompts = ["prompt1", "prompt2"]
        fallbacks = [(10, 15), (20, 25)]
        results = pipeline.generate_batch(prompts, fallback_positions=fallbacks)
        assert len(results) == 2
        assert results[0]["move"] == [10, 15]
        assert results[0]["stated_intention"] == "staying"
        assert results[0]["actual_target"] == "home"
        assert results[1]["move"] == [20, 25]

    def test_fallback_without_fallback_positions(self):
        mock_vllm = MockVLLM(response_fn=lambda p: "MALFORMED")
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)
        results = pipeline.generate_batch(["prompt1"])
        assert results[0]["move"] == [0, 0]


class TestSimulationRunnerInit:

    def test_initialization_state(self, tmp_path):
        config = RunConfig(num_agents=10, grid_size=30, num_epochs=3, output_dir=str(tmp_path), seed=42)
        runner = SimulationRunner(config=config, llm_pipeline=MockLLMPipeline())
        assert runner._initialized is False
        runner.initialize()
        assert runner._initialized is True
        assert len(runner.agent_pool.all_agents()) == 10
        assert runner.validate_state() is True


class TestSimulationRunnerStep:

    def test_step_with_broadcast_enabled(self, tmp_path):
        config = RunConfig(num_agents=5, grid_size=20, num_epochs=1, output_dir=str(tmp_path), broadcast_enabled=True)
        resp = lambda p: '{"move": [10, 10], "broadcast": "broadcast_test", "proximity_speech": null, "stated_intention": "going", "actual_target": "bar"}'
        mock_vllm = MockVLLM(response_fn=resp)
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)
        runner = SimulationRunner(config=config, llm_pipeline=pipeline)
        runner.initialize()
        runner._step(0)
        assert len(runner.broadcast_history) == 5
        assert runner.broadcast_history[0] == "broadcast_test"
        assert len(runner.metrics_logger.epoch_snapshots) == 1

    def test_step_with_broadcast_disabled(self, tmp_path):
        config = RunConfig(num_agents=5, grid_size=20, num_epochs=1, output_dir=str(tmp_path), broadcast_enabled=False)
        resp = lambda p: '{"move": [10, 10], "broadcast": "hidden_broadcast", "proximity_speech": null, "stated_intention": "going", "actual_target": "bar"}'
        mock_vllm = MockVLLM(response_fn=resp)
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)
        runner = SimulationRunner(config=config, llm_pipeline=pipeline)
        runner.initialize()
        runner._step(0)
        assert len(runner.broadcast_history) == 0
        snapshot = runner.metrics_logger.epoch_snapshots[0]
        assert snapshot["broadcast_count"] == 0


class TestSimulationRunnerRunAndExport:

    def test_run_flushes_epochs_and_exports(self, tmp_path):
        config = RunConfig(num_agents=8, grid_size=25, num_epochs=4, output_dir=str(tmp_path), seed=100)
        mock_vllm = MockVLLM()
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)
        runner = SimulationRunner(config=config, llm_pipeline=pipeline)
        metrics = runner.run()

        assert len(runner.epoch_timings) == 4
        for ep in range(4):
            ep_file = tmp_path / f"epoch_{ep:04d}.json"
            assert ep_file.exists()
            with open(ep_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert data["epoch"] == ep
                assert data["agent_count"] == 8
                assert "epoch_duration_s" in data

        summary = runner.export_results()
        assert (tmp_path / "epoch_metrics.json").exists()
        assert (tmp_path / "epoch_metrics.csv").exists()
        assert (tmp_path / "agent_epoch_details.csv").exists()
        assert (tmp_path / "summary.json").exists()
        assert summary["condition"] == "delta2"
        assert summary["broadcast_enabled"] is True
        assert summary["total_epochs"] == 4
        assert runner.validate_state() is True


class TestSimulationRunnerFullScale:

    def test_50_agents_100_epochs_mock_simulation(self, tmp_path):
        config = RunConfig(
            num_agents=50,
            grid_size=50,
            num_epochs=100,
            seed=42,
            output_dir=str(tmp_path),
            broadcast_enabled=True,
        )

        def mixed_response(prompt):
            r = random.random()
            if r < 0.4:
                return '{"move": [25, 25], "broadcast": "Going to bar!", "proximity_speech": null, "stated_intention": "going", "actual_target": "bar"}'
            elif r < 0.7:
                return '{"move": [5, 5], "broadcast": "Going to bar!", "proximity_speech": null, "stated_intention": "going", "actual_target": "home"}'
            else:
                return '{"move": [45, 45], "broadcast": "Staying home", "proximity_speech": null, "stated_intention": "staying", "actual_target": "home"}'

        mock_vllm = MockVLLM(response_fn=mixed_response)
        pipeline = VLLMBatchPipeline(mock_vllm, sampling_params=None)
        runner = SimulationRunner(config=config, llm_pipeline=pipeline)
        runner.run()

        assert runner.validate_state() is True
        summary = runner.export_results()
        assert summary["total_epochs"] == 100
        assert len(runner.agent_pool.all_agents()) == 50
        assert 0.0 <= summary["population_truthfulness"] <= 1.0
        assert 0.0 <= summary["average_deception_index"] <= 1.0
        assert len(runner.metrics_logger.epoch_snapshots) == 100
