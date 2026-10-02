import json
import os
import random
import numpy as np
import pytest
from elfarol.engine import BrainSimulationEngine
from elfarol.llm import MockLLMPipeline


class TestBrainSimulationEngineInit:

    def test_creates_grid_and_pool(self):
        mock = MockLLMPipeline()
        engine = BrainSimulationEngine(50, 50, mock, 10, seed=42)
        assert engine.grid.size == 50
        assert engine.num_agents == 50

    def test_metrics_logger_initialized(self):
        mock = MockLLMPipeline()
        engine = BrainSimulationEngine(50, 50, mock, 10, seed=42)
        assert engine.metrics_logger is not None
        assert engine.broadcast_history == []


class TestBrainSimulationEngineRun:

    def _make_brain_response_fn(self, rng):
        def fn(idx, prompt):
            x = rng.randint(0, 49)
            y = rng.randint(0, 49)
            going = rng.choice([True, False])
            deceptive = rng.random() < 0.3
            if going:
                stated = "staying" if deceptive else "going"
                target = "bar"
                move = [rng.randint(20, 29), rng.randint(20, 29)]
            else:
                stated = "going" if deceptive else "staying"
                target = "home"
                move = [rng.randint(0, 19), rng.randint(0, 19)]
            return {
                "move": move,
                "broadcast": f"Agent {idx}: I am {stated}" if rng.random() > 0.5 else None,
                "proximity_speech": f"psst from {idx}" if rng.random() > 0.7 else None,
                "stated_intention": stated,
                "actual_target": target,
            }
        return fn

    def test_50_agents_100_epochs_no_collisions(self):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 100, seed=42)
        engine.initialize()

        for epoch in range(100):
            engine._tick(epoch)
            assert engine.validate_state(), f"State corruption at epoch {epoch}"

        agents = engine.agent_pool.all_agents()
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))
        assert int(np.count_nonzero(engine.grid.grid)) == 50

    def test_metrics_logger_populated(self):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 100, seed=42)
        logger = engine.run()
        assert len(logger.epoch_snapshots) == 100

    def test_deception_index_tracked(self):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 100, seed=42)
        logger = engine.run()
        for snap in logger.epoch_snapshots:
            assert 0.0 <= snap["deception_index"] <= 1.0
            assert 0.0 <= snap["truthfulness_ratio"] <= 1.0
            assert snap["deception_count"] + snap["truthful_count"] == 50

    def test_utility_history_accumulated(self):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 100, seed=42)
        engine.run()
        for agent in engine.agent_pool.all_agents():
            assert len(agent.utility_history) == 100

    def test_bar_attendance_vs_threshold(self):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 50, seed=42)
        logger = engine.run()
        for snap in logger.epoch_snapshots:
            assert snap["comfort_threshold"] == 60
            assert isinstance(snap["attendance_over_threshold"], bool)

    def test_broadcast_history_grows(self):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 10, seed=42)
        engine.run()
        assert len(engine.broadcast_history) > 0

    def test_summary_after_run(self):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 50, seed=42)
        logger = engine.run()
        s = logger.summary()
        assert s["total_epochs"] == 50
        assert 0.0 <= s["average_deception_index"] <= 1.0
        assert 0.0 <= s["population_truthfulness"] <= 1.0

    def test_export_json(self, tmp_path):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 10, seed=42)
        logger = engine.run()
        filepath = str(tmp_path / "metrics.json")
        logger.export_json(filepath)
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        assert len(data) == 10

    def test_export_csv(self, tmp_path):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 10, seed=42)
        logger = engine.run()
        filepath = str(tmp_path / "metrics.csv")
        logger.export_csv(filepath)
        assert os.path.exists(filepath)

    def test_export_agent_csv(self, tmp_path):
        rng = random.Random(12345)
        mock = MockLLMPipeline(response_fn=self._make_brain_response_fn(rng))
        engine = BrainSimulationEngine(50, 50, mock, 10, seed=42)
        logger = engine.run()
        filepath = str(tmp_path / "agent.csv")
        logger.export_agent_csv(filepath)
        assert os.path.exists(filepath)


class TestBrainSimulationAllGoingToBar:

    def test_all_agents_target_bar(self):
        def bar_response(idx, prompt):
            return {
                "move": [24, 24],
                "broadcast": "I am going!",
                "proximity_speech": None,
                "stated_intention": "going",
                "actual_target": "bar",
            }

        mock = MockLLMPipeline(response_fn=bar_response)
        engine = BrainSimulationEngine(50, 50, mock, 50, seed=42)
        engine.initialize()

        for epoch in range(50):
            engine._tick(epoch)
            assert engine.validate_state()

        agents = engine.agent_pool.all_agents()
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))

        at_target = sum(1 for a in agents if a.x == 24 and a.y == 24)
        assert at_target == 1


class TestBrainSimulationAllDeceptive:

    def test_full_deception_detected(self):
        def deceptive_response(idx, prompt):
            return {
                "move": [0, 0],
                "broadcast": "I am going to the bar!",
                "proximity_speech": None,
                "stated_intention": "going",
                "actual_target": "home",
            }

        mock = MockLLMPipeline(response_fn=deceptive_response)
        engine = BrainSimulationEngine(50, 50, mock, 10, seed=42)
        engine.initialize()

        for epoch in range(10):
            engine._tick(epoch)
            assert engine.validate_state()

        logger = engine.metrics_logger
        for snap in logger.epoch_snapshots:
            assert snap["deception_count"] > 0


class TestBrainSimulationDeterministic:

    def test_same_seed_same_result(self):
        def fixed_response(idx, prompt):
            return {
                "move": [idx % 50, (idx * 3) % 50],
                "broadcast": None,
                "proximity_speech": None,
                "stated_intention": "going" if idx % 2 == 0 else "staying",
                "actual_target": "bar" if idx % 2 == 0 else "home",
            }

        mock1 = MockLLMPipeline(response_fn=fixed_response)
        engine1 = BrainSimulationEngine(50, 50, mock1, 10, seed=42)
        logger1 = engine1.run()

        mock2 = MockLLMPipeline(response_fn=fixed_response)
        engine2 = BrainSimulationEngine(50, 50, mock2, 10, seed=42)
        logger2 = engine2.run()

        pos1 = [(a.x, a.y) for a in engine1.agent_pool.all_agents()]
        pos2 = [(a.x, a.y) for a in engine2.agent_pool.all_agents()]
        assert pos1 == pos2

        for s1, s2 in zip(logger1.epoch_snapshots, logger2.epoch_snapshots):
            assert s1["deception_index"] == s2["deception_index"]
            assert s1["bar_attendance"] == s2["bar_attendance"]
