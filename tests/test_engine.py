import random
import numpy as np
import pytest
from elfarol.engine import SimulationEngine
from elfarol.llm import MockLLMPipeline


class TestSimulationEngineInit:

    def test_grid_created(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 10, seed=42)
        assert engine.grid.size == 50

    def test_metrics_initialized(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 10, seed=42)
        assert engine.metrics["bar_occupancy"] == []
        assert engine.metrics["collision_attempts"] == []
        assert engine.metrics["message_counts"] == []


class TestSimulationEngineInitialize:

    def test_agents_spawned(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 10, seed=42)
        engine.initialize()
        assert len(engine.agent_pool.all_agents()) == 50

    def test_all_agents_on_grid(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 10, seed=42)
        engine.initialize()
        for a in engine.agent_pool.all_agents():
            assert engine.grid.grid[a.y, a.x] == a.id

    def test_no_initial_collisions(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 10, seed=42)
        engine.initialize()
        positions = [(a.x, a.y) for a in engine.agent_pool.all_agents()]
        assert len(positions) == len(set(positions))


class TestSimulationEngineValidateState:

    def test_valid_after_init(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 10, seed=42)
        engine.initialize()
        assert engine.validate_state() is True


class TestSimulationEngineRun:

    def test_run_completes(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 100, seed=42)
        metrics = engine.run()
        assert len(metrics["bar_occupancy"]) == 100
        assert engine.validate_state() is True

    def test_run_with_stationary_agents(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 100, seed=42)
        engine.run()
        assert engine.validate_state() is True
        positions = [(a.x, a.y) for a in engine.agent_pool.all_agents()]
        assert len(positions) == len(set(positions))


class TestFullSimulation:

    def test_50_agents_100_epochs_random_moves_no_collisions(self):
        move_rng = random.Random(12345)

        def random_response(idx, prompt):
            x = move_rng.randint(0, 49)
            y = move_rng.randint(0, 49)
            return {"move": [x, y], "speak": f"moving to {x},{y}"}

        mock_llm = MockLLMPipeline(response_fn=random_response)
        engine = SimulationEngine(
            num_agents=50,
            grid_size=50,
            llm_pipeline=mock_llm,
            epochs=100,
            seed=42,
        )
        engine.initialize()

        for epoch in range(100):
            engine._tick(epoch)
            assert engine.validate_state(), f"State corruption at epoch {epoch}"

        agents = engine.agent_pool.all_agents()
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))
        assert len(engine.metrics["bar_occupancy"]) == 100

        for a in agents:
            assert 0 <= a.x < 50
            assert 0 <= a.y < 50
            assert engine.grid.grid[a.y, a.x] == a.id

        total_nonzero = int(np.count_nonzero(engine.grid.grid))
        assert total_nonzero == 50

    def test_50_agents_100_epochs_aggressive_bar_moves(self):
        def bar_response(idx, prompt):
            return {"move": [24, 24], "speak": "going to bar"}

        mock_llm = MockLLMPipeline(response_fn=bar_response)
        engine = SimulationEngine(
            num_agents=50,
            grid_size=50,
            llm_pipeline=mock_llm,
            epochs=100,
            seed=42,
        )
        engine.initialize()

        for epoch in range(100):
            engine._tick(epoch)
            assert engine.validate_state(), f"State corruption at epoch {epoch}"

        agents = engine.agent_pool.all_agents()
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))

        bar_count = sum(1 for a in agents if a.x == 24 and a.y == 24)
        assert bar_count == 1

    def test_50_agents_100_epochs_mixed_moves_and_speech(self):
        move_rng = random.Random(54321)

        def mixed_response(idx, prompt):
            x = move_rng.randint(0, 49)
            y = move_rng.randint(0, 49)
            speak = f"agent {idx} says hello" if move_rng.random() > 0.5 else ""
            return {"move": [x, y], "speak": speak}

        mock_llm = MockLLMPipeline(response_fn=mixed_response)
        engine = SimulationEngine(
            num_agents=50,
            grid_size=50,
            llm_pipeline=mock_llm,
            epochs=100,
            seed=42,
        )
        engine.initialize()

        for epoch in range(100):
            engine._tick(epoch)
            assert engine.validate_state(), f"State corruption at epoch {epoch}"

        agents = engine.agent_pool.all_agents()
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))
        assert int(np.count_nonzero(engine.grid.grid)) == 50

    def test_deterministic_with_same_seed(self):
        def fixed_response(idx, prompt):
            return {"move": [idx % 50, (idx * 7) % 50], "speak": ""}

        mock1 = MockLLMPipeline(response_fn=fixed_response)
        engine1 = SimulationEngine(50, 50, mock1, 10, seed=42)
        engine1.run()
        positions1 = [(a.x, a.y) for a in engine1.agent_pool.all_agents()]

        mock2 = MockLLMPipeline(response_fn=fixed_response)
        engine2 = SimulationEngine(50, 50, mock2, 10, seed=42)
        engine2.run()
        positions2 = [(a.x, a.y) for a in engine2.agent_pool.all_agents()]

        assert positions1 == positions2

    def test_metrics_populated(self):
        mock = MockLLMPipeline()
        engine = SimulationEngine(50, 50, mock, 100, seed=42)
        metrics = engine.run()
        assert len(metrics["bar_occupancy"]) == 100
        assert len(metrics["collision_attempts"]) == 100
        assert len(metrics["message_counts"]) == 100
        for occ in metrics["bar_occupancy"]:
            assert 0 <= occ <= 100
