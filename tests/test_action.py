import random
import pytest
import numpy as np
from elfarol.grid import SpatialGrid
from elfarol.agent import Agent, AgentPool
from elfarol.action import validate_action, execute_actions


class TestValidateAction:

    def test_valid_action(self):
        result = validate_action({"move": [10, 20], "speak": "hello"}, grid_size=50)
        assert result["move"] == [10, 20]
        assert result["speak"] == "hello"

    def test_clamp_high(self):
        result = validate_action({"move": [100, 200], "speak": ""}, grid_size=50)
        assert result["move"] == [49, 49]

    def test_clamp_low(self):
        result = validate_action({"move": [-5, -10], "speak": ""}, grid_size=50)
        assert result["move"] == [0, 0]

    def test_missing_move(self):
        result = validate_action({"speak": "hello"}, grid_size=50)
        assert result["move"] == [0, 0]

    def test_missing_speak(self):
        result = validate_action({"move": [5, 5]}, grid_size=50)
        assert result["speak"] == ""

    def test_non_dict_input(self):
        result = validate_action("garbage", grid_size=50)
        assert result == {"move": [0, 0], "speak": ""}

    def test_none_input(self):
        result = validate_action(None, grid_size=50)
        assert result == {"move": [0, 0], "speak": ""}

    def test_bad_move_type(self):
        result = validate_action({"move": "bad", "speak": "hi"}, grid_size=50)
        assert result["move"] == [0, 0]

    def test_bad_speak_type(self):
        result = validate_action({"move": [5, 5], "speak": 123}, grid_size=50)
        assert result["speak"] == ""


class TestExecuteActions:

    def _setup_agents(self, positions):
        grid = SpatialGrid(50)
        pool = AgentPool()
        agents = []
        for i, (x, y) in enumerate(positions):
            agent_id = i + 1
            a = Agent(id=agent_id, x=x, y=y)
            grid.place(agent_id, x, y)
            pool.agents[agent_id] = a
            agents.append(a)
        return grid, pool, agents

    def test_simple_move(self):
        grid, pool, agents = self._setup_agents([(0, 0)])
        actions = [{"move": [5, 5], "speak": ""}]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        assert agents[0].x == 5
        assert agents[0].y == 5
        assert grid.grid[5, 5] == 1
        assert grid.grid[0, 0] == 0

    def test_collision_prevented(self):
        grid, pool, agents = self._setup_agents([(0, 0), (10, 10)])
        actions = [
            {"move": [10, 10], "speak": ""},
            {"move": [10, 10], "speak": ""},
        ]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))

    def test_no_move_when_target_occupied(self):
        grid, pool, agents = self._setup_agents([(0, 0), (5, 5)])
        actions = [
            {"move": [5, 5], "speak": ""},
            {"move": [5, 5], "speak": ""},
        ]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        assert grid.grid[5, 5] != 0
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))

    def test_speak_adds_local_message(self):
        grid, pool, agents = self._setup_agents([(0, 0)])
        actions = [{"move": [0, 0], "speak": "hello world"}]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        assert "hello world" in agents[0].local_messages

    def test_speak_broadcasts_globally(self):
        grid, pool, agents = self._setup_agents([(0, 0), (49, 49)])
        actions = [
            {"move": [0, 0], "speak": "shout"},
            {"move": [49, 49], "speak": ""},
        ]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        assert any("shout" in m for m in agents[1].global_messages)

    def test_many_agents_no_collisions(self):
        rng = random.Random(99)
        positions = random.Random(99).sample(
            [(x, y) for x in range(50) for y in range(50)], 50
        )
        grid, pool, agents = self._setup_agents(positions)
        actions = [
            {"move": [rng.randint(0, 49), rng.randint(0, 49)], "speak": ""}
            for _ in agents
        ]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        final_positions = [(a.x, a.y) for a in agents]
        assert len(final_positions) == len(set(final_positions))
        for a in agents:
            assert grid.grid[a.y, a.x] == a.id

    def test_stay_in_place(self):
        grid, pool, agents = self._setup_agents([(10, 10)])
        actions = [{"move": [10, 10], "speak": ""}]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        assert agents[0].x == 10
        assert agents[0].y == 10
        assert grid.grid[10, 10] == 1

    def test_grid_consistency_after_execution(self):
        rng = random.Random(77)
        positions = random.Random(77).sample(
            [(x, y) for x in range(50) for y in range(50)], 30
        )
        grid, pool, agents = self._setup_agents(positions)
        actions = [
            {"move": [rng.randint(0, 49), rng.randint(0, 49)], "speak": ""}
            for _ in agents
        ]
        execute_actions(agents, actions, grid, pool, rng=random.Random(42))
        agent_cells = 0
        for a in agents:
            assert grid.grid[a.y, a.x] == a.id
            agent_cells += 1
        assert int(np.count_nonzero(grid.grid)) == agent_cells
