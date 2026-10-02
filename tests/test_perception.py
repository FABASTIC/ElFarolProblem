import random
import pytest
from elfarol.grid import SpatialGrid
from elfarol.agent import Agent, AgentPool
from elfarol.perception import build_prompt, build_batch_prompts


class TestBuildPrompt:

    def setup_method(self):
        random.seed(42)
        self.grid = SpatialGrid(50)
        self.pool = AgentPool()
        self.pool.spawn(5, self.grid)

    def test_contains_agent_id(self):
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert f"Agent {agent.id}" in prompt

    def test_contains_position(self):
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert f"({agent.x}, {agent.y})" in prompt

    def test_contains_grid_size(self):
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert "50x50" in prompt

    def test_contains_bar_zone(self):
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert "20" in prompt
        assert "29" in prompt

    def test_contains_bar_status(self):
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert "inside" in prompt or "outside" in prompt

    def test_contains_occupancy(self):
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert "/100 seats" in prompt

    def test_contains_json_instruction(self):
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert '"move"' in prompt
        assert '"speak"' in prompt

    def test_contains_global_broadcasts(self):
        self.pool.broadcast("global test")
        agent = self.pool.get_agent(1)
        prompt = build_prompt(agent, self.grid, self.pool)
        assert "global test" in prompt


class TestBuildBatchPrompts:

    def test_length_matches_agents(self):
        random.seed(42)
        grid = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(20, grid)
        prompts = build_batch_prompts(agents, grid, pool)
        assert len(prompts) == 20

    def test_each_prompt_is_string(self):
        random.seed(42)
        grid = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(5, grid)
        prompts = build_batch_prompts(agents, grid, pool)
        for p in prompts:
            assert isinstance(p, str)
            assert len(p) > 0

    def test_prompts_contain_unique_agent_ids(self):
        random.seed(42)
        grid = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(10, grid)
        prompts = build_batch_prompts(agents, grid, pool)
        for i, prompt in enumerate(prompts):
            assert f"Agent {agents[i].id}" in prompt
