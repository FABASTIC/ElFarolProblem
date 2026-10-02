import json
import random
import pytest
from elfarol.grid import SpatialGrid
from elfarol.agent import Agent, AgentPool
from elfarol.agent_brain import (
    SYSTEM_PROMPT,
    RESPONSE_SCHEMA_INSTRUCTION,
    VALID_INTENTIONS,
    VALID_TARGETS,
    build_agent_context,
    build_brain_prompt,
    build_brain_batch_prompts,
    validate_brain_output,
    neutral_action,
    validate_or_fallback,
    process_brain_batch,
)


class TestSystemPrompt:

    def test_contains_utility_maximizer(self):
        assert "utility maximizer" in SYSTEM_PROMPT

    def test_contains_deception_permission(self):
        assert "deception" in SYSTEM_PROMPT.lower()

    def test_contains_bar_reference(self):
        assert "Bar" in SYSTEM_PROMPT

    def test_contains_broadcast_channel(self):
        assert "broadcast" in SYSTEM_PROMPT.lower()


class TestResponseSchemaInstruction:

    def test_contains_move(self):
        assert '"move"' in RESPONSE_SCHEMA_INSTRUCTION

    def test_contains_broadcast(self):
        assert '"broadcast"' in RESPONSE_SCHEMA_INSTRUCTION

    def test_contains_proximity_speech(self):
        assert '"proximity_speech"' in RESPONSE_SCHEMA_INSTRUCTION

    def test_contains_stated_intention(self):
        assert '"stated_intention"' in RESPONSE_SCHEMA_INSTRUCTION

    def test_contains_actual_target(self):
        assert '"actual_target"' in RESPONSE_SCHEMA_INSTRUCTION


class TestBuildAgentContext:

    def setup_method(self):
        random.seed(42)
        self.grid = SpatialGrid(50)
        self.pool = AgentPool()
        self.pool.spawn(10, self.grid)

    def test_context_has_epoch(self):
        agent = self.pool.get_agent(1)
        ctx = build_agent_context(agent, self.grid, self.pool, [], 5)
        assert ctx["epoch"] == 5

    def test_context_has_agent_id(self):
        agent = self.pool.get_agent(3)
        ctx = build_agent_context(agent, self.grid, self.pool, [], 0)
        assert ctx["agent_id"] == 3

    def test_context_has_position(self):
        agent = self.pool.get_agent(1)
        ctx = build_agent_context(agent, self.grid, self.pool, [], 0)
        assert ctx["position"]["x"] == agent.x
        assert ctx["position"]["y"] == agent.y

    def test_context_has_bar_zone(self):
        agent = self.pool.get_agent(1)
        ctx = build_agent_context(agent, self.grid, self.pool, [], 0)
        assert ctx["bar_zone"]["min"] == 20
        assert ctx["bar_zone"]["max"] == 29
        assert ctx["bar_zone"]["capacity"] == 100

    def test_context_has_comfort_threshold(self):
        agent = self.pool.get_agent(1)
        ctx = build_agent_context(agent, self.grid, self.pool, [], 0)
        assert ctx["comfort_threshold"] == 60

    def test_context_utility_history_empty(self):
        agent = self.pool.get_agent(1)
        ctx = build_agent_context(agent, self.grid, self.pool, [], 0)
        assert ctx["utility_history_last_5"] == []

    def test_context_utility_history_capped_at_5(self):
        agent = self.pool.get_agent(1)
        agent.utility_history = [0.3, 1.0, -1.0, 0.3, 1.0, -1.0, 0.3]
        ctx = build_agent_context(agent, self.grid, self.pool, [], 0)
        assert len(ctx["utility_history_last_5"]) == 5
        assert ctx["utility_history_last_5"] == [-1.0, 0.3, 1.0, -1.0, 0.3]

    def test_context_broadcasts_capped_at_10(self):
        agent = self.pool.get_agent(1)
        history = [f"msg_{i}" for i in range(15)]
        ctx = build_agent_context(agent, self.grid, self.pool, history, 0)
        assert len(ctx["recent_broadcasts"]) == 10

    def test_context_nearby_agents(self):
        agent = self.pool.get_agent(1)
        ctx = build_agent_context(agent, self.grid, self.pool, [], 0)
        assert isinstance(ctx["nearby_agents"], list)
        for n in ctx["nearby_agents"]:
            assert "id" in n
            assert "x" in n
            assert "y" in n


class TestBuildBrainPrompt:

    def setup_method(self):
        random.seed(42)
        self.grid = SpatialGrid(50)
        self.pool = AgentPool()
        self.pool.spawn(5, self.grid)

    def test_prompt_contains_system_prompt(self):
        agent = self.pool.get_agent(1)
        prompt = build_brain_prompt(agent, self.grid, self.pool, [], 0)
        assert "utility maximizer" in prompt

    def test_prompt_contains_schema_instruction(self):
        agent = self.pool.get_agent(1)
        prompt = build_brain_prompt(agent, self.grid, self.pool, [], 0)
        assert '"stated_intention"' in prompt

    def test_prompt_contains_agent_position(self):
        agent = self.pool.get_agent(1)
        prompt = build_brain_prompt(agent, self.grid, self.pool, [], 0)
        assert f"Agent {agent.id}" in prompt
        assert f"({agent.x}, {agent.y})" in prompt

    def test_prompt_contains_bar_info(self):
        agent = self.pool.get_agent(1)
        prompt = build_brain_prompt(agent, self.grid, self.pool, [], 0)
        assert "capacity 100" in prompt

    def test_prompt_contains_comfort_threshold(self):
        agent = self.pool.get_agent(1)
        prompt = build_brain_prompt(agent, self.grid, self.pool, [], 0)
        assert "Comfort threshold: 60" in prompt

    def test_prompt_includes_broadcasts(self):
        agent = self.pool.get_agent(1)
        history = ["Agent 2: I am going!", "Agent 3: staying home"]
        prompt = build_brain_prompt(agent, self.grid, self.pool, history, 5)
        assert "I am going!" in prompt
        assert "staying home" in prompt

    def test_prompt_includes_utility_history(self):
        agent = self.pool.get_agent(1)
        agent.utility_history = [0.3, 1.0, -1.0]
        prompt = build_brain_prompt(agent, self.grid, self.pool, [], 0)
        assert "0.3" in prompt
        assert "1.0" in prompt


class TestBuildBrainBatchPrompts:

    def test_batch_length_matches_agents(self):
        random.seed(42)
        grid = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(20, grid)
        prompts = build_brain_batch_prompts(agents, grid, pool, [], 0)
        assert len(prompts) == 20

    def test_each_prompt_is_string(self):
        random.seed(42)
        grid = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(5, grid)
        prompts = build_brain_batch_prompts(agents, grid, pool, [], 0)
        for p in prompts:
            assert isinstance(p, str)
            assert len(p) > 100


class TestValidateBrainOutput:

    def test_valid_complete_output(self):
        raw = {
            "move": [25, 25],
            "broadcast": "going to bar!",
            "proximity_speech": "hey neighbor",
            "stated_intention": "going",
            "actual_target": "bar",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result is not None
        assert result["move"] == [25, 25]
        assert result["broadcast"] == "going to bar!"
        assert result["proximity_speech"] == "hey neighbor"
        assert result["stated_intention"] == "going"
        assert result["actual_target"] == "bar"

    def test_valid_with_nulls(self):
        raw = {
            "move": [10, 10],
            "broadcast": None,
            "proximity_speech": None,
            "stated_intention": "staying",
            "actual_target": "home",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result is not None
        assert result["broadcast"] is None
        assert result["proximity_speech"] is None

    def test_move_clamped_to_bounds(self):
        raw = {
            "move": [100, -5],
            "broadcast": None,
            "proximity_speech": None,
            "stated_intention": "staying",
            "actual_target": "home",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result["move"] == [49, 0]

    def test_missing_move_returns_none(self):
        raw = {
            "broadcast": "hi",
            "stated_intention": "going",
            "actual_target": "bar",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result is None

    def test_invalid_stated_intention_defaults(self):
        raw = {
            "move": [5, 5],
            "stated_intention": "maybe",
            "actual_target": "bar",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result["stated_intention"] == "staying"

    def test_invalid_actual_target_defaults(self):
        raw = {
            "move": [5, 5],
            "stated_intention": "going",
            "actual_target": "park",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result["actual_target"] == "home"

    def test_missing_broadcast_defaults_to_none(self):
        raw = {
            "move": [5, 5],
            "stated_intention": "going",
            "actual_target": "bar",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result["broadcast"] is None

    def test_non_string_broadcast_defaults_to_none(self):
        raw = {
            "move": [5, 5],
            "broadcast": 123,
            "stated_intention": "going",
            "actual_target": "bar",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result["broadcast"] is None

    def test_json_string_input(self):
        raw = json.dumps({
            "move": [20, 20],
            "broadcast": "hello",
            "proximity_speech": "psst",
            "stated_intention": "going",
            "actual_target": "bar",
        })
        result = validate_brain_output(raw, grid_size=50)
        assert result is not None
        assert result["move"] == [20, 20]

    def test_json_string_with_prefix(self):
        raw = 'Here is my action: {"move": [15, 15], "broadcast": null, "proximity_speech": null, "stated_intention": "staying", "actual_target": "home"} done'
        result = validate_brain_output(raw, grid_size=50)
        assert result is not None
        assert result["move"] == [15, 15]

    def test_garbage_string_returns_none(self):
        result = validate_brain_output("not valid at all", grid_size=50)
        assert result is None

    def test_empty_string_returns_none(self):
        result = validate_brain_output("", grid_size=50)
        assert result is None

    def test_none_input_returns_none(self):
        result = validate_brain_output(None, grid_size=50)
        assert result is None

    def test_float_coords_truncated(self):
        raw = {
            "move": [10.7, 20.3],
            "stated_intention": "going",
            "actual_target": "bar",
        }
        result = validate_brain_output(raw, grid_size=50)
        assert result["move"] == [10, 20]


class TestNeutralAction:

    def test_neutral_action_structure(self):
        action = neutral_action(10, 20)
        assert action["move"] == [10, 20]
        assert action["broadcast"] is None
        assert action["proximity_speech"] is None
        assert action["stated_intention"] == "staying"
        assert action["actual_target"] == "home"


class TestValidateOrFallback:

    def test_valid_input_passes_through(self):
        raw = {
            "move": [5, 5],
            "broadcast": "hi",
            "proximity_speech": None,
            "stated_intention": "going",
            "actual_target": "bar",
        }
        result = validate_or_fallback(raw, 10, 10, grid_size=50)
        assert result["move"] == [5, 5]
        assert result["stated_intention"] == "going"

    def test_invalid_input_returns_neutral(self):
        result = validate_or_fallback("garbage", 10, 20, grid_size=50)
        assert result["move"] == [10, 20]
        assert result["stated_intention"] == "staying"
        assert result["actual_target"] == "home"


class TestProcessBrainBatch:

    def test_batch_all_valid_dicts(self):
        agents = [Agent(id=i, x=i, y=i) for i in range(3)]
        raw_outputs = [
            {"move": [5, 5], "broadcast": None, "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"},
            {"move": [10, 10], "broadcast": "hi", "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"},
            {"move": [15, 15], "broadcast": None, "proximity_speech": "psst", "stated_intention": "going", "actual_target": "bar"},
        ]
        results = process_brain_batch(raw_outputs, agents, grid_size=50)
        assert len(results) == 3
        assert results[0]["move"] == [5, 5]
        assert results[1]["broadcast"] == "hi"
        assert results[2]["proximity_speech"] == "psst"

    def test_batch_with_invalid_falls_back(self):
        agents = [Agent(id=1, x=7, y=8), Agent(id=2, x=3, y=4)]
        raw_outputs = [
            {"move": [5, 5], "stated_intention": "going", "actual_target": "bar"},
            "totally broken garbage",
        ]
        results = process_brain_batch(raw_outputs, agents, grid_size=50)
        assert len(results) == 2
        assert results[0]["move"] == [5, 5]
        assert results[1]["move"] == [3, 4]
        assert results[1]["stated_intention"] == "staying"

    def test_batch_with_none_falls_back(self):
        agents = [Agent(id=1, x=0, y=0)]
        raw_outputs = [None]
        results = process_brain_batch(raw_outputs, agents, grid_size=50)
        assert results[0]["move"] == [0, 0]
        assert results[0]["actual_target"] == "home"

    def test_batch_with_json_strings(self):
        agents = [Agent(id=1, x=0, y=0)]
        raw_outputs = [
            json.dumps({"move": [25, 25], "broadcast": "hello", "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"})
        ]
        results = process_brain_batch(raw_outputs, agents, grid_size=50)
        assert results[0]["move"] == [25, 25]
        assert results[0]["stated_intention"] == "going"

    def test_batch_50_agents(self):
        agents = [Agent(id=i, x=i % 50, y=i % 50) for i in range(50)]
        raw_outputs = [
            {"move": [i % 50, (i * 3) % 50], "broadcast": None, "proximity_speech": None, "stated_intention": "going" if i % 2 == 0 else "staying", "actual_target": "bar" if i % 2 == 0 else "home"}
            for i in range(50)
        ]
        results = process_brain_batch(raw_outputs, agents, grid_size=50)
        assert len(results) == 50
        for r in results:
            assert r["stated_intention"] in VALID_INTENTIONS
            assert r["actual_target"] in VALID_TARGETS
