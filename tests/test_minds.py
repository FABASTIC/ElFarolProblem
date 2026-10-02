import csv
import json
import random

import pytest

from elfarol.agent import Agent
from elfarol.agent_brain import RESPONSE_SCHEMA_INSTRUCTION, SYSTEM_PROMPT
from elfarol.grid import SpatialGrid
from elfarol.minds import (
    MINDS_FILENAME,
    NOTE_INSTRUCTION,
    OPTIONS,
    TRACE_FIELDS,
    TRACE_FILENAME,
    MindPopulation,
    MindRunner,
    resolve_targets,
    spawn_mind,
)
from elfarol.simulation_runner import RunConfig, VLLMBatchPipeline


class _Choice:

    def __init__(self, text):
        self.text = text


class _Output:

    def __init__(self, text):
        self.outputs = [_Choice(text)]


class ScriptedMinds:

    def __init__(self, lie_every=3):
        self.lie_every = lie_every
        self.calls = 0

    def generate(self, prompts, sampling_params=None, **kwargs):
        self.calls += 1
        outputs = []
        for i, prompt in enumerate(prompts):
            go = (i + self.calls) % 2 == 0
            lie = i % self.lie_every == 0
            stated = ("staying" if go else "going") if lie else ("going" if go else "staying")
            outputs.append(_Output(json.dumps({
                "move": [0, 0],
                "broadcast": f"I am {stated}",
                "proximity_speech": None,
                "stated_intention": stated,
                "actual_target": "bar" if go else "home",
                "private_note": "bluffing" if lie else "honest",
            })))
        return outputs


def _grid_with_agents(positions, size=20):
    grid = SpatialGrid(size)
    agents = []
    for i, (x, y) in enumerate(positions, start=1):
        grid.place(i, x, y)
        agents.append(Agent(id=i, x=x, y=y))
    return grid, agents


class TestMindIdentity:

    def test_same_seed_and_agent_yield_identical_mind(self):
        assert spawn_mind(42, 7, 50).traits == spawn_mind(42, 7, 50).traits

    def test_agents_differ_within_a_population(self):
        traits = {tuple(sorted(spawn_mind(42, aid, 50).traits.items())) for aid in range(1, 21)}
        assert len(traits) == 20

    def test_population_is_paired_across_conditions(self):
        control = MindPopulation(42, range(1, 11), 10, 60, broadcast_enabled=False)
        treatment = MindPopulation(42, range(1, 11), 10, 60, broadcast_enabled=True)
        for aid in range(1, 11):
            assert control.minds[aid].traits == treatment.minds[aid].traits
            assert control.minds[aid].archetype == treatment.minds[aid].archetype

    def test_disposition_is_a_distribution(self):
        population = MindPopulation(1, range(1, 6), 5, 60, broadcast_enabled=True)
        population.deliberate(range(1, 6))
        for mind in population.minds.values():
            assert set(mind.disposition) == set(OPTIONS)
            assert sum(mind.disposition.values()) == pytest.approx(1.0)


class TestTargetResolver:

    def test_bar_intents_get_distinct_free_cells(self):
        grid, agents = _grid_with_agents([(0, i) for i in range(6)])
        actions = [{"move": [grid.bar_min, grid.bar_min], "actual_target": "bar", "stated_intention": "going"} for _ in agents]
        resolved, rerouted = resolve_targets(agents, actions, grid, random.Random(0))
        cells = [tuple(a["move"]) for a in resolved]
        assert len(set(cells)) == len(cells)
        assert all(grid.is_in_bar(*c) for c in cells)
        assert sum(rerouted) >= len(cells) - 1

    def test_home_intent_leaves_the_bar(self):
        grid, agents = _grid_with_agents([(grid_cell, grid_cell) for grid_cell in (6, 7)])
        actions = [{"move": [6, 6], "actual_target": "home"}, {"move": [7, 7], "actual_target": "home"}]
        resolved, _ = resolve_targets(agents, actions, grid, random.Random(0))
        assert all(not grid.is_in_bar(*a["move"]) for a in resolved)

    def test_valid_choice_is_respected(self):
        grid, agents = _grid_with_agents([(0, 0)])
        target = [grid.bar_min + 3, grid.bar_min + 4]
        resolved, rerouted = resolve_targets(agents, [{"move": target, "actual_target": "bar"}], grid, random.Random(0))
        assert resolved[0]["move"] == target
        assert rerouted == [False]


class TestPromptRendering:

    def test_private_block_sits_before_schema(self):
        population = MindPopulation(3, [1, 2], 2, 60, broadcast_enabled=True)
        population.deliberate([1, 2])
        prompt = SYSTEM_PROMPT + "\n\nEpoch: 0\n\n" + RESPONSE_SCHEMA_INSTRUCTION
        rendered = population.render(1, prompt, 0)
        assert rendered.startswith(SYSTEM_PROMPT)
        assert rendered.index("YOUR MIND") < rendered.index(RESPONSE_SCHEMA_INSTRUCTION)
        assert rendered.endswith(NOTE_INSTRUCTION)
        assert "Who you trust" in rendered


class TestMindRunner:

    @pytest.fixture
    def finished(self, tmp_path):
        config = RunConfig(num_agents=12, grid_size=20, num_epochs=6, seed=5, broadcast_enabled=True, output_dir=str(tmp_path), condition_label="delta2")
        runner = MindRunner(config, VLLMBatchPipeline(ScriptedMinds(), None))
        runner.run()
        summary = runner.export_results()
        return runner, summary, tmp_path

    def test_state_stays_valid_and_routing_lands_intent(self, finished):
        runner, summary, _ = finished
        assert runner.validate_state()
        for row in runner.population.trace:
            assert row["in_bar"] == (row["actual_target"] == "bar")

    def test_liars_lose_trust(self, finished):
        runner, _, _ = finished
        listener = runner.population.minds[2]
        liars = [aid for aid, m in runner.population.minds.items() if m.public_lies and aid != 2]
        honest = [aid for aid, m in runner.population.minds.items() if m.public_claims and not m.public_lies and aid != 2]
        assert liars and honest
        assert max(listener.trust[a] for a in liars) < min(listener.trust[a] for a in honest)

    def test_exports_trace_and_minds(self, finished):
        runner, summary, out = finished
        with open(out / TRACE_FILENAME, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12 * 6
        assert list(rows[0].keys()) == list(TRACE_FIELDS)
        assert {r["private_note"] for r in rows} == {"bluffing", "honest"}
        with open(out / MINDS_FILENAME, encoding="utf-8") as f:
            minds = json.load(f)
        assert len(minds) == 12
        assert summary["agent_model"] == "minds"
        assert 0.0 < summary["mind_lie_rate"] < 1.0

    def test_request_seeds_are_issued_per_agent(self, tmp_path):
        class Channel(ScriptedMinds):
            request_seeds = None
            seen = []

            def generate(self, prompts, sampling_params=None, **kwargs):
                self.seen.append(self.request_seeds)
                self.request_seeds = None
                return super().generate(prompts, sampling_params, **kwargs)

        channel = Channel()
        config = RunConfig(num_agents=4, grid_size=20, num_epochs=2, seed=9, output_dir=str(tmp_path))
        MindRunner(config, VLLMBatchPipeline(channel, None)).run()
        assert len(channel.seen) == 2
        assert all(len(set(seeds)) == 4 for seeds in channel.seen)
        assert channel.seen[0] != channel.seen[1]
