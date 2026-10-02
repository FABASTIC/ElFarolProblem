import csv
import json
import os
import sys
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

import experiment
from experiment import (
    OPTIONS,
    AgentBrain,
    AgentMemory,
    BrainColony,
    main,
    observation_dim,
    run_experiment,
)


@pytest.fixture(autouse=True)
def _clear_interrupt_flag():
    experiment._SIGNALS["interrupted"] = False
    yield
    experiment._SIGNALS["interrupted"] = False


class _Agent:

    def __init__(self, agent_id, x, y):
        self.id = agent_id
        self.x = x
        self.y = y
        self.in_bar_flag = False


def _colony(n=6, history=4, broadcast=True):
    return BrainColony(seed=3, agent_ids=list(range(1, n + 1)), num_agents=n, threshold=max(1, n // 2), broadcast_enabled=broadcast, device="cpu", hidden=8, memory=32, batch_size=4, updates=2, target_every=2, history=history)


def _night(colony, agents, epoch, goers):
    actions = colony.decide(agents, epoch)
    for agent in agents:
        agent.in_bar_flag = agent.id in goers
    utilities = [1.0 if a.in_bar_flag else 0.3 for a in agents]
    colony.observe(agents, actions, epoch, utilities, [False] * len(agents))
    return actions


class TestAgentBrain:

    @pytest.mark.parametrize("input_dim", [3, 17, 64])
    def test_brain_accepts_any_input_dimension(self, input_dim):
        brain = AgentBrain(1, 7, input_dim, 12, "cpu")
        q, f = brain.forward(torch.randn(5, input_dim))
        assert q.shape == (5, len(OPTIONS))
        assert f.shape == (5,)
        assert brain.params["w1"].shape == (12, input_dim)

    def test_history_length_sets_state_width(self):
        colony = _colony(history=11)
        assert colony.input_dim == observation_dim(11)
        assert all(b.params["w1"].shape[1] == observation_dim(11) for b in colony.brains)
        assert all(b.memory.states.shape[1] == observation_dim(11) for b in colony.brains)

    def test_memory_ring_buffer_wraps(self):
        memory = AgentMemory(3, 2, "cpu")
        state = torch.zeros(observation_dim(2))
        for i in range(5):
            memory.push(state + i, torch.zeros(len(OPTIONS)), state, 0.5)
        assert memory.size == 3
        assert sorted(memory.states[:, 0].tolist()) == [2.0, 3.0, 4.0]


class TestIsolation:

    def test_every_agent_owns_distinct_weights_optimizer_and_memory(self):
        colony = _colony(n=5)
        weights = [b.params["w1"] for b in colony.brains]
        assert len({w.data_ptr() for w in weights}) == 5
        assert not torch.equal(weights[0], weights[1])
        assert len({id(b.optimizer) for b in colony.brains}) == 5
        assert len({b.memory.states.data_ptr() for b in colony.brains}) == 5
        for brain in colony.brains:
            owned = {p.data_ptr() for p in brain.params.values()}
            group = {p.data_ptr() for p in brain.optimizer.param_groups[0]["params"]}
            assert owned == group

    def test_an_agent_without_experience_is_never_updated(self):
        colony = _colony(n=4)
        silent = colony.brains[2]
        before = {k: v.detach().clone() for k, v in silent.params.items()}
        agents = [_Agent(i, i, 0) for i in range(1, 5)]
        colony.decide(agents, 0)
        state = colony.pending["x"]
        for i, brain in enumerate(colony.brains):
            if i != 2:
                brain.memory.push(state[i], torch.ones(len(OPTIONS)), state[i], 0.5)
        colony._train()
        assert all(torch.equal(before[k], silent.params[k]) for k in before)
        assert silent.updates == 0
        assert colony.brains[0].updates == colony.updates

    def test_gradient_of_one_agent_never_touches_another(self):
        colony = _colony(n=3)
        stacked = colony._stack()
        x = torch.randn(3, 2, colony.input_dim)
        q, _ = colony._forward(stacked, x)
        q[0].sum().backward()
        assert colony.brains[0].params["w1"].grad is not None
        assert colony.brains[1].params["w1"].grad is None or torch.count_nonzero(colony.brains[1].params["w1"].grad) == 0

    def test_batched_forward_matches_each_brain_alone(self):
        colony = _colony(n=4)
        x = torch.randn(4, 3, colony.input_dim)
        with torch.no_grad():
            q, f = colony._forward(colony._stack(), x)
            for i, brain in enumerate(colony.brains):
                q_i, f_i = brain.forward(x[i])
                assert torch.allclose(q[i], q_i, atol=1e-5)
                assert torch.allclose(f[i], f_i, atol=1e-5)


class TestColony:

    def test_decisions_are_valid_and_reproducible(self):
        agents = [_Agent(i, i, 1) for i in range(1, 7)]
        first = _colony().decide(agents, 0)
        second = _colony().decide(agents, 0)
        assert first == second
        for action in first:
            assert action["stated_intention"] in ("going", "staying")
            assert action["actual_target"] in ("bar", "home")

    def test_control_colony_never_broadcasts(self):
        agents = [_Agent(i, i, 1) for i in range(1, 7)]
        assert all(a["broadcast"] is None for a in _colony(broadcast=False).decide(agents, 0))

    def test_learning_changes_weights_and_writes_trace(self):
        colony = _colony()
        agents = [_Agent(i, i, 1) for i in range(1, 7)]
        start = colony.brains[0].params["w1"].detach().clone()
        for epoch in range(4):
            _night(colony, agents, epoch, goers={1, 2})
        assert not torch.equal(start, colony.brains[0].params["w1"])
        assert len(colony.trace) == 24
        row = colony.trace[-1]
        assert abs(row["p_honest_go"] + row["p_honest_stay"] + row["p_false_go"] + row["p_false_stay"] - 1.0) < 1e-3
        view = colony.live_view(1)
        assert view["note"] and view["archetype"]
        exported = colony.export()
        assert exported["1"]["brain"]["input_dim"] == colony.input_dim
        assert "honesty" in exported["1"]["traits"]


class TestNeuralSweep:

    def test_sweep_keeps_every_output_format(self, tmp_path):
        summaries, comparison_path = run_experiment(model_name="brains", base_output_dir=str(tmp_path), num_agents=12, num_epochs=4, grid_size=20, seeds=[5], device="cpu", hidden=8, memory=16, batch_size=4, updates=1)
        assert set(summaries) == {"control", "delta2"}
        with open(comparison_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        assert len(rows) == 2 and all(r["agent_model"] == "neural" for r in rows)
        trial = tmp_path / "delta2_seed_5"
        for name in ("epoch_metrics.csv", "epoch_metrics.json", "agent_epoch_details.csv", "summary.json", "mind_trace.csv", "minds.json", "brains.pt", "trial_manifest.json"):
            assert (trial / name).exists(), name
        with open(trial / "agent_epoch_details.csv", "r", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == ["epoch", "agent_id", "x", "y", "in_bar", "utility", "cumulative_utility", "stated_intention", "actual_target", "actual_location", "is_deceptive", "broadcast"]
        weights = torch.load(trial / "brains.pt")
        assert len(weights) == 12
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            live = json.load(f)
        assert live["status"] == "complete"

    @pytest.mark.parametrize("agents", [7, 130])
    def test_agent_count_is_not_capped(self, tmp_path, agents):
        summaries, _ = run_experiment(model_name="brains", base_output_dir=str(tmp_path), num_agents=agents, num_epochs=2, grid_size=50, seeds=[1], device="cpu", hidden=4, memory=8, batch_size=2, updates=1)
        assert summaries["control"]["brain_count"] == agents

    def test_stop_file_interrupts_neural_sweep(self, tmp_path):
        stop = tmp_path / "STOP"
        stop.write_text("stop", encoding="utf-8")
        with pytest.raises(KeyboardInterrupt):
            run_experiment(model_name="brains", base_output_dir=str(tmp_path), num_agents=6, num_epochs=3, grid_size=20, seeds=[1], device="cpu", stop_file=str(stop))
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            assert json.load(f)["status"] == "interrupted"

    def test_resume_skips_verified_neural_trials(self, tmp_path):
        kwargs = dict(model_name="brains", base_output_dir=str(tmp_path), num_agents=6, num_epochs=2, grid_size=20, seeds=[1], device="cpu", hidden=4)
        run_experiment(**kwargs)
        stamp = os.path.getmtime(tmp_path / "control_seed_1" / "epoch_metrics.csv")
        summaries, _ = run_experiment(resume=True, **kwargs)
        assert os.path.getmtime(tmp_path / "control_seed_1" / "epoch_metrics.csv") == stamp
        assert all(s["runtime"].get("resumed") for s in summaries.values())

    def test_cli_accepts_dashboard_flags(self, tmp_path):
        args = ["experiment.py", "--rehearsal", "--model", "ignored", "--output-dir", str(tmp_path), "--agents", "5", "--epochs", "2", "--seed", "3", "--history", "3", "--pace", "0"]
        with patch.object(sys, "argv", args):
            main()
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            live = json.load(f)
        assert live["status"] == "complete"
        assert "cpu" in live["model"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_brains_run_on_the_gpu(self, tmp_path):
        summaries, _ = run_experiment(model_name="brains", base_output_dir=str(tmp_path), num_agents=20, num_epochs=3, grid_size=20, seeds=[1], device="cuda")
        assert summaries["control"]["brain_device"].startswith("cuda")
