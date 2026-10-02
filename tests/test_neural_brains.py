import csv
import json
import os
import sys
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

import experiment
from experiment import (
    EPOCH_FIELDS,
    AgentBrain,
    AgentMemory,
    BrainColony,
    actor_dim,
    act_forward,
    main,
    observation_dim,
    run_experiment,
    speak_forward,
)

ORIGINAL_EPOCH_FIELDS = [
    "epoch", "bar_attendance", "bar_capacity", "comfort_threshold",
    "attendance_over_threshold", "total_utility", "mean_utility",
    "deception_index", "truthfulness_ratio", "deception_count",
    "truthful_count", "broadcast_count", "broadcast_correlation",
]


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


def _night(colony, agents, epoch, script):
    colony.broadcast_phase(agents, epoch)
    colony.action_phase(agents, epoch, 0.5)
    actions = []
    for agent in agents:
        says, goes = script.get(agent.id, (False, False))
        agent.in_bar_flag = goes
        actions.append({"stated_intention": "going" if says else "staying", "actual_target": "bar" if goes else "home"})
    utilities = [1.0 if a.in_bar_flag else 0.3 for a in agents]
    colony.observe(agents, actions, epoch, utilities, [False] * len(agents))


class TestAgentBrain:

    @pytest.mark.parametrize("input_dim", [3, 17, 64])
    def test_brain_accepts_any_input_dimension(self, input_dim):
        brain = AgentBrain(1, 7, input_dim, 12, "cpu")
        assert brain.speak(torch.randn(5, input_dim)).shape == (5, 2)
        q, f = brain.act(torch.randn(5, input_dim + 2))
        assert q.shape == (5, 2) and f.shape == (5,)
        assert brain.params["s_w1"].shape == (12, input_dim)
        assert brain.params["a_w1"].shape == (12, input_dim + 2)

    def test_history_length_sets_state_width(self):
        colony = _colony(history=11)
        assert colony.input_dim == observation_dim(11)
        assert colony.actor_dim == actor_dim(11)
        assert all(b.memory.speak_states.shape[1] == observation_dim(11) for b in colony.brains)
        assert all(b.memory.act_states.shape[1] == actor_dim(11) for b in colony.brains)

    def test_memory_ring_buffers_wrap(self):
        memory = AgentMemory(3, 2, "cpu")
        s = torch.zeros(observation_dim(2))
        x = torch.zeros(actor_dim(2))
        for i in range(5):
            memory.push_act(x + i, torch.zeros(2), s, 0.5)
            memory.push_speech(s + i, 1, 0.0, s)
        assert memory.act_size == 3 and memory.speak_size == 3
        assert sorted(memory.act_states[:, 0].tolist()) == [2.0, 3.0, 4.0]
        assert sorted(memory.speak_states[:, 0].tolist()) == [2.0, 3.0, 4.0]


class TestIsolation:

    def test_every_agent_owns_distinct_weights_optimizer_and_memory(self):
        colony = _colony(n=5)
        weights = [b.params["a_w1"] for b in colony.brains]
        assert len({w.data_ptr() for w in weights}) == 5
        assert not torch.equal(weights[0], weights[1])
        assert len({id(b.optimizer) for b in colony.brains}) == 5
        assert len({b.memory.act_states.data_ptr() for b in colony.brains}) == 5
        for brain in colony.brains:
            owned = {p.data_ptr() for p in brain.params.values()}
            group = {p.data_ptr() for p in brain.optimizer.param_groups[0]["params"]}
            assert owned == group

    def test_an_agent_without_experience_is_never_updated(self):
        colony = _colony(n=4)
        silent = colony.brains[2]
        before = {k: v.detach().clone() for k, v in silent.params.items()}
        agents = [_Agent(i, i, 0) for i in range(1, 5)]
        colony.broadcast_phase(agents, 0)
        colony.action_phase(agents, 0, 0.5)
        for i, brain in enumerate(colony.brains):
            if i != 2:
                brain.memory.push_act(colony.pending["x"][i], torch.ones(2), colony.pending["s"][i], 0.5)
        colony._train()
        assert all(torch.equal(before[k], silent.params[k]) for k in before)
        assert silent.updates == 0
        assert colony.brains[0].updates == colony.updates

    def test_gradient_of_one_agent_never_touches_another(self):
        colony = _colony(n=3)
        x = torch.randn(3, 2, colony.actor_dim)
        q, _ = act_forward(colony._stack(), x)
        q[0].sum().backward()
        assert colony.brains[0].params["a_w1"].grad is not None
        assert colony.brains[1].params["a_w1"].grad is None or torch.count_nonzero(colony.brains[1].params["a_w1"].grad) == 0

    def test_batched_forward_matches_each_brain_alone(self):
        colony = _colony(n=4)
        s = torch.randn(4, 3, colony.input_dim)
        x = torch.randn(4, 3, colony.actor_dim)
        with torch.no_grad():
            stacked = colony._stack()
            q_s = speak_forward(stacked, s)
            q_a, f = act_forward(stacked, x)
            for i, brain in enumerate(colony.brains):
                assert torch.allclose(q_s[i], brain.speak(s[i]), atol=1e-5)
                q_i, f_i = brain.act(x[i])
                assert torch.allclose(q_a[i], q_i, atol=1e-5)
                assert torch.allclose(f[i], f_i, atol=1e-5)


class TestTwoStepNight:

    def test_public_intents_are_binary_and_feed_the_town_ratio(self):
        colony = _colony()
        agents = [_Agent(i, i, 1) for i in range(1, 7)]
        intents = colony.broadcast_phase(agents, 0)
        assert set(intents) <= {0, 1}
        actions = colony.action_phase(agents, 0, 0.83)
        x = colony.pending["x"]
        assert torch.allclose(x[:, -2], torch.full((6,), 0.83))
        assert x[:, -1].tolist() == [float(v) for v in colony.pending["claims"].tolist()]
        assert [a["stated_intention"] == "going" for a in actions] == [bool(v) for v in intents]

    def test_control_town_cannot_hear_the_ratio(self):
        colony = _colony(broadcast=False)
        agents = [_Agent(i, i, 1) for i in range(1, 7)]
        colony.broadcast_phase(agents, 0)
        actions = colony.action_phase(agents, 0, 0.9)
        assert torch.count_nonzero(colony.pending["x"][:, -2]) == 0
        assert all(a["broadcast"] is None for a in actions)

    def test_actor_reads_the_town_broadcast(self):
        brain = AgentBrain(1, 4, observation_dim(4), 16, "cpu")
        base = torch.zeros(observation_dim(4) + 2)
        quiet = base.clone()
        loud = base.clone()
        loud[-2] = 1.0
        q_quiet, _ = brain.act(quiet.unsqueeze(0))
        q_loud, _ = brain.act(loud.unsqueeze(0))
        assert not torch.allclose(q_quiet, q_loud)

    def test_decisions_are_reproducible(self):
        agents = [_Agent(i, i, 1) for i in range(1, 7)]
        assert _colony().decide(agents, 0) == _colony().decide(agents, 0)


class TestDeceptionGradient:

    def test_bluff_then_win_is_credited_to_the_claim(self):
        colony = _colony(n=4)
        agents = [_Agent(i, i, 1) for i in range(1, 5)]
        _night(colony, agents, 0, {1: (True, False)})
        _night(colony, agents, 1, {1: (True, True)})
        memory = colony.brains[0].memory
        gamma = colony.brains[0].temperament["discount"]
        assert memory.speak_size == 1
        assert memory.speak_claims[0].item() == 1
        assert memory.speak_returns[0].item() == pytest.approx(0.3 + gamma * 1.0)
        assert colony.manipulation_wins[0] == 1
        assert colony.manipulation_wins[1] == 0

    def test_speaker_learns_towards_the_manipulation_return(self):
        colony = _colony(n=2)
        brain = colony.brains[0]
        s = torch.zeros(colony.input_dim)
        target = 2.5
        brain.memory.push_speech(s, 1, target, s)
        for other in colony.brains:
            other.memory.push_act(torch.zeros(colony.actor_dim), torch.zeros(2), s, 0.0)
        frozen = {k: v.detach().clone() for k, v in brain.target.items()}
        start = abs(brain.speak(s.unsqueeze(0))[0, 1].item() - target)
        for _ in range(60):
            colony._train()
            for key, value in frozen.items():
                brain.target[key].copy_(value)
        end = abs(brain.speak(s.unsqueeze(0))[0, 1].item() - target)
        assert end < start

    def test_honesty_coupling_is_learnable(self):
        brain = AgentBrain(1, 9, observation_dim(3), 8, "cpu")
        assert brain.params["a_h"].requires_grad
        assert any(p is brain.params["a_h"] for p in brain.optimizer.param_groups[0]["params"])


class TestNeuralSweep:

    def test_sweep_logs_ratio_and_attendance_without_breaking_formats(self, tmp_path):
        summaries, comparison_path = run_experiment(model_name="brains", base_output_dir=str(tmp_path), num_agents=12, num_epochs=4, grid_size=20, seeds=[5], device="cpu", hidden=8, memory=16, batch_size=4, updates=1)
        assert set(summaries) == {"control", "delta2"}
        with open(comparison_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            assert row["agent_model"] == "neural"
            assert len(row["town_broadcast_ratio"]) == 4 and len(row["actual_attendance"]) == 4
            assert all(0.0 <= v <= 1.0 for v in row["town_broadcast_ratio"])
            assert "average_town_broadcast_ratio" in row and "average_actual_attendance" in row
        trial = tmp_path / "delta2_seed_5"
        for name in ("epoch_metrics.csv", "epoch_metrics.json", "agent_epoch_details.csv", "summary.json", "mind_trace.csv", "minds.json", "brains.pt", "trial_manifest.json"):
            assert (trial / name).exists(), name
        with open(trial / "epoch_metrics.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames[: len(ORIGINAL_EPOCH_FIELDS)] == ORIGINAL_EPOCH_FIELDS
            assert reader.fieldnames == EPOCH_FIELDS
            epochs = list(reader)
        assert all(row["actual_attendance"] == row["bar_attendance"] for row in epochs)
        with open(trial / "agent_epoch_details.csv", "r", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == ["epoch", "agent_id", "x", "y", "in_bar", "utility", "cumulative_utility", "stated_intention", "actual_target", "actual_location", "is_deceptive", "broadcast"]
        with open(trial / "epoch_metrics.json", "r", encoding="utf-8") as f:
            assert "town_broadcast_ratio" in json.load(f)[0]
        with open(trial / "minds.json", "r", encoding="utf-8") as f:
            minds = json.load(f)
        assert "manipulation_wins" in minds["1"] and "honesty" in minds["1"]["traits"]
        assert len(torch.load(trial / "brains.pt")) == 12

    def test_live_frame_carries_ratio_history(self, tmp_path):
        frames = []

        def capture(path, payload):
            if isinstance(payload, dict) and payload.get("current") and payload["current"].get("agents"):
                frames.append(payload["current"])
            return path

        with patch("experiment._atomic_write_json", side_effect=capture):
            run_experiment(model_name="brains", base_output_dir=str(tmp_path), num_agents=8, num_epochs=3, grid_size=20, seeds=[2], device="cpu", hidden=4)
        last = frames[-1]
        assert len(last["history"]["town_broadcast_ratio"]) == 3
        assert "town_broadcast_ratio" in last["metrics"] and "actual_attendance" in last["metrics"]
        assert last["agent_model"] == "neural"

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
