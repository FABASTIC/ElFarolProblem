import json
import random
import sys
from unittest.mock import patch

import pytest

import experiment
from elfarol.minds import OPTION_TEXT, OPTIONS
from experiment import REHEARSAL_MODEL, _InstinctPipeline, _rehearse, main, run_experiment


@pytest.fixture(autouse=True)
def _clear_interrupt_flag():
    experiment._SIGNALS["interrupted"] = False
    yield
    experiment._SIGNALS["interrupted"] = False


def _mind_prompt(agent_id=7, x=3, y=4, weights=None, forecast=72, threshold=60):
    weights = weights or {"honest_go": 0.1, "honest_stay": 0.2, "false_go": 0.0, "false_stay": 0.7}
    instinct = ", ".join(f"{OPTION_TEXT[k]} {weights[k]:.2f}" for k in OPTIONS)
    return "\n".join([
        f"You are Agent {agent_id} at ({x}, {y}).",
        "Profile: GAMBLER. Temperament: honesty 0.20.",
        "Mood: hope 0.30.",
        f"Your forecast for tonight: {forecast} agents in the bar (method: same as last night, confidence 0.50); the comfort threshold is {threshold}.",
        f"Your gut instinct: {instinct}.",
    ])


class TestRehearse:

    def test_action_follows_a_certain_instinct(self):
        prompt = _mind_prompt(weights={"honest_go": 0.0, "honest_stay": 0.0, "false_go": 0.0, "false_stay": 1.0})
        action = _rehearse(prompt, random.Random(1))
        assert action["stated_intention"] == "staying"
        assert action["actual_target"] == "bar"
        assert action["move"] == [3, 4]
        assert action["private_note"]

    def test_same_seed_same_action(self):
        prompt = _mind_prompt()
        assert _rehearse(prompt, random.Random(9)) == _rehearse(prompt, random.Random(9))

    def test_note_matches_forecast_side_of_threshold(self):
        prompt = _mind_prompt(weights={"honest_go": 0.0, "honest_stay": 1.0, "false_go": 0.0, "false_stay": 0.0}, forecast=40)
        notes = {_rehearse(prompt, random.Random(s))["private_note"] for s in range(40)}
        assert not any("past the" in n for n in notes)

    def test_prompt_without_mind_still_yields_valid_action(self):
        prompt = "You are Agent 2 at (5, 6).\nComfort threshold: 60 agents.\nCurrent bar occupancy: 10."
        action = _rehearse(prompt, random.Random(3))
        assert action["stated_intention"] in ("going", "staying")
        assert action["actual_target"] in ("bar", "home")
        assert action["move"] == [5, 6]


class TestInstinctPipeline:

    def test_request_seeds_make_batches_reproducible(self):
        prompts = [_mind_prompt(agent_id=i, weights={k: 0.25 for k in OPTIONS}) for i in range(1, 30)]
        first = _InstinctPipeline()
        second = _InstinctPipeline()
        first.llm.request_seeds = list(range(100, 129))
        second.llm.request_seeds = list(range(100, 129))
        assert first.generate_batch(prompts) == second.generate_batch(prompts)
        assert first.request_seeds is None
        assert first.total_generations == 29
        assert first.fallback_count == 0


class TestRehearsalSweep:

    def test_minds_rehearsal_publishes_broadcasts_and_notes(self, tmp_path):
        summaries, _ = run_experiment(
            model_name=REHEARSAL_MODEL,
            base_output_dir=str(tmp_path),
            num_agents=12,
            num_epochs=4,
            grid_size=20,
            seeds=[5],
            dry_run_pipeline=_InstinctPipeline(),
        )
        assert set(summaries) == {"control", "delta2"}
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            live = json.load(f)
        assert live["status"] == "complete"
        assert live["model"] == REHEARSAL_MODEL
        trace = (tmp_path / "delta2_seed_5" / "mind_trace.csv").read_text(encoding="utf-8")
        assert "private_note" in trace.splitlines()[0]
        assert summaries["delta2"]["agent_model"] == "minds"

    def test_live_frame_carries_every_broadcast(self, tmp_path):
        frames = []

        def capture(path, payload):
            if isinstance(payload, dict) and payload.get("current") and payload["current"].get("agents"):
                frames.append(payload["current"])
            return path

        with patch("experiment._atomic_write_json", side_effect=capture):
            run_experiment(
                model_name=REHEARSAL_MODEL,
                base_output_dir=str(tmp_path),
                num_agents=16,
                num_epochs=3,
                grid_size=20,
                seeds=[11],
                dry_run_pipeline=_InstinctPipeline(),
            )
        delta = [f for f in frames if f["condition"] == "delta2"]
        assert delta
        assert max(len(f["broadcasts"]) for f in delta) > 1
        assert all(not f["broadcasts"] for f in frames if f["condition"] == "control")

    def test_stop_file_interrupts_the_sweep(self, tmp_path):
        stop = tmp_path / "STOP"
        stop.write_text("stop", encoding="utf-8")
        with pytest.raises(KeyboardInterrupt):
            run_experiment(
                model_name=REHEARSAL_MODEL,
                base_output_dir=str(tmp_path),
                num_agents=6,
                num_epochs=3,
                grid_size=20,
                seeds=[1],
                dry_run_pipeline=_InstinctPipeline(),
                stop_file=str(stop),
            )
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            assert json.load(f)["status"] == "interrupted"

    def test_stop_file_mid_trial_marks_running_trial_interrupted(self, tmp_path):
        stop = tmp_path / "STOP"
        pipeline = _InstinctPipeline()
        original = pipeline.generate_batch

        def generate(prompts, fallback_positions=None):
            if pipeline.calls >= 1:
                stop.write_text("stop", encoding="utf-8")
            return original(prompts, fallback_positions=fallback_positions)

        pipeline.generate_batch = generate
        with pytest.raises(KeyboardInterrupt):
            run_experiment(
                model_name=REHEARSAL_MODEL,
                base_output_dir=str(tmp_path),
                num_agents=6,
                num_epochs=5,
                grid_size=20,
                seeds=[1],
                dry_run_pipeline=pipeline,
                stop_file=str(stop),
            )
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            live = json.load(f)
        assert live["status"] == "interrupted"
        assert live["trials"][0]["status"] == "interrupted"
        assert live["current"]["epoch"] < 4


class TestRehearsalCLI:

    def test_cli_rehearsal_needs_no_engine(self, tmp_path):
        args = ["experiment.py", "--rehearsal", "--output-dir", str(tmp_path), "--agents", "8", "--epochs", "2", "--seed", "3"]
        with patch.object(sys, "argv", args):
            with patch("experiment._build_vllm_pipeline", side_effect=AssertionError("engine must not be built")):
                main()
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            live = json.load(f)
        assert live["model"] == REHEARSAL_MODEL
        assert live["status"] == "complete"
