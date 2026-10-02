import json
import os
from unittest.mock import MagicMock

import pytest

from experiment import _save_comparison, run_experiment
from elfarol.simulation_runner import VLLMBatchPipeline


class _Choice:

    def __init__(self, text):
        self.text = text


class _Output:

    def __init__(self, text):
        self.outputs = [_Choice(text)]


class TestComparisonWrites:

    def test_comparison_write_leaves_no_temp_files(self, tmp_path):
        _save_comparison({"control": {"total_epochs": 1}}, str(tmp_path))
        assert sorted(os.listdir(tmp_path)) == ["comparison.json"]


def _pipeline():
    def respond(prompts, sampling_params=None, **kwargs):
        return [_Output('{"move": [9, 9], "broadcast": "hi", "proximity_speech": null, "stated_intention": "going", "actual_target": "bar"}') for _ in prompts]

    llm = MagicMock()
    llm.generate.side_effect = respond
    return VLLMBatchPipeline(llm, sampling_params=None)


class TestSweepArmor:

    def test_live_state_and_manifests_are_published(self, tmp_path):
        summaries, _ = run_experiment(
            model_name="test-model",
            base_output_dir=str(tmp_path),
            num_agents=4,
            num_epochs=3,
            grid_size=20,
            seeds=[7, 8],
            dry_run_pipeline=_pipeline(),
        )
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            live = json.load(f)
        assert live["status"] == "complete"
        assert [t["status"] for t in live["trials"]] == ["complete"] * 4
        assert len(live["trials"]) == 4
        for key, summary in summaries.items():
            assert summary["trial"] == key
            assert summary["status"] == "complete"
            with open(tmp_path / summary["trial_dir"] / "trial_manifest.json", "r", encoding="utf-8") as f:
                assert json.load(f)["seed"] == summary["seed"]

    def test_resume_skips_verified_trials(self, tmp_path):
        kwargs = dict(model_name="test-model", base_output_dir=str(tmp_path), num_agents=4, num_epochs=3, grid_size=20, seeds=[7, 8])
        run_experiment(dry_run_pipeline=_pipeline(), **kwargs)
        stamp = os.path.getmtime(tmp_path / "control_seed_7" / "epoch_metrics.csv")
        idle = _pipeline()
        summaries, _ = run_experiment(dry_run_pipeline=idle, resume=True, **kwargs)
        assert idle.llm.generate.call_count == 0
        assert os.path.getmtime(tmp_path / "control_seed_7" / "epoch_metrics.csv") == stamp
        assert all(s["runtime"].get("resumed") for s in summaries.values())

    def test_resume_reruns_mismatched_configuration(self, tmp_path):
        kwargs = dict(model_name="test-model", base_output_dir=str(tmp_path), num_agents=4, grid_size=20, seeds=[7])
        run_experiment(dry_run_pipeline=_pipeline(), num_epochs=2, **kwargs)
        rerun = _pipeline()
        run_experiment(dry_run_pipeline=rerun, resume=True, num_epochs=3, **kwargs)
        assert rerun.llm.generate.call_count == 6
