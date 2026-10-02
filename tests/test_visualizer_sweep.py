import sys

import pytest
from unittest.mock import patch

from experiment import run_experiment
from elfarol.simulation_runner import VLLMBatchPipeline
from tests.test_analyzer import ScriptedVLLM
from visualizer import generate_sweep_figures, main


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    out = tmp_path_factory.mktemp("figures_sweep")
    run_experiment(
        model_name="test-model",
        base_output_dir=str(out),
        num_agents=10,
        num_epochs=8,
        grid_size=20,
        seeds=[3, 4],
        dry_run_pipeline=VLLMBatchPipeline(ScriptedVLLM(), sampling_params=None),
    )
    return out


class TestSweepFigures:

    def test_generates_phase_atlas_heatmaps_and_editorial(self, sweep, tmp_path):
        paths = generate_sweep_figures(str(sweep), figures_dir=str(tmp_path), formats=("png", "pdf"), dpi=60, bootstrap=20)
        names = sorted(p.split("\\")[-1].split("/")[-1] for p in paths)
        assert names == [
            "editorial_seed_3.pdf",
            "phase_transitions.pdf",
            "phase_transitions.png",
            "strategy_heatmaps.pdf",
            "strategy_heatmaps.png",
        ]
        with open(tmp_path / "phase_transitions.png", "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"
        with open(tmp_path / "strategy_heatmaps.pdf", "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_cli_defaults_to_sweep_mode(self, sweep, tmp_path):
        argv = ["visualizer.py", "--experiment-dir", str(sweep), "--figures-dir", str(tmp_path), "--formats", "png", "--dpi", "60", "--no-editorial"]
        with patch.object(sys, "argv", argv):
            main()
        assert (tmp_path / "phase_transitions.png").exists()
        assert (tmp_path / "strategy_heatmaps.png").exists()

    def test_empty_directory_exits_cleanly(self, tmp_path):
        with patch.object(sys, "argv", ["visualizer.py", "--experiment-dir", str(tmp_path)]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
