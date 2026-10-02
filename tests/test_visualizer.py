import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

from visualizer import (
    apply_editorial_theme,
    plot_simulation_results,
    generate_figures_from_csv,
    main,
    COLOR_BG,
    COLOR_SPINE,
)


class TestVisualizerTheme:

    def test_editorial_theme_colors(self):
        fig, axes = plt.subplots(1, 3)
        apply_editorial_theme(fig, axes)

        assert fig.get_facecolor()[:3] == COLOR_BG
        for ax in axes:
            assert ax.get_facecolor()[:3] == COLOR_BG
            assert ax.spines["top"].get_visible() is False
            assert ax.spines["right"].get_visible() is False
        plt.close(fig)


class TestPlotSimulationResults:

    def test_plot_generates_valid_pdf(self, tmp_path):
        epochs = list(range(100))
        ctrl_att = [25 + np.sin(e) * 5 for e in epochs]
        d2_att = [20 + e * 0.4 for e in epochs]
        d2_dec = [0.1 + (0.7 if e > 45 else 0.0) for e in epochs]

        control_df = pd.DataFrame({
            "epoch": epochs,
            "bar_attendance": ctrl_att,
            "comfort_threshold": [60.0] * 100,
        })
        delta2_df = pd.DataFrame({
            "epoch": epochs,
            "bar_attendance": d2_att,
            "deception_index": d2_dec,
            "comfort_threshold": [60.0] * 100,
        })

        ctrl_u = np.random.normal(30.0, 5.0, size=50)
        d2_u = np.random.normal(15.0, 8.0, size=50)

        pdf_path = tmp_path / "test_figure.pdf"
        out = plot_simulation_results(
            control_df=control_df,
            delta2_df=delta2_df,
            control_cumulative_utility=ctrl_u,
            delta2_cumulative_utility=d2_u,
            phase_transition_epoch=46,
            output_path=str(pdf_path),
            window=5,
        )

        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000
        with open(out, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-"


class TestGenerateFiguresFromCSV:

    def test_end_to_end_figure_generation(self, tmp_path):
        ctrl_csv = tmp_path / "control_metrics.csv"
        d2_csv = tmp_path / "delta2_metrics.csv"
        pdf_out = tmp_path / "publication_figure.pdf"

        pd.DataFrame({
            "epoch": range(50),
            "bar_attendance": [28] * 50,
            "deception_index": [0.0] * 50,
            "mean_utility": [0.4] * 50,
            "total_utility": [20.0] * 50,
            "comfort_threshold": [60] * 50,
        }).to_csv(ctrl_csv, index=False)

        pd.DataFrame({
            "epoch": range(50),
            "bar_attendance": [25] * 25 + [68] * 25,
            "deception_index": [0.1] * 25 + [0.85] * 25,
            "mean_utility": [0.6] * 25 + [-0.8] * 25,
            "total_utility": [30.0] * 25 + [-40.0] * 25,
            "comfort_threshold": [60] * 50,
        }).to_csv(d2_csv, index=False)

        res = generate_figures_from_csv(
            control_csv=str(ctrl_csv),
            delta2_csv=str(d2_csv),
            output_path=str(pdf_out),
            window=3,
        )

        assert os.path.exists(res)
        assert os.path.getsize(res) > 1000


class TestVisualizerCLI:

    def test_main_cli(self, tmp_path):
        ctrl_csv = tmp_path / "control_metrics.csv"
        d2_csv = tmp_path / "delta2_metrics.csv"
        pdf_out = tmp_path / "cli_figure.pdf"

        pd.DataFrame({
            "epoch": range(20),
            "bar_attendance": [25] * 20,
            "deception_index": [0.05] * 20,
            "mean_utility": [0.3] * 20,
            "total_utility": [15.0] * 20,
        }).to_csv(ctrl_csv, index=False)

        pd.DataFrame({
            "epoch": range(20),
            "bar_attendance": [30] * 20,
            "deception_index": [0.25] * 20,
            "mean_utility": [0.1] * 20,
            "total_utility": [5.0] * 20,
        }).to_csv(d2_csv, index=False)

        test_args = [
            "visualizer.py",
            "--control", str(ctrl_csv),
            "--delta2", str(d2_csv),
            "--output", str(pdf_out),
        ]
        with patch.object(sys, "argv", test_args):
            main()

        assert pdf_out.exists()
        assert pdf_out.stat().st_size > 1000
