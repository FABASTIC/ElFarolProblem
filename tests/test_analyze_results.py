import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from analyze_results import (
    load_metrics_csv,
    compute_rolling_deception,
    detect_phase_transition,
    extract_cumulative_utilities,
    analyze_experiment,
    main,
)


class TestLoadMetricsCSV:

    def test_load_epoch_level_csv(self, tmp_path):
        csv_path = tmp_path / "epoch_metrics.csv"
        df = pd.DataFrame({
            "epoch": [0, 1, 2],
            "bar_attendance": [20, 35, 65],
            "comfort_threshold": [60, 60, 60],
            "total_utility": [10.0, 15.0, -25.0],
            "mean_utility": [0.2, 0.3, -0.5],
            "deception_index": [0.05, 0.2, 0.8],
        })
        df.to_csv(csv_path, index=False)

        loaded = load_metrics_csv(str(csv_path))
        assert len(loaded) == 3
        assert list(loaded["epoch"]) == [0, 1, 2]
        assert "deception_index" in loaded.columns
        assert loaded["deception_index"].iloc[2] == 0.8

    def test_load_agent_level_csv(self, tmp_path):
        csv_path = tmp_path / "agent_details.csv"
        df = pd.DataFrame({
            "epoch": [0, 0, 1, 1],
            "agent_id": [1, 2, 1, 2],
            "stated_intention": ["going", "staying", "going", "going"],
            "actual_target": ["going", "staying", "staying", "going"],
            "in_bar": [1, 0, 0, 1],
            "utility": [1.0, 0.3, 0.3, 1.0],
        })
        df.to_csv(csv_path, index=False)

        loaded = load_metrics_csv(str(csv_path))
        assert len(loaded) == 2
        assert loaded["deception_index"].iloc[0] == 0.0
        assert loaded["deception_index"].iloc[1] == 0.5


class TestRollingDeception:

    def test_rolling_average_window(self):
        df = pd.DataFrame({
            "epoch": list(range(6)),
            "deception_index": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
        rolling = compute_rolling_deception(df, window=3)
        assert len(rolling) == 6
        assert rolling.iloc[0] == 0.0
        assert round(rolling.iloc[2], 4) == 0.2
        assert round(rolling.iloc[5], 4) == 0.8


class TestDetectPhaseTransition:

    def test_exact_phase_transition_detection(self):
        epochs = list(range(20))
        deception = [0.1] * 10 + [0.85] * 10
        mean_utility = [0.6] * 10 + [-0.8] * 10
        df = pd.DataFrame({
            "epoch": epochs,
            "deception_index": deception,
            "mean_utility": mean_utility,
        })

        result = detect_phase_transition(df, window=2)
        assert result["transition_epoch"] == 10
        assert result["deception_spike"] > 0
        assert result["utility_crash"] < 0
        assert result["score"] > 0

    def test_short_dataframe_handles_gracefully(self):
        df = pd.DataFrame({"epoch": [0], "deception_index": [0.1]})
        result = detect_phase_transition(df)
        assert result["transition_epoch"] == 0
        assert result["score"] == 0.0


class TestExtractCumulativeUtilities:

    def test_extract_from_agent_csv(self, tmp_path):
        agent_csv = tmp_path / "agent_epoch_details.csv"
        df = pd.DataFrame({
            "epoch": [0, 0, 1, 1],
            "agent_id": [0, 1, 0, 1],
            "cumulative_utility": [0.3, 1.0, 0.6, 2.0],
        })
        df.to_csv(agent_csv, index=False)

        utils = extract_cumulative_utilities(str(agent_csv))
        assert len(utils) == 2
        assert 0.6 in utils
        assert 2.0 in utils

    def test_extract_fallback_from_epoch_metrics(self, tmp_path):
        epoch_csv = tmp_path / "epoch_metrics.csv"
        df = pd.DataFrame({
            "epoch": [0, 1],
            "total_utility": [50.0, 40.0],
            "mean_utility": [1.0, 0.8],
        })
        df.to_csv(epoch_csv, index=False)

        utils = extract_cumulative_utilities(str(epoch_csv), default_agents=10)
        assert len(utils) == 10


class TestAnalyzeExperimentAndCLI:

    def test_end_to_end_analysis(self, tmp_path):
        ctrl_csv = tmp_path / "ctrl.csv"
        d2_csv = tmp_path / "d2.csv"

        df_c = pd.DataFrame({
            "epoch": range(10),
            "bar_attendance": [25] * 10,
            "deception_index": [0.0] * 10,
            "mean_utility": [0.5] * 10,
            "total_utility": [25.0] * 10,
        })
        df_d2 = pd.DataFrame({
            "epoch": range(10),
            "bar_attendance": [20] * 5 + [70] * 5,
            "deception_index": [0.1] * 5 + [0.9] * 5,
            "mean_utility": [0.8] * 5 + [-0.9] * 5,
            "total_utility": [40.0] * 5 + [-45.0] * 5,
        })
        df_c.to_csv(ctrl_csv, index=False)
        df_d2.to_csv(d2_csv, index=False)

        analysis = analyze_experiment(str(ctrl_csv), str(d2_csv), window=2)
        assert "phase_transition" in analysis
        assert analysis["phase_transition"]["transition_epoch"] == 5

    def test_cli_execution(self, tmp_path):
        ctrl_csv = tmp_path / "ctrl.csv"
        d2_csv = tmp_path / "d2.csv"
        out_json = tmp_path / "analysis.json"

        pd.DataFrame({
            "epoch": range(5),
            "bar_attendance": [20] * 5,
            "deception_index": [0.0] * 5,
            "mean_utility": [0.5] * 5,
        }).to_csv(ctrl_csv, index=False)

        pd.DataFrame({
            "epoch": range(5),
            "bar_attendance": [20] * 5,
            "deception_index": [0.2] * 5,
            "mean_utility": [0.4] * 5,
        }).to_csv(d2_csv, index=False)

        test_args = [
            "analyze_results.py",
            "--control", str(ctrl_csv),
            "--delta2", str(d2_csv),
            "--output", str(out_json),
        ]
        with patch.object(sys, "argv", test_args):
            main()

        assert out_json.exists()
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "transition_epoch" in data
