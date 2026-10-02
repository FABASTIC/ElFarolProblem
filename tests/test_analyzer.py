import json
import math
import re
import sys

import numpy as np
import pytest
from unittest.mock import patch

from analyzer import (
    PAYOFFS,
    binomial_cdf,
    build_report,
    changepoint,
    contrast,
    gini,
    gini_classic,
    js_divergence_bits,
    kl_divergence,
    main,
    nash_equilibrium,
    write_report,
)
from experiment import run_experiment
from elfarol.simulation_runner import VLLMBatchPipeline


class _Choice:

    def __init__(self, text):
        self.text = text


class _Output:

    def __init__(self, text):
        self.outputs = [_Choice(text)]


class ScriptedVLLM:

    def generate(self, prompts, sampling_params=None, **kwargs):
        outputs = []
        for prompt in prompts:
            agent = int(re.search(r"You are Agent (\d+)", prompt).group(1))
            lo = int(re.search(r"Bar zone: \((\d+),", prompt).group(1))
            epoch = int(re.search(r"Epoch: (\d+)", prompt).group(1))
            going = (agent + epoch) % 3 != 0
            target = [lo + agent % 10, lo + (agent // 10) % 10] if going else [0, agent]
            stated = "going" if going != (agent == 2) else "staying"
            outputs.append(_Output(json.dumps({
                "move": target,
                "broadcast": f"agent {agent} says {stated}",
                "proximity_speech": None,
                "stated_intention": stated,
                "actual_target": "bar" if going else "home",
            })))
        return outputs


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep")
    run_experiment(
        model_name="test-model",
        base_output_dir=str(out),
        num_agents=10,
        num_epochs=6,
        grid_size=20,
        seeds=[1, 2],
        dry_run_pipeline=VLLMBatchPipeline(ScriptedVLLM(), sampling_params=None),
    )
    return out


class TestGini:

    def test_equal_values_are_perfectly_equal(self):
        assert gini([3.0, 3.0, 3.0, 3.0]) == 0.0

    def test_full_concentration_is_one(self):
        assert gini([0.0, 0.0, 0.0, 10.0]) == pytest.approx(1.0)

    def test_matches_pairwise_definition_with_negative_payoffs(self):
        x = np.array([-2.0, 0.5, 1.0, 4.0, 4.0])
        expected = np.abs(x[:, None] - x[None, :]).sum() / (2 * (len(x) - 1) * np.abs(x).sum())
        assert gini(x) == pytest.approx(expected)

    def test_bounded_for_signed_samples(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            value = gini(rng.normal(0.0, 5.0, size=20))
            assert 0.0 <= value <= 1.0 + 1e-12

    def test_rescales_classic_gini_for_non_negative_data(self):
        x = [1.0, 2.0, 3.0, 10.0]
        assert gini(x) == pytest.approx(gini_classic(x) * 4 / 3)

    def test_classic_gini_undefined_for_negative_payoffs(self):
        assert math.isnan(gini_classic([-1.0, 2.0]))


class TestNashEquilibrium:

    def test_binomial_cdf_matches_direct_sum(self):
        n, p = 12, 0.37
        for k in range(-1, 14):
            direct = sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, min(k, n) + 1)) if k >= 0 else 0.0
            assert binomial_cdf(k, n, p) == pytest.approx(min(1.0, direct), abs=1e-12)

    def test_population_below_threshold_makes_going_dominant(self):
        ne = nash_equilibrium(50, 60, PAYOFFS)
        assert ne["pure_attendance"] == [50]
        assert ne["mixed_go_probability"] == 1.0
        assert ne["dominant_strategy"] == "go"
        assert ne["congestion_possible"] is False

    def test_classic_el_farol_mixed_equilibrium_is_indifferent(self):
        ne = nash_equilibrium(100, 60, PAYOFFS)
        assert ne["pure_attendance"] == [60]
        assert ne["social_optimum_attendance"] == 60
        assert ne["dominant_strategy"] is None
        p = ne["mixed_go_probability"]
        f = binomial_cdf(59, 99, p)
        expected_go = PAYOFFS["bar_comfortable"] * f + PAYOFFS["bar_overcrowded"] * (1 - f)
        assert 0.0 < p < 1.0
        assert expected_go == pytest.approx(PAYOFFS["home"], abs=1e-9)


class TestDivergences:

    def test_kl_vanishes_for_identical_distributions(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        assert kl_divergence(p, p) == pytest.approx(0.0)

    def test_kl_is_positive_and_asymmetric(self):
        p = np.array([0.7, 0.1, 0.1, 0.1])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, q) > 0.0
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))

    def test_jensen_shannon_is_bounded_in_bits(self):
        p = np.array([0.999, 0.0005, 0.0003, 0.0002])
        q = np.array([0.0002, 0.0003, 0.0005, 0.999])
        assert 0.0 <= js_divergence_bits(p, q) <= 1.0


class TestChangepoint:

    def test_recovers_planted_step(self):
        result = changepoint([0.1] * 12 + [0.8] * 18)
        assert result["epoch"] == 12
        assert result["r2"] == pytest.approx(1.0)

    def test_flat_series_has_no_changepoint(self):
        assert changepoint([0.3] * 10) is None


class TestContrast:

    @staticmethod
    def _entries(values, shift):
        return {seed: {"value": v + shift, "replicates": np.full(200, v + shift)} for seed, v in values.items()}

    def test_paired_design_uses_exact_sign_flip(self):
        base = {42: 1.0, 100: 2.0, 2026: 3.0}
        result = contrast(self._entries(base, 0.0), self._entries(base, 0.5), np.random.default_rng(1), 200)
        assert result["design"] == "paired"
        assert result["estimate"] == pytest.approx(0.5)
        assert result["p_value"] == pytest.approx(0.25)
        assert result["ci95"] == pytest.approx([0.5, 0.5])

    def test_disjoint_seeds_fall_back_to_exact_permutation(self):
        control = {1: {"value": 0.0, "replicates": np.zeros(100)}, 2: {"value": 1.0, "replicates": np.ones(100)}}
        treatment = {3: {"value": 5.0, "replicates": np.full(100, 5.0)}, 4: {"value": 6.0, "replicates": np.full(100, 6.0)}}
        result = contrast(control, treatment, np.random.default_rng(2), 100)
        assert result["design"] == "unpaired"
        assert result["estimate"] == pytest.approx(5.0)
        assert result["p_value"] == pytest.approx(2 / 6)


class TestReportEndToEnd:

    def test_report_covers_both_conditions_with_paired_contrast(self, sweep):
        report = build_report(str(sweep), bootstrap=40)
        assert report["source"]["trials_analyzed"] == 4
        assert set(report["conditions"]) == {"control", "delta2"}
        assert report["contrast"]["paired_seeds"] == [1, 2]
        assert report["game"]["nash"]["pure_attendance"] == [10]
        for section in ("gini", "drift", "nash", "behavior"):
            assert report["contrast"]["metrics"][section]
        assert [t["trial"] for t in report["trials"]] == ["control_seed_1", "control_seed_2", "delta2_seed_1", "delta2_seed_2"]

    def test_regret_equals_forgone_dominant_payoff(self, sweep):
        report = build_report(str(sweep), bootstrap=10)
        gain = PAYOFFS["bar_comfortable"] - PAYOFFS["home"]
        for trial in report["trials"]:
            n = trial["agents"]
            for attendance, regret in zip(trial["series"]["bar_attendance"], trial["series"]["regret"]):
                assert regret == pytest.approx(gain * (n - attendance) / n)

    def test_report_is_strict_json(self, sweep, tmp_path):
        path = write_report(build_report(str(sweep), bootstrap=20), str(tmp_path / "report.json"))
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text)["schema"] == "elfarol.analytics/1"

    def test_cli_writes_report_next_to_comparison(self, sweep):
        with patch.object(sys, "argv", ["analyzer.py", "--experiment-dir", str(sweep), "--bootstrap", "20", "--quiet"]):
            main()
        assert (sweep / "analytics_report.json").exists()
