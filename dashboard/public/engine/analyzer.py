import os
import re
import sys
import json
import math
import argparse
import itertools
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from elfarol.metrics_logger import (
        UTILITY_AT_BAR_COMFORTABLE,
        UTILITY_AT_BAR_OVERCROWDED,
        UTILITY_AT_HOME,
        COMFORT_THRESHOLD_RATIO,
    )
except ImportError:
    UTILITY_AT_BAR_COMFORTABLE = 1.0
    UTILITY_AT_BAR_OVERCROWDED = -1.0
    UTILITY_AT_HOME = 0.3
    COMFORT_THRESHOLD_RATIO = 0.6

REPORT_SCHEMA = "elfarol.analytics/1"
REPORT_FILENAME = "analytics_report.json"
CONTROL_LABEL = "control"
TREATMENT_LABEL = "delta2"
STRATEGIES = ("honest_go", "honest_stay", "false_go", "false_stay")
STRATEGY_DEFINITIONS = {
    "honest_go": "stated 'going' and ended the epoch inside the bar",
    "honest_stay": "stated 'staying' and ended the epoch outside the bar",
    "false_go": "stated 'going' but ended the epoch outside the bar",
    "false_stay": "stated 'staying' but ended the epoch inside the bar",
}
SMOOTHING_ALPHA = 0.5
DEFAULT_BOOTSTRAP = 1000
DEFAULT_RNG_SEED = 20261001
DEFAULT_WINDOW_FRACTION = 0.2
EXACT_PERMUTATION_LIMIT = 50000
MONTE_CARLO_PERMUTATIONS = 20000
TRIAL_DIR_PATTERN = re.compile(r"^(?P<condition>[A-Za-z0-9]+)_seed_(?P<seed>-?\d+)$")
STAMPED_DIR_PATTERN = re.compile(r"^(?P<condition>[A-Za-z0-9]+)_(?P<stamp>\d{8}_\d{6})$")
LN2 = math.log(2.0)
PAYOFFS = {
    "bar_comfortable": float(UTILITY_AT_BAR_COMFORTABLE),
    "bar_overcrowded": float(UTILITY_AT_BAR_OVERCROWDED),
    "home": float(UTILITY_AT_HOME),
}
CONTRAST_METRICS = (
    ("gini", "final"),
    ("gini", "bar_success"),
    ("drift", "kl_step_mean"),
    ("drift", "kl_total"),
    ("drift", "js_step_mean"),
    ("drift", "agent_drift_mean"),
    ("drift", "entropy_mean"),
    ("nash", "mean_abs_gap"),
    ("nash", "mean_abs_gap_ratio"),
    ("nash", "rmse_gap"),
    ("nash", "mean_signed_gap"),
    ("nash", "late_mean_abs_gap"),
    ("nash", "mean_abs_gap_mixed"),
    ("nash", "equilibrium_hit_rate"),
    ("nash", "mean_regret"),
    ("nash", "late_mean_regret"),
    ("nash", "mean_epsilon"),
    ("nash", "efficiency_mean"),
    ("nash", "welfare_gap_mean"),
    ("behavior", "attendance_mean"),
    ("behavior", "deception_mean"),
    ("behavior", "utility_mean"),
)
DEFINITIONS = {
    "gini.final": "Max-normalised Gini of final cumulative utility: sum_ij |x_i - x_j| / (2 (n - 1) sum_i |x_i|). Bounded in [0, 1] for signed payoffs; equals n/(n-1) times the classic Gini when all payoffs are non-negative.",
    "gini.classic": "Classic Gini sum_ij |x_i - x_j| / (2 n^2 mean(x)); null when any payoff is negative or the mean is non-positive.",
    "gini.bar_success": "Max-normalised Gini of per-agent counts of comfortable-bar epochs (the only outcome paying the top payoff).",
    "gini.trajectory": "Max-normalised Gini of cumulative utility evaluated after every epoch.",
    "drift.strategy_space": "Behavioural strategy = (stated_intention, realised location) in {honest_go, honest_stay, false_go, false_stay}.",
    "drift.smoothing": "Population shares use additive Jeffreys smoothing (alpha = 0.5) so every divergence is finite.",
    "drift.kl_step": "D_KL(p_t || p_{t-1}) in nats between consecutive population strategy distributions.",
    "drift.kl_anchor": "D_KL(p_t || p_0) in nats: cumulative departure from the opening distribution.",
    "drift.kl_stationary": "D_KL(p_t || p_bar) in nats against the time-pooled distribution of the run.",
    "drift.js_step": "Jensen-Shannon divergence between consecutive distributions in bits, bounded in [0, 1].",
    "drift.agent_drift": "Per-agent D_KL(late || early) in nats between that agent's strategy mix in the final and opening windows.",
    "drift.entropy": "Shannon entropy of the smoothed population strategy distribution in bits (max 2).",
    "nash.pure_attendance": "Attendance levels A at which no goer gains by staying home and no stayer gains by going, given the exact payoff rule.",
    "nash.mixed_go_probability": "Symmetric mixed equilibrium p* solving u_c F(T-1; N-1, p) + u_x (1 - F) = u_h with the exact binomial CDF F; p* = 1 when going is dominant.",
    "nash.gap": "Realised attendance minus the nearest pure-equilibrium attendance, per epoch.",
    "nash.regret": "Exact ex-post unilateral deviation gain per agent-epoch; epsilon is the per-epoch maximum, so the profile is an epsilon-Nash equilibrium.",
    "nash.efficiency": "Realised total utility divided by the welfare-maximising total utility.",
    "nash.welfare_gap": "Realised mean utility minus the mean payoff at the pure Nash equilibrium.",
    "nash.late_window": "Late metrics use the final window_fraction of epochs to isolate converged behaviour.",
    "contrast.design": "delta2 minus control. Paired by seed when seeds overlap (common initial positions); unpaired otherwise.",
    "contrast.ci95": "Hierarchical percentile bootstrap: seeds resampled with replacement, then agents (agent-level metrics) or circular blocks of epochs (time-series metrics) within each seed.",
    "contrast.p_value": "Exact two-sided sign-flip test over seed-level paired differences, or exact two-sided permutation test over seed-level values when unpaired.",
    "phase.changepoint": "Single change point minimising the two-segment within-segment sum of squares; r2 is the variance explained by the split.",
    "phase.coupled_transition": "Legacy coupled detector from analyze_results.detect_phase_transition (rolling deception spike against utility crash).",
    "minds.instinct_agreement": "Share of agent-epochs where the LLM's realised strategy matched the argmax of the agent's own bounded-rational softmax instinct.",
    "minds.honesty_lie_spearman": "Spearman rank correlation between each agent's honesty trait and its lie rate; negative means personality drives deception.",
    "minds.reputation_gap_honest_minus_liars": "Final public reputation of agents whose broadcasts were mostly true minus that of agents whose broadcasts were mostly false.",
}


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _seed_key(seed):
    text = str(seed)
    return (0, int(text), text) if text.lstrip("-").isdigit() else (1, 0, text)


def _trial_key(trial):
    order = {CONTROL_LABEL: 0, TREATMENT_LABEL: 1}
    return (order.get(trial["condition"], 2), trial["condition"], _seed_key(trial["seed"]))


def _clean(value, digits=6):
    if isinstance(value, dict):
        return {str(k): _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v, digits) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return round(value, digits)
    return value


def _finite(values):
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _mean(values):
    arr = _finite(values)
    return float(arr.mean()) if arr.size else float("nan")


def _as_bool(series):
    if series.dtype == bool:
        return series.to_numpy()
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["true", "1", "1.0", "yes"]).to_numpy()


def gini(values):
    x = _finite(values)
    n = x.size
    if n < 2:
        return 0.0
    scale = 2.0 * (n - 1) * np.abs(x).sum()
    if scale == 0.0:
        return 0.0
    xs = np.sort(x)
    ranks = np.arange(1, n + 1, dtype=float)
    return float(2.0 * np.sum((2.0 * ranks - n - 1.0) * xs) / scale)


def gini_classic(values):
    x = _finite(values)
    n = x.size
    if n < 2 or np.any(x < 0) or x.sum() <= 0:
        return float("nan")
    xs = np.sort(x)
    ranks = np.arange(1, n + 1, dtype=float)
    return float(2.0 * np.sum((2.0 * ranks - n - 1.0) * xs) / (2.0 * n * xs.sum()))


def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    j = np.arange(1, k + 1, dtype=float)
    log_coeff = np.concatenate(([0.0], np.cumsum(np.log(n - j + 1.0) - np.log(j))))
    support = np.arange(0, k + 1, dtype=float)
    log_terms = log_coeff + support * math.log(p) + (n - support) * math.log1p(-p)
    peak = float(log_terms.max())
    return float(min(1.0, math.exp(peak) * np.exp(log_terms - peak).sum()))


def payoff_go(attendance, threshold, payoffs=PAYOFFS):
    return payoffs["bar_comfortable"] if attendance <= threshold else payoffs["bar_overcrowded"]


def expected_go_payoff(p, num_agents, threshold, payoffs=PAYOFFS):
    f = binomial_cdf(int(threshold) - 1, int(num_agents) - 1, p)
    return payoffs["bar_comfortable"] * f + payoffs["bar_overcrowded"] * (1.0 - f)


def nash_equilibrium(num_agents, threshold, payoffs=PAYOFFS):
    n = int(num_agents)
    t = int(threshold)
    u_home = payoffs["home"]
    pure = []
    for a in range(n + 1):
        goers_hold = a == 0 or payoff_go(a, t, payoffs) >= u_home
        stayers_hold = a == n or payoff_go(a + 1, t, payoffs) <= u_home
        if goers_hold and stayers_hold:
            pure.append(a)
    welfare = [a * payoff_go(a, t, payoffs) + (n - a) * u_home for a in range(n + 1)]
    optimum = int(np.argmax(welfare))
    payoff_all_go = expected_go_payoff(1.0, n, t, payoffs)
    payoff_none_go = expected_go_payoff(0.0, n, t, payoffs)
    if payoff_all_go >= u_home:
        p_star = 1.0
    elif payoff_none_go <= u_home:
        p_star = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if expected_go_payoff(mid, n, t, payoffs) > u_home:
                lo = mid
            else:
                hi = mid
        p_star = 0.5 * (lo + hi)
    if p_star >= 1.0:
        mixed_payoff = payoff_all_go
    else:
        mixed_payoff = u_home
    if pure:
        primary = min(pure, key=lambda a: (abs(a - n * p_star), a))
    else:
        primary = int(round(n * p_star))
    spread = payoffs["bar_comfortable"] - payoffs["bar_overcrowded"]
    worst_go = payoff_go(n, t, payoffs)
    best_go = payoff_go(1, t, payoffs)
    if worst_go > u_home:
        dominant = "go"
    elif best_go < u_home:
        dominant = "stay"
    else:
        dominant = None
    return {
        "num_agents": n,
        "comfort_threshold": t,
        "pure_attendance": pure,
        "primary_attendance": primary,
        "primary_attendance_ratio": primary / n if n else None,
        "pure_mean_payoff": welfare[primary] / n if n else None,
        "mixed_go_probability": p_star,
        "mixed_expected_attendance": n * p_star,
        "mixed_expected_payoff": mixed_payoff,
        "indifference_cdf": (u_home - payoffs["bar_overcrowded"]) / spread if spread else None,
        "social_optimum_attendance": optimum,
        "social_optimum_mean_payoff": welfare[optimum] / n if n else None,
        "congestion_possible": n > t,
        "dominant_strategy": dominant,
    }


def _smooth(counts, alpha=SMOOTHING_ALPHA):
    counts = np.asarray(counts, dtype=float)
    return (counts + alpha) / (counts.sum(axis=-1, keepdims=True) + alpha * counts.shape[-1])


def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def js_divergence_bits(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return (0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)) / LN2


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log(p), axis=-1) / LN2


def strategy_counts(strategy_matrix):
    matrix = np.asarray(strategy_matrix)
    return np.stack([(matrix == k).sum(axis=0) for k in range(len(STRATEGIES))], axis=-1).astype(float)


def changepoint(series, epochs=None):
    y = np.asarray(series, dtype=float)
    n = y.size
    if n < 4 or not np.all(np.isfinite(y)):
        return None
    total_sse = float(np.sum((y - y.mean()) ** 2))
    if total_sse <= 1e-12 * max(1.0, float(np.sum(y * y))):
        return None
    cs = np.cumsum(y)
    cs2 = np.cumsum(y * y)
    best_k = None
    best_sse = None
    for k in range(2, n - 1):
        left_sum = cs[k - 1]
        right_sum = cs[-1] - left_sum
        left_sse = cs2[k - 1] - left_sum * left_sum / k
        right_sse = (cs2[-1] - cs2[k - 1]) - right_sum * right_sum / (n - k)
        sse = left_sse + right_sse
        if best_sse is None or sse < best_sse:
            best_sse = float(sse)
            best_k = k
    best_sse = max(0.0, best_sse)
    before = y[:best_k]
    after = y[best_k:]
    pooled = math.sqrt(best_sse / max(n - 2, 1))
    epoch_axis = np.arange(n) if epochs is None else np.asarray(epochs)
    return {
        "index": int(best_k),
        "epoch": int(epoch_axis[best_k]),
        "mean_before": float(before.mean()),
        "mean_after": float(after.mean()),
        "shift": float(after.mean() - before.mean()),
        "effect_size": float((after.mean() - before.mean()) / pooled) if pooled > 0 else None,
        "r2": float(min(1.0, max(0.0, 1.0 - best_sse / total_sse))),
    }


def _coupled_transition(epochs, deception, utility, window=5):
    try:
        from analyze_results import detect_phase_transition
    except ImportError:
        return None
    frame = pd.DataFrame({"epoch": epochs, "deception_index": deception, "mean_utility": utility})
    frame = frame.dropna()
    if len(frame) < 2:
        return None
    result = detect_phase_transition(frame.reset_index(drop=True), window=window)
    return {k: result[k] for k in ("transition_epoch", "deception_spike", "utility_crash", "score") if k in result}


def _block_bootstrap_means(series, rng, replicates, block=None):
    x = _finite(series)
    n = x.size
    if n == 0:
        return np.full(replicates, np.nan)
    if n == 1:
        return np.full(replicates, x[0])
    length = block or max(1, int(round(n ** (1.0 / 3.0))))
    blocks = int(math.ceil(n / length))
    starts = rng.integers(0, n, size=(replicates, blocks))
    idx = (starts[:, :, None] + np.arange(length)[None, None, :]) % n
    idx = idx.reshape(replicates, -1)[:, :n]
    return x[idx].mean(axis=1)


def _agent_bootstrap(values, statistic, rng, replicates):
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return np.full(replicates, np.nan)
    idx = rng.integers(0, n, size=(replicates, n))
    return np.array([statistic(x[row]) for row in idx], dtype=float)


def discover_trials(experiment_dir):
    comparison = _read_json(os.path.join(experiment_dir, "comparison.json"))
    comparison = comparison if isinstance(comparison, list) else []
    rows_by_dir = {row["trial_dir"]: row for row in comparison if isinstance(row, dict) and row.get("trial_dir")}
    trials = []
    if not os.path.isdir(experiment_dir):
        return trials, comparison
    for name in sorted(os.listdir(experiment_dir)):
        path = os.path.join(experiment_dir, name)
        epoch_csv = os.path.join(path, "epoch_metrics.csv")
        agent_csv = os.path.join(path, "agent_epoch_details.csv")
        if not (os.path.isdir(path) and os.path.isfile(epoch_csv) and os.path.isfile(agent_csv)):
            continue
        manifest = _read_json(os.path.join(path, "trial_manifest.json")) or {}
        condition = manifest.get("condition")
        seed = manifest.get("seed")
        match = TRIAL_DIR_PATTERN.match(name)
        stamped = STAMPED_DIR_PATTERN.match(name)
        if condition is None:
            condition = match.group("condition") if match else (stamped.group("condition") if stamped else None)
        if seed is None:
            seed = int(match.group("seed")) if match else (stamped.group("stamp") if stamped else name)
        if condition is None:
            continue
        row = rows_by_dir.get(name, {})
        runtime = manifest.get("runtime") or row.get("runtime") or {}
        trials.append({
            "trial": name,
            "condition": str(condition),
            "seed": seed,
            "path": path,
            "epoch_csv": epoch_csv,
            "agent_csv": agent_csv,
            "status": manifest.get("status") or row.get("status") or "complete",
            "fallback_rate": row.get("regex_fallback_rate", runtime.get("fallback_rate")),
            "model_name": manifest.get("model_name"),
            "grid_size": manifest.get("grid_size"),
            "runtime": runtime,
        })
    trials.sort(key=_trial_key)
    return trials, comparison


def load_trial(trial):
    epochs_df = pd.read_csv(trial["epoch_csv"])
    agents_df = pd.read_csv(trial["agent_csv"])
    required = ("epoch", "agent_id", "utility", "stated_intention")
    missing = [c for c in required if c not in agents_df.columns]
    if missing:
        raise ValueError(f"{trial['agent_csv']} is missing columns {missing}")
    if agents_df.empty:
        raise ValueError(f"{trial['agent_csv']} has no agent rows")
    agent_ids = np.sort(agents_df["agent_id"].unique())
    epoch_values = np.sort(agents_df["epoch"].unique())
    a_idx = np.searchsorted(agent_ids, agents_df["agent_id"].to_numpy())
    e_idx = np.searchsorted(epoch_values, agents_df["epoch"].to_numpy())
    shape = (agent_ids.size, epoch_values.size)
    if "in_bar" in agents_df.columns:
        at_bar_flat = _as_bool(agents_df["in_bar"])
    else:
        at_bar_flat = agents_df["actual_location"].astype(str).str.strip().str.lower().eq("bar").to_numpy()
    stated_go_flat = agents_df["stated_intention"].astype(str).str.strip().str.lower().eq("going").to_numpy()
    codes_flat = np.where(stated_go_flat, np.where(at_bar_flat, 0, 2), np.where(at_bar_flat, 3, 1))
    strategy = np.full(shape, -1, dtype=np.int8)
    strategy[a_idx, e_idx] = codes_flat
    in_bar = np.zeros(shape, dtype=bool)
    in_bar[a_idx, e_idx] = at_bar_flat
    utility = np.full(shape, np.nan)
    utility[a_idx, e_idx] = pd.to_numeric(agents_df["utility"], errors="coerce").to_numpy(dtype=float)
    if "cumulative_utility" in agents_df.columns:
        cumulative = np.full(shape, np.nan)
        cumulative[a_idx, e_idx] = pd.to_numeric(agents_df["cumulative_utility"], errors="coerce").to_numpy(dtype=float)
        gaps = ~np.isfinite(cumulative)
        if gaps.any():
            cumulative[gaps] = np.nancumsum(utility, axis=1)[gaps]
    else:
        cumulative = np.nancumsum(utility, axis=1)
    capacity = 100
    if "bar_capacity" in epochs_df.columns and epochs_df["bar_capacity"].notna().any():
        capacity = int(pd.to_numeric(epochs_df["bar_capacity"], errors="coerce").dropna().iloc[0])
    if "comfort_threshold" in epochs_df.columns and epochs_df["comfort_threshold"].notna().any():
        threshold = int(pd.to_numeric(epochs_df["comfort_threshold"], errors="coerce").dropna().iloc[0])
    else:
        threshold = int(COMFORT_THRESHOLD_RATIO * capacity)
    logged_attendance = None
    if "bar_attendance" in epochs_df.columns and "epoch" in epochs_df.columns:
        logged = epochs_df.set_index("epoch")["bar_attendance"]
        logged_attendance = np.array([float(logged.get(e, np.nan)) for e in epoch_values])
    return {
        "trial": trial,
        "agent_ids": agent_ids,
        "epochs": epoch_values,
        "strategy": strategy,
        "in_bar": in_bar,
        "utility": utility,
        "cumulative": cumulative,
        "capacity": capacity,
        "threshold": threshold,
        "logged_attendance": logged_attendance,
    }


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = np.asarray(values, dtype=float)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    rx = _rank(x[mask])
    ry = _rank(y[mask])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_minds(trial):
    trace_path = os.path.join(trial["path"], "mind_trace.csv")
    minds_path = os.path.join(trial["path"], "minds.json")
    if not (os.path.isfile(trace_path) and os.path.isfile(minds_path)):
        return None
    trace = pd.read_csv(trace_path)
    minds = _read_json(minds_path) or {}
    if trace.empty or not minds:
        return None
    lied = _as_bool(trace["lied"]).astype(float)
    trace = trace.assign(lied_flag=lied)
    stated_go = trace["stated_intention"].astype(str).str.lower().eq("going")
    in_bar = _as_bool(trace["in_bar"])
    realised = np.where(stated_go, np.where(in_bar, "honest_go", "false_go"), np.where(in_bar, "false_stay", "honest_stay"))
    agreement = float(np.mean(realised == trace["instinct"].astype(str).to_numpy()))
    per_agent = trace.groupby("agent_id")["lied_flag"].mean()
    honesty = pd.Series({int(k): v["traits"]["honesty"] for k, v in minds.items()})
    aligned = per_agent.reindex(honesty.index)
    by_archetype = trace.groupby("archetype")["lied_flag"].agg(["mean", "count"])
    public = pd.Series({int(k): (v["public_lies"] / v["public_claims"]) if v.get("public_claims") else np.nan for k, v in minds.items()})
    final_epoch = trace["epoch"].max()
    reputations = trace[trace["epoch"] == final_epoch].set_index("agent_id")["reputation"]
    liars = public[public > 0.5].index
    honest = public[public <= 0.5].index
    trust_gap = None
    if len(liars) and len(honest):
        trust_gap = float(reputations.reindex(honest).mean() - reputations.reindex(liars).mean())
    return {
        "agents": int(len(minds)),
        "lie_rate": float(lied.mean()),
        "instinct_agreement": agreement,
        "reroute_rate": float(_as_bool(trace["rerouted"]).mean()) if "rerouted" in trace else None,
        "honesty_lie_spearman": spearman(honesty.to_numpy(), aligned.to_numpy()),
        "public_liars": int(len(liars)),
        "reputation_gap_honest_minus_liars": trust_gap,
        "mean_trust_given_final": float(trace[trace["epoch"] == final_epoch]["trust_given_mean"].mean()),
        "notes_written": float(trace["private_note"].fillna("").astype(str).str.len().gt(0).mean()),
        "lie_rate_by_archetype": {str(k): {"lie_rate": float(r["mean"]), "agent_epochs": int(r["count"])} for k, r in by_archetype.iterrows()},
        "lie_rate_series": trace.groupby("epoch")["lied_flag"].mean().tolist(),
    }


def _window(n_epochs, fraction):
    return max(1, min(n_epochs, int(round(n_epochs * fraction))))


def analyze_trial(data, rng, replicates=DEFAULT_BOOTSTRAP, window_fraction=DEFAULT_WINDOW_FRACTION, payoffs=PAYOFFS):
    strategy = data["strategy"]
    in_bar = data["in_bar"]
    utility = data["utility"]
    cumulative = data["cumulative"]
    epochs = data["epochs"]
    n_agents, n_epochs = strategy.shape
    ne = nash_equilibrium(n_agents, data["threshold"], payoffs)
    window = _window(n_epochs, window_fraction)
    threshold = ne["comfort_threshold"]
    u_c = payoffs["bar_comfortable"]
    u_x = payoffs["bar_overcrowded"]
    u_h = payoffs["home"]

    attendance = in_bar.sum(axis=0).astype(float)
    deception = np.isin(strategy, (2, 3)).mean(axis=0)
    mean_utility = np.nanmean(np.where(np.isfinite(utility), utility, np.nan), axis=0)

    final_cumulative = cumulative[:, -1]
    wins = np.sum(np.isclose(np.nan_to_num(utility, nan=-np.inf), u_c), axis=1).astype(float)
    gini_trajectory = np.array([gini(cumulative[:, t]) for t in range(n_epochs)])

    counts = strategy_counts(strategy)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    p = _smooth(counts)
    pooled = _smooth(counts.sum(axis=0))
    kl_step = np.full(n_epochs, np.nan)
    js_step = np.full(n_epochs, np.nan)
    if n_epochs > 1:
        kl_step[1:] = kl_divergence(p[1:], p[:-1])
        js_step[1:] = js_divergence_bits(p[1:], p[:-1])
    kl_anchor = kl_divergence(p, p[0][None, :])
    kl_stationary = kl_divergence(p, pooled[None, :])
    entropy = entropy_bits(p)
    early_counts = np.stack([(strategy[:, :window] == k).sum(axis=1) for k in range(len(STRATEGIES))], axis=-1)
    late_counts = np.stack([(strategy[:, -window:] == k).sum(axis=1) for k in range(len(STRATEGIES))], axis=-1)
    agent_drift = kl_divergence(_smooth(late_counts), _smooth(early_counts))
    finite_steps = kl_step[np.isfinite(kl_step)]
    peak_index = int(np.nanargmax(kl_step)) if finite_steps.size else None

    pure = np.array(ne["pure_attendance"] or [ne["primary_attendance"]], dtype=float)
    nearest = pure[np.argmin(np.abs(attendance[:, None] - pure[None, :]), axis=1)]
    gap = attendance - nearest
    gap_mixed = attendance - ne["mixed_expected_attendance"]
    go_now = np.where(attendance <= threshold, u_c, u_x)
    go_deviation = np.where(attendance + 1 <= threshold, u_c, u_x)
    gain = np.where(in_bar, np.maximum(0.0, u_h - go_now[None, :]), np.maximum(0.0, go_deviation[None, :] - u_h))
    regret = gain.mean(axis=0)
    epsilon = gain.max(axis=0)
    welfare = np.nansum(utility, axis=0)
    optimum_welfare = ne["social_optimum_mean_payoff"] * n_agents
    efficiency = welfare / optimum_welfare if optimum_welfare > 0 else np.full(n_epochs, np.nan)
    welfare_gap = (welfare - ne["pure_mean_payoff"] * n_agents) / n_agents
    at_equilibrium = np.isin(attendance, pure).astype(float)

    deception_cp = changepoint(deception, epochs)
    attendance_cp = changepoint(attendance, epochs)
    utility_cp = changepoint(mean_utility, epochs)

    def block(series):
        return _block_bootstrap_means(series, rng, replicates)

    metrics = {
        ("gini", "final"): (gini(final_cumulative), _agent_bootstrap(final_cumulative, gini, rng, replicates)),
        ("gini", "bar_success"): (gini(wins), _agent_bootstrap(wins, gini, rng, replicates)),
        ("drift", "kl_step_mean"): (_mean(kl_step), block(kl_step)),
        ("drift", "kl_total"): (float(np.nansum(kl_step)), block(kl_step) * max(finite_steps.size, 1)),
        ("drift", "js_step_mean"): (_mean(js_step), block(js_step)),
        ("drift", "agent_drift_mean"): (_mean(agent_drift), _agent_bootstrap(agent_drift, _mean, rng, replicates)),
        ("drift", "entropy_mean"): (_mean(entropy), block(entropy)),
        ("nash", "mean_abs_gap"): (_mean(np.abs(gap)), block(np.abs(gap))),
        ("nash", "mean_abs_gap_ratio"): (_mean(np.abs(gap)) / n_agents, block(np.abs(gap)) / n_agents),
        ("nash", "rmse_gap"): (math.sqrt(_mean(gap ** 2)), np.sqrt(block(gap ** 2))),
        ("nash", "mean_signed_gap"): (_mean(gap), block(gap)),
        ("nash", "late_mean_abs_gap"): (_mean(np.abs(gap[-window:])), block(np.abs(gap[-window:]))),
        ("nash", "mean_abs_gap_mixed"): (_mean(np.abs(gap_mixed)), block(np.abs(gap_mixed))),
        ("nash", "equilibrium_hit_rate"): (_mean(at_equilibrium), block(at_equilibrium)),
        ("nash", "mean_regret"): (_mean(regret), block(regret)),
        ("nash", "late_mean_regret"): (_mean(regret[-window:]), block(regret[-window:])),
        ("nash", "mean_epsilon"): (_mean(epsilon), block(epsilon)),
        ("nash", "efficiency_mean"): (_mean(efficiency), block(efficiency)),
        ("nash", "welfare_gap_mean"): (_mean(welfare_gap), block(welfare_gap)),
        ("behavior", "attendance_mean"): (_mean(attendance), block(attendance)),
        ("behavior", "deception_mean"): (_mean(deception), block(deception)),
        ("behavior", "utility_mean"): (_mean(mean_utility), block(mean_utility)),
    }

    attendance_mismatch = None
    if data["logged_attendance"] is not None:
        diff = np.abs(data["logged_attendance"] - attendance)
        diff = diff[np.isfinite(diff)]
        attendance_mismatch = int(np.sum(diff > 0)) if diff.size else 0

    trial = data["trial"]
    summary = {
        "trial": trial["trial"],
        "condition": trial["condition"],
        "seed": trial["seed"],
        "status": trial["status"],
        "model_name": trial.get("model_name"),
        "fallback_rate": trial.get("fallback_rate"),
        "grid_size": trial.get("grid_size"),
        "agents": n_agents,
        "epochs": n_epochs,
        "window": window,
        "game": {"bar_capacity": data["capacity"], "comfort_threshold": threshold},
        "nash_reference": ne,
        "gini": {
            "final": metrics[("gini", "final")][0],
            "classic": gini_classic(final_cumulative),
            "bar_success": metrics[("gini", "bar_success")][0],
            "final_cumulative_utility": {
                "mean": _mean(final_cumulative),
                "min": float(np.nanmin(final_cumulative)),
                "max": float(np.nanmax(final_cumulative)),
            },
        },
        "drift": {
            "kl_step_mean": metrics[("drift", "kl_step_mean")][0],
            "kl_step_max": float(finite_steps.max()) if finite_steps.size else None,
            "kl_step_peak_epoch": int(epochs[peak_index]) if peak_index is not None else None,
            "kl_total": metrics[("drift", "kl_total")][0],
            "kl_anchor_final": float(kl_anchor[-1]),
            "kl_stationary_mean": _mean(kl_stationary),
            "js_step_mean": metrics[("drift", "js_step_mean")][0],
            "entropy_mean": metrics[("drift", "entropy_mean")][0],
            "agent_drift_mean": metrics[("drift", "agent_drift_mean")][0],
            "agent_drift_median": float(np.median(agent_drift)),
            "agent_drift_p90": float(np.percentile(agent_drift, 90)),
            "strategy_shares_mean": {name: float(shares[:, k].mean()) for k, name in enumerate(STRATEGIES)},
        },
        "nash": {
            "mean_abs_gap": metrics[("nash", "mean_abs_gap")][0],
            "mean_abs_gap_ratio": metrics[("nash", "mean_abs_gap_ratio")][0],
            "rmse_gap": metrics[("nash", "rmse_gap")][0],
            "mean_signed_gap": metrics[("nash", "mean_signed_gap")][0],
            "late_mean_abs_gap": metrics[("nash", "late_mean_abs_gap")][0],
            "mean_abs_gap_mixed": metrics[("nash", "mean_abs_gap_mixed")][0],
            "equilibrium_hit_rate": metrics[("nash", "equilibrium_hit_rate")][0],
            "mean_regret": metrics[("nash", "mean_regret")][0],
            "late_mean_regret": metrics[("nash", "late_mean_regret")][0],
            "mean_epsilon": metrics[("nash", "mean_epsilon")][0],
            "efficiency_mean": metrics[("nash", "efficiency_mean")][0],
            "welfare_gap_mean": metrics[("nash", "welfare_gap_mean")][0],
        },
        "behavior": {
            "attendance_mean": metrics[("behavior", "attendance_mean")][0],
            "deception_mean": metrics[("behavior", "deception_mean")][0],
            "utility_mean": metrics[("behavior", "utility_mean")][0],
            "attendance_log_mismatches": attendance_mismatch,
        },
        "phase": {
            "changepoints": {
                "deception_index": deception_cp,
                "bar_attendance": attendance_cp,
                "mean_utility": utility_cp,
            },
            "coupled_transition": _coupled_transition(epochs, deception, mean_utility),
        },
        "series": {
            "epoch": epochs,
            "bar_attendance": attendance,
            "deception_index": deception,
            "mean_utility": mean_utility,
            "nash_gap": gap,
            "nash_gap_mixed": gap_mixed,
            "regret": regret,
            "epsilon": epsilon,
            "efficiency": efficiency,
            "kl_step": kl_step,
            "kl_anchor": kl_anchor,
            "js_step": js_step,
            "entropy": entropy,
            "gini": gini_trajectory,
            "strategy_shares": {name: shares[:, k] for k, name in enumerate(STRATEGIES)},
        },
    }
    replicates_by_metric = {key: {"value": value, "replicates": reps} for key, (value, reps) in metrics.items()}
    return summary, replicates_by_metric, counts


def _sign_flip_p(diffs, rng):
    d = _finite(diffs)
    k = d.size
    if k == 0:
        return None, None
    observed = abs(d.mean())
    if k <= 16:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=k)))
        method = f"exact sign-flip ({signs.shape[0]} relabelings)"
    else:
        signs = rng.choice((1.0, -1.0), size=(MONTE_CARLO_PERMUTATIONS, k))
        method = f"Monte Carlo sign-flip ({MONTE_CARLO_PERMUTATIONS} relabelings)"
    stats = np.abs((signs * d[None, :]).mean(axis=1))
    return float(np.mean(stats >= observed - 1e-12)), method


def _permutation_p(control_values, treatment_values, rng):
    a = _finite(control_values)
    b = _finite(treatment_values)
    if a.size == 0 or b.size == 0:
        return None, None
    pooled = np.concatenate([a, b])
    n = pooled.size
    observed = abs(b.mean() - a.mean())
    total = math.comb(n, b.size)
    if total <= EXACT_PERMUTATION_LIMIT:
        stats = []
        for combo in itertools.combinations(range(n), b.size):
            mask = np.zeros(n, dtype=bool)
            mask[list(combo)] = True
            stats.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
        stats = np.array(stats)
        method = f"exact permutation ({total} relabelings)"
    else:
        stats = np.empty(MONTE_CARLO_PERMUTATIONS)
        for i in range(MONTE_CARLO_PERMUTATIONS):
            perm = rng.permutation(n)
            stats[i] = abs(pooled[perm[: b.size]].mean() - pooled[perm[b.size:]].mean())
        method = f"Monte Carlo permutation ({MONTE_CARLO_PERMUTATIONS} relabelings)"
    return float(np.mean(stats >= observed - 1e-12)), method


def _seed_level_resample(entries, rng, replicates):
    seeds = list(entries)
    stacked = np.stack([entries[s]["replicates"] for s in seeds])
    if len(seeds) == 1:
        return stacked[0]
    pick = rng.integers(0, len(seeds), size=(replicates, len(seeds)))
    return np.nanmean(stacked[pick, np.arange(replicates)[:, None]], axis=1)


def _describe(entries):
    values = _finite([e["value"] for e in entries.values()])
    return {
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "by_seed": {str(seed): entry["value"] for seed, entry in entries.items()},
    }


def contrast(control_entries, treatment_entries, rng, replicates):
    if not control_entries or not treatment_entries:
        return None
    shared = [s for s in treatment_entries if s in control_entries]
    if shared:
        diffs = {s: treatment_entries[s]["value"] - control_entries[s]["value"] for s in shared}
        estimate = _mean(list(diffs.values()))
        stacked = np.stack([treatment_entries[s]["replicates"] - control_entries[s]["replicates"] for s in shared])
        if len(shared) > 1:
            pick = rng.integers(0, len(shared), size=(replicates, len(shared)))
            boot = np.nanmean(stacked[pick, np.arange(replicates)[:, None]], axis=1)
        else:
            boot = stacked[0]
        p_value, method = _sign_flip_p(list(diffs.values()), rng)
        design = "paired"
        by_seed_diff = {str(s): d for s, d in diffs.items()}
    else:
        estimate = _mean([e["value"] for e in treatment_entries.values()]) - _mean([e["value"] for e in control_entries.values()])
        boot = _seed_level_resample(treatment_entries, rng, replicates) - _seed_level_resample(control_entries, rng, replicates)
        p_value, method = _permutation_p(
            [e["value"] for e in control_entries.values()],
            [e["value"] for e in treatment_entries.values()],
            rng,
        )
        design = "unpaired"
        by_seed_diff = None
    finite_boot = _finite(boot)
    ci = [float(np.percentile(finite_boot, 2.5)), float(np.percentile(finite_boot, 97.5))] if finite_boot.size else [None, None]
    return {
        CONTROL_LABEL: _describe(control_entries),
        TREATMENT_LABEL: _describe(treatment_entries),
        "estimate": estimate,
        "design": design,
        "ci95": ci,
        "p_value": p_value,
        "p_method": method,
        "by_seed_difference": by_seed_diff,
    }


def _stack_series(trial_summaries, key):
    epochs = sorted({int(e) for s in trial_summaries for e in s["series"]["epoch"]})
    position = {e: i for i, e in enumerate(epochs)}
    grid = np.full((len(trial_summaries), len(epochs)), np.nan)
    for row, summary in enumerate(trial_summaries):
        values = summary["series"]
        source = values["strategy_shares"][key] if key in STRATEGIES else values[key]
        for e, v in zip(values["epoch"], source):
            grid[row, position[int(e)]] = v
    return epochs, grid


def _series_stats(grid):
    grid = np.asarray(grid, dtype=float)
    finite = np.isfinite(grid)
    counts = finite.sum(axis=0)
    filled = np.where(finite, grid, 0.0)
    totals = filled.sum(axis=0)
    mean = np.full(grid.shape[1], np.nan)
    np.divide(totals, counts, out=mean, where=counts > 0)
    squares = np.where(finite, (grid - mean[None, :]) ** 2, 0.0).sum(axis=0)
    sd = np.full(grid.shape[1], np.nan)
    np.divide(squares, counts - 1, out=sd, where=counts > 1)
    return mean, np.sqrt(sd)


def condition_series(trial_summaries):
    if not trial_summaries:
        return None
    keys = (
        "bar_attendance",
        "deception_index",
        "mean_utility",
        "nash_gap",
        "nash_gap_mixed",
        "regret",
        "epsilon",
        "efficiency",
        "kl_step",
        "js_step",
        "entropy",
        "gini",
    )
    epochs, _ = _stack_series(trial_summaries, "bar_attendance")
    series = {"epoch": epochs}
    for key in keys:
        _, grid = _stack_series(trial_summaries, key)
        mean, sd = _series_stats(grid)
        series[f"{key}_mean"] = mean
        series[f"{key}_sd"] = sd
    shares = {}
    for name in STRATEGIES:
        _, grid = _stack_series(trial_summaries, name)
        shares[name], _ = _series_stats(grid)
    series["strategy_shares"] = shares
    return series


def _pooled_divergence(counts_by_condition):
    control = counts_by_condition.get(CONTROL_LABEL)
    treatment = counts_by_condition.get(TREATMENT_LABEL)
    if not control or not treatment:
        return None
    length = min(min(c.shape[0] for c in control), min(c.shape[0] for c in treatment))
    if length == 0:
        return None
    control_counts = np.sum([c[:length] for c in control], axis=0)
    treatment_counts = np.sum([c[:length] for c in treatment], axis=0)
    p_t = _smooth(treatment_counts)
    p_c = _smooth(control_counts)
    overall_t = _smooth(treatment_counts.sum(axis=0))
    overall_c = _smooth(control_counts.sum(axis=0))
    return {
        "kl_treatment_vs_control_per_epoch": kl_divergence(p_t, p_c),
        "js_bits_per_epoch": js_divergence_bits(p_t, p_c),
        "kl_treatment_vs_control_pooled": float(kl_divergence(overall_t, overall_c)),
        "kl_control_vs_treatment_pooled": float(kl_divergence(overall_c, overall_t)),
        "js_bits_pooled": float(js_divergence_bits(overall_t, overall_c)),
    }


def _phase_summary(series):
    if series is None:
        return None
    epochs = np.asarray(series["epoch"])
    deception = np.asarray(series["deception_index_mean"], dtype=float)
    utility = np.asarray(series["mean_utility_mean"], dtype=float)
    attendance = np.asarray(series["bar_attendance_mean"], dtype=float)
    return {
        "changepoints": {
            "deception_index": changepoint(deception, epochs),
            "bar_attendance": changepoint(attendance, epochs),
            "mean_utility": changepoint(utility, epochs),
            "nash_gap": changepoint(np.asarray(series["nash_gap_mean"], dtype=float), epochs),
            "kl_step": changepoint(np.nan_to_num(np.asarray(series["kl_step_mean"], dtype=float), nan=0.0)[1:], epochs[1:]) if len(epochs) > 4 else None,
        },
        "coupled_transition": _coupled_transition(epochs, deception, utility),
    }


def build_report(experiment_dir, bootstrap=DEFAULT_BOOTSTRAP, rng_seed=DEFAULT_RNG_SEED, window_fraction=DEFAULT_WINDOW_FRACTION):
    rng = np.random.default_rng(rng_seed)
    trials, comparison = discover_trials(experiment_dir)
    warnings = []
    trial_summaries = []
    replicate_index = {}
    counts_by_condition = {}
    references = {}
    for trial in trials:
        if trial["status"] != "complete":
            warnings.append(f"{trial['trial']}: status '{trial['status']}' excluded from inference")
            continue
        try:
            data = load_trial(trial)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            warnings.append(f"{trial['trial']}: unreadable trace ({exc})")
            continue
        summary, replicates_by_metric, counts = analyze_trial(data, rng, bootstrap, window_fraction)
        try:
            summary["minds"] = analyze_minds(trial)
        except (ValueError, KeyError, OSError) as exc:
            summary["minds"] = None
            warnings.append(f"{trial['trial']}: mind trace unreadable ({exc})")
        trial_summaries.append(summary)
        replicate_index.setdefault(trial["condition"], {})[trial["seed"]] = replicates_by_metric
        counts_by_condition.setdefault(trial["condition"], []).append(counts)
        references[(summary["agents"], summary["game"]["comfort_threshold"])] = summary["nash_reference"]
        if summary["behavior"]["attendance_log_mismatches"]:
            warnings.append(f"{trial['trial']}: {summary['behavior']['attendance_log_mismatches']} epochs where agent-level attendance disagrees with epoch_metrics.csv")
        fallback = trial.get("fallback_rate")
        if isinstance(fallback, (int, float)) and fallback > 0.1:
            warnings.append(
                f"{trial['trial']}: LLM parse fallback rate {fallback:.1%}; fallback actions declare 'staying' in place, which registers as false_stay for agents already inside the bar"
            )

    for (n_agents, threshold), reference in references.items():
        if reference["dominant_strategy"] == "go":
            warnings.append(
                f"comfort threshold {threshold} >= population {n_agents}: the bar can never be overcrowded, attending strictly dominates staying home "
                f"({PAYOFFS['bar_comfortable']} > {PAYOFFS['home']}), and the unique Nash equilibrium is full attendance (A* = {n_agents}). "
                "Deviation from Nash here measures failure to exploit a dominant strategy, not El Farol coordination failure."
            )
    if len(references) > 1:
        warnings.append("trials disagree on population or threshold; Nash references are reported per trial")

    conditions = {}
    for label in sorted({s["condition"] for s in trial_summaries}):
        members = [s for s in trial_summaries if s["condition"] == label]
        series = condition_series(members)
        aggregate = {}
        for section, metric in CONTRAST_METRICS:
            values = _finite([m[section].get(metric) if m[section].get(metric) is not None else np.nan for m in members])
            aggregate.setdefault(section, {})[metric] = {
                "mean": float(values.mean()) if values.size else None,
                "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                "n": int(values.size),
            }
        mind_blocks = [m["minds"] for m in members if m.get("minds")]
        minds_aggregate = None
        if mind_blocks:
            minds_aggregate = {}
            for key in ("lie_rate", "instinct_agreement", "reroute_rate", "honesty_lie_spearman", "reputation_gap_honest_minus_liars", "mean_trust_given_final"):
                values = _finite([b.get(key) if b.get(key) is not None else np.nan for b in mind_blocks])
                minds_aggregate[key] = {
                    "mean": float(values.mean()) if values.size else None,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                    "n": int(values.size),
                }
            archetypes = {}
            for block in mind_blocks:
                for name, stats in block["lie_rate_by_archetype"].items():
                    bucket = archetypes.setdefault(name, [0.0, 0])
                    bucket[0] += stats["lie_rate"] * stats["agent_epochs"]
                    bucket[1] += stats["agent_epochs"]
            minds_aggregate["lie_rate_by_archetype"] = {k: v[0] / v[1] for k, v in sorted(archetypes.items()) if v[1]}
        conditions[label] = {
            "trials": [m["trial"] for m in members],
            "seeds": [m["seed"] for m in members],
            "metrics": aggregate,
            "series": series,
            "phase": _phase_summary(series),
            "minds": minds_aggregate,
        }

    contrast_block = None
    if CONTROL_LABEL in replicate_index and TREATMENT_LABEL in replicate_index:
        control_reps = replicate_index[CONTROL_LABEL]
        treatment_reps = replicate_index[TREATMENT_LABEL]
        metrics = {}
        for section, metric in CONTRAST_METRICS:
            control_entries = {seed: reps[(section, metric)] for seed, reps in control_reps.items()}
            treatment_entries = {seed: reps[(section, metric)] for seed, reps in treatment_reps.items()}
            metrics.setdefault(section, {})[metric] = contrast(control_entries, treatment_entries, rng, bootstrap)
        per_epoch = None
        control_series = conditions[CONTROL_LABEL]["series"]
        treatment_series = conditions[TREATMENT_LABEL]["series"]
        shared_epochs = sorted(set(control_series["epoch"]) & set(treatment_series["epoch"]))
        if shared_epochs:
            c_pos = {e: i for i, e in enumerate(control_series["epoch"])}
            t_pos = {e: i for i, e in enumerate(treatment_series["epoch"])}
            per_epoch = {"epoch": shared_epochs}
            for key in ("bar_attendance", "deception_index", "mean_utility", "nash_gap", "regret", "efficiency", "gini", "kl_step", "entropy"):
                c_vals = np.asarray(control_series[f"{key}_mean"], dtype=float)
                t_vals = np.asarray(treatment_series[f"{key}_mean"], dtype=float)
                per_epoch[f"{key}_diff"] = [t_vals[t_pos[e]] - c_vals[c_pos[e]] for e in shared_epochs]
        divergence = _pooled_divergence(counts_by_condition)
        shared_seeds = sorted(set(control_reps) & set(treatment_reps), key=_seed_key)
        contrast_block = {
            "label": f"{TREATMENT_LABEL} - {CONTROL_LABEL}",
            "paired_seeds": [s for s in shared_seeds],
            "metrics": metrics,
            "per_epoch": per_epoch,
            "strategy_divergence": divergence,
        }
        if len(shared_seeds) < 6:
            warnings.append(
                f"{len(shared_seeds)} paired seeds: the smallest attainable two-sided sign-flip p-value is {2.0 / (2 ** len(shared_seeds)):.3f}" if shared_seeds else
                "no paired seeds: contrasts fall back to unpaired permutation tests"
            )
    else:
        warnings.append(f"contrast requires both '{CONTROL_LABEL}' and '{TREATMENT_LABEL}' trials; found {sorted(replicate_index)}")

    game = None
    if references:
        (n_agents, threshold), reference = next(iter(references.items()))
        game = {
            "num_agents": n_agents,
            "bar_capacity": trial_summaries[0]["game"]["bar_capacity"] if trial_summaries else None,
            "comfort_threshold": threshold,
            "payoffs": PAYOFFS,
            "nash": reference,
        }

    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "experiment_dir": os.path.abspath(experiment_dir),
            "comparison_rows": len(comparison),
            "trials_discovered": len(trials),
            "trials_analyzed": len(trial_summaries),
        },
        "parameters": {
            "bootstrap_replicates": bootstrap,
            "rng_seed": rng_seed,
            "window_fraction": window_fraction,
            "smoothing_alpha": SMOOTHING_ALPHA,
            "strategies": list(STRATEGIES),
            "strategy_definitions": STRATEGY_DEFINITIONS,
        },
        "game": game,
        "definitions": DEFINITIONS,
        "conditions": conditions,
        "contrast": contrast_block,
        "trials": trial_summaries,
        "warnings": warnings,
    }
    return _clean(report)


def write_report(report, output_path):
    directory = os.path.dirname(output_path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{output_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    os.replace(tmp_path, output_path)
    return output_path


def _fmt(value, digits=4):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if isinstance(value, float) else str(value)


def print_digest(report):
    print("EL FAROL // ANALYTICS DIGEST")
    print("=" * 78)
    source = report["source"]
    print(f"trials analysed : {source['trials_analyzed']}/{source['trials_discovered']}  ::  {source['experiment_dir']}")
    game = report.get("game")
    if game:
        nash = game["nash"]
        print(
            f"game            : N={game['num_agents']}  T={game['comfort_threshold']}  "
            f"A*={nash['pure_attendance']}  p*={nash['mixed_go_probability']:.6f}  dominant={nash['dominant_strategy']}"
        )
    contrast_block = report.get("contrast")
    if contrast_block:
        print("-" * 78)
        print(f"{'metric':<34}{'control':>10}{'delta2':>10}{'delta':>10}  {'ci95':<21}{'p':>6}")
        for section in ("gini", "drift", "nash", "behavior"):
            for metric, block in (contrast_block["metrics"].get(section) or {}).items():
                if block is None:
                    continue
                lo, hi = block["ci95"]
                ci = f"[{_fmt(lo, 3)}, {_fmt(hi, 3)}]"
                p = block["p_value"]
                print(
                    f"{section + '.' + metric:<34}"
                    f"{_fmt(block[CONTROL_LABEL]['mean']):>10}"
                    f"{_fmt(block[TREATMENT_LABEL]['mean']):>10}"
                    f"{_fmt(block['estimate']):>10}  {ci:<21}"
                    f"{(f'{p:.3f}' if p is not None else '—'):>6}"
                )
    for label, block in (report.get("conditions") or {}).items():
        minds = block.get("minds")
        if not minds:
            continue
        print("-" * 78)
        print(
            f"minds // {label:<8} lie {_fmt(minds['lie_rate']['mean'])}  instinct-agree {_fmt(minds['instinct_agreement']['mean'])}  "
            f"honesty~lie rho {_fmt(minds['honesty_lie_spearman']['mean'])}  reputation gap {_fmt(minds['reputation_gap_honest_minus_liars']['mean'])}"
        )
        print("   by archetype: " + "  ".join(f"{k} {v:.2f}" for k, v in minds["lie_rate_by_archetype"].items()))
    for warning in report.get("warnings", []):
        print(f"! {warning}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default="outputs/experiment")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--window-fraction", type=float, default=DEFAULT_WINDOW_FRACTION)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = build_report(
        args.experiment_dir,
        bootstrap=max(1, args.bootstrap),
        rng_seed=args.rng_seed,
        window_fraction=min(max(args.window_fraction, 0.01), 1.0),
    )
    output_path = args.output or os.path.join(args.experiment_dir, REPORT_FILENAME)
    write_report(report, output_path)
    if not args.quiet:
        print_digest(report)
    print(f"analytics report written to: {output_path}")
    if report["source"]["trials_analyzed"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
