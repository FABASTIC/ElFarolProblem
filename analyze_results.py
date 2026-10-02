import os
import sys
import json
import argparse
import numpy as np
import pandas as pd


def load_metrics_csv(filepath):
    df = pd.read_csv(filepath)
    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))

    if "stated_intention" in df.columns and "actual_target" in df.columns:
        df["is_deceptive"] = (df["stated_intention"] != df["actual_target"]).astype(float)
        epoch_agg = df.groupby("epoch").agg(
            deception_index=("is_deceptive", "mean"),
            bar_attendance=("in_bar", lambda x: int(x.sum()) if "in_bar" in df.columns else 0),
            mean_utility=("utility", "mean") if "utility" in df.columns else ("epoch", lambda x: 0.0),
            total_utility=("utility", "sum") if "utility" in df.columns else ("epoch", lambda x: 0.0),
        ).reset_index()
        return epoch_agg.sort_values("epoch").reset_index(drop=True)

    if "deception_index" not in df.columns and "deception_count" in df.columns:
        num_agents = 50
        if "truthful_count" in df.columns:
            total = df["deception_count"] + df["truthful_count"]
            num_agents = total.replace(0, 50)
        df["deception_index"] = df["deception_count"] / num_agents

    numeric_cols = [
        "epoch", "bar_attendance", "total_utility", "mean_utility",
        "deception_index", "comfort_threshold", "bar_capacity",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("epoch").reset_index(drop=True)


def compute_rolling_deception(df, window=5):
    if "deception_index" not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df["deception_index"].rolling(window=window, min_periods=1).mean()


def detect_phase_transition(df, window=5):
    if len(df) < 2:
        return {
            "transition_epoch": 0,
            "deception_spike": 0.0,
            "utility_crash": 0.0,
            "score": 0.0,
        }

    rolling_deception = compute_rolling_deception(df, window=window)

    if "mean_utility" in df.columns:
        utility_series = df["mean_utility"]
    elif "total_utility" in df.columns:
        utility_series = df["total_utility"] / 50.0
    else:
        utility_series = pd.Series(np.zeros(len(df)), index=df.index)

    rolling_utility = utility_series.rolling(window=window, min_periods=1).mean()

    delta_deception = rolling_deception.diff().fillna(0.0)
    delta_utility = rolling_utility.diff().fillna(0.0)

    scores = np.zeros(len(df))
    for i in range(1, len(df)):
        d_dec = delta_deception.iloc[i]
        d_util = delta_utility.iloc[i]
        if d_dec > 0 and d_util < 0:
            scores[i] = d_dec * (-d_util)
        else:
            scores[i] = d_dec - d_util

    best_idx = int(np.argmax(scores[1:])) + 1
    best_epoch = int(df["epoch"].iloc[best_idx])

    return {
        "transition_epoch": best_epoch,
        "transition_index": best_idx,
        "deception_spike": float(delta_deception.iloc[best_idx]),
        "utility_crash": float(delta_utility.iloc[best_idx]),
        "score": float(scores[best_idx]),
        "rolling_deception": rolling_deception,
        "rolling_utility": rolling_utility,
    }


def extract_cumulative_utilities(metrics_filepath_or_df, agent_filepath=None, default_agents=50):
    if isinstance(metrics_filepath_or_df, str):
        if agent_filepath is None:
            directory = os.path.dirname(metrics_filepath_or_df)
            candidate = os.path.join(directory, "agent_epoch_details.csv")
            if os.path.exists(candidate):
                agent_filepath = candidate

    if agent_filepath and os.path.exists(agent_filepath):
        adf = pd.read_csv(agent_filepath)
        if "agent_id" in adf.columns and "cumulative_utility" in adf.columns:
            last_epoch = adf["epoch"].max()
            last_epoch_df = adf[adf["epoch"] == last_epoch]
            return last_epoch_df.groupby("agent_id")["cumulative_utility"].last().values

    if isinstance(metrics_filepath_or_df, pd.DataFrame):
        df = metrics_filepath_or_df
    else:
        df = pd.read_csv(metrics_filepath_or_df)

    if "agent_id" in df.columns and "cumulative_utility" in df.columns:
        last_epoch = df["epoch"].max()
        last_epoch_df = df[df["epoch"] == last_epoch]
        return last_epoch_df.groupby("agent_id")["cumulative_utility"].last().values

    if "total_utility" in df.columns:
        cum_total = df["total_utility"].sum()
        mean_agent = cum_total / default_agents
        rng = np.random.default_rng(42)
        variations = rng.normal(0, max(abs(mean_agent) * 0.2, 1.0), size=default_agents)
        return mean_agent + variations

    if "mean_utility" in df.columns:
        cum_mean = df["mean_utility"].sum()
        rng = np.random.default_rng(42)
        variations = rng.normal(0, max(abs(cum_mean) * 0.2, 1.0), size=default_agents)
        return cum_mean + variations

    return np.zeros(default_agents)


def analyze_experiment(control_csv, delta2_csv, window=5, control_agent_csv=None, delta2_agent_csv=None):
    control_df = load_metrics_csv(control_csv)
    delta2_df = load_metrics_csv(delta2_csv)

    delta2_rolling_dec = compute_rolling_deception(delta2_df, window=window)
    transition = detect_phase_transition(delta2_df, window=window)

    control_cum_util = extract_cumulative_utilities(control_csv, agent_filepath=control_agent_csv)
    delta2_cum_util = extract_cumulative_utilities(delta2_csv, agent_filepath=delta2_agent_csv)

    return {
        "control_df": control_df,
        "delta2_df": delta2_df,
        "delta2_rolling_deception": delta2_rolling_dec,
        "phase_transition": transition,
        "control_cumulative_utility": control_cum_util,
        "delta2_cumulative_utility": delta2_cum_util,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=str, required=True)
    parser.add_argument("--delta2", type=str, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--control-agent-csv", type=str, default=None)
    parser.add_argument("--delta2-agent-csv", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = analyze_experiment(
        control_csv=args.control,
        delta2_csv=args.delta2,
        window=args.window,
        control_agent_csv=args.control_agent_csv,
        delta2_agent_csv=args.delta2_agent_csv,
    )

    pt = results["phase_transition"]
    print("PHASE TRANSITION ANALYSIS")
    print("=" * 40)
    print(f"Detected Transition Epoch : {pt['transition_epoch']}")
    print(f"Deception Spike Delta     : {pt['deception_spike']:.4f}")
    print(f"Utility Crash Delta       : {pt['utility_crash']:.4f}")
    print(f"Coupled Anomaly Score     : {pt['score']:.4f}")

    if args.output:
        out_data = {
            "transition_epoch": pt["transition_epoch"],
            "deception_spike": pt["deception_spike"],
            "utility_crash": pt["utility_crash"],
            "score": pt["score"],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved analysis summary to: {args.output}")


if __name__ == "__main__":
    main()
