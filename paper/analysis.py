import json
import math
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analyzer import nash_equilibrium

DATA = os.path.join(ROOT, "outputs", "paper")
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
SIZES = (80, 100, 150)
SEEDS = tuple(range(1, 13))
CONDITIONS = ("control", "delta2")
LABEL = {"control": "Silent", "delta2": "Broadcast"}
COLOR = {"control": "#2b6cb0", "delta2": "#c53030"}
THRESHOLD = 60
SHORT = 100
LONG = 1000
LATE = 200
SMOOTH = 20
CORR_WINDOW = 50
BOOT = 2000
VIEWS = {
    "Short": (f"h{SHORT}", "analytics_report.json"),
    "Long": (f"h{LONG}", "analytics_report.json"),
    "Late": (f"h{LONG}", "analytics_report_late.json"),
}
KEY_METRICS = (
    ("behavior", "attendance_mean", "Mean attendance"),
    ("behavior", "deception_mean", "Deception index"),
    ("behavior", "utility_mean", "Mean utility"),
    ("nash", "mean_abs_gap", r"$|A_t - A^*|$"),
    ("nash", "mean_regret", "Mean regret"),
    ("nash", "efficiency_mean", "Efficiency"),
    ("gini", "final", "Gini (utility)"),
    ("drift", "entropy_mean", "Strategy entropy"),
)
ROMAN = (("150", "CL"), ("100", "C"), ("80", "LXXX"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def trial_dir(horizon, n, condition, seed):
    return os.path.join(DATA, f"h{horizon}", f"n{n}", f"{condition}_seed_{seed}")


def load_report(folder, n, name):
    with open(os.path.join(DATA, folder, f"n{n}", name), encoding="utf-8") as f:
        return json.load(f)


def load_epochs(horizon, n, condition, seed):
    return pd.read_csv(os.path.join(trial_dir(horizon, n, condition, seed), "epoch_metrics.csv"))


def load_agents(horizon, n, condition, seed, columns=None):
    return pd.read_csv(os.path.join(trial_dir(horizon, n, condition, seed), "agent_epoch_details.csv"), usecols=columns)


def load_trace(horizon, n, condition, seed, columns=None):
    return pd.read_csv(os.path.join(trial_dir(horizon, n, condition, seed), "mind_trace.csv"), usecols=columns)


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(("true", "1"))


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def sign_flip_p(diffs):
    d = np.asarray(diffs, dtype=float)
    observed = abs(d.mean())
    n = d.size
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
    means = np.abs((signs * d).mean(axis=1))
    return float(np.mean(means >= observed - 1e-12))


def seed_band(matrix, rng):
    m = np.asarray(matrix, dtype=float)
    picks = rng.integers(0, m.shape[0], size=(BOOT, m.shape[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(m[picks], axis=1)
        return np.nanmean(m, axis=0), np.nanpercentile(means, 2.5, axis=0), np.nanpercentile(means, 97.5, axis=0)


def rolling_mean(values, window):
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def rolling_corr(x, y, window):
    return pd.Series(x).rolling(window, min_periods=window).corr(pd.Series(y)).to_numpy()


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def signed(value, digits=3):
    return f"{value:+.{digits}f}"


def signed_tex(value, digits=3):
    return f"$+${abs(value):.{digits}f}" if value >= 0 else f"$-${abs(value):.{digits}f}"


def fmt_p(value):
    if value is None:
        return "--"
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


def macro_name(key):
    for digits, roman in ROMAN:
        key = key.replace(digits, roman)
    return key


def contrast_view(view, reports, macros):
    rows = []
    for n in SIZES:
        for group, ms in reports[n]["contrast"]["metrics"].items():
            for key, value in ms.items():
                if value and value.get("p_value") is not None:
                    rows.append((n, group, key, value))
    adjusted = benjamini_hochberg([r[3]["p_value"] for r in rows])
    q_lookup = {(n, g, k): q for (n, g, k, _), q in zip(rows, adjusted)}
    nominal = [r for r in rows if r[3]["p_value"] < 0.05]
    macros[f"nTests{view}"] = str(len(rows))
    macros[f"nNominal{view}"] = str(len(nominal))
    macros[f"nSurvive{view}"] = str(int(np.sum(adjusted < 0.05)))
    macros[f"minQ{view}"] = f"{float(np.min(adjusted)):.2f}"
    lines = []
    for group, key, label in KEY_METRICS:
        cells = [label]
        for n in SIZES:
            v = reports[n]["contrast"]["metrics"][group][key]
            digits = 2 if key in ("attendance_mean", "mean_abs_gap") else 3
            cells.append(f"{fmt(v['control']['mean'], digits)} & {fmt(v['delta2']['mean'], digits)} & {signed_tex(v['estimate'], digits)} & {fmt_p(v['p_value'])} & {fmt_p(q_lookup[(n, group, key)])}")
        lines.append(" & ".join(cells) + r" \\")
    with open(os.path.join(OUT, f"table_contrast_{view.lower()}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\\bottomrule\n")
    for n in SIZES:
        for condition, tag in (("control", "Ctl"), ("delta2", "Bc")):
            block = reports[n]["conditions"][condition]["metrics"]
            macros[f"att{tag}{view}{n}"] = f"{block['behavior']['attendance_mean']['mean']:.1f}"
            macros[f"dec{tag}{view}{n}"] = f"{block['behavior']['deception_mean']['mean']:.3f}"
            macros[f"util{tag}{view}{n}"] = f"{block['behavior']['utility_mean']['mean']:.3f}"
            macros[f"gap{tag}{view}{n}"] = f"{block['nash']['mean_abs_gap']['mean']:.2f}"
            minds = reports[n]["conditions"][condition].get("minds") or {}
            for key, short, digits in (("honesty_lie_spearman", "rho", 2), ("reputation_gap_honest_minus_liars", "repgap", 3)):
                stat = minds.get(key) or {}
                value = stat.get("mean") if isinstance(stat, dict) else None
                macros[f"{short}{tag}{view}{n}"] = (f"{value:+.{digits}f}" if short == "rho" else f"{value:.{digits}f}") if value is not None else "--"
    rhos = [reports[n]["conditions"][c]["minds"]["honesty_lie_spearman"]["mean"] for n in SIZES for c in CONDITIONS]
    macros[f"rhoLo{view}"] = f"{min(rhos):+.2f}"
    macros[f"rhoHi{view}"] = f"{max(rhos):+.2f}"
    scale_changes = []
    for n in SIZES:
        v = reports[n]["contrast"]["metrics"]["behavior"]["utility_mean"]
        scale_changes.append(abs(v["estimate"]) / abs(v["control"]["mean"]) * 100 if v["control"]["mean"] else 0.0)
    macros[f"maxUtilPct{view}"] = f"{max(scale_changes):.1f}"
    return {(n, g, k): (v, q_lookup[(n, g, k)]) for n, g, k, v in rows}


def claims_view(horizon, start):
    stats = {}
    for n in SIZES:
        for condition in CONDITIONS:
            corrs, shares, bluffs, coverts, sds = [], [], [], [], []
            for seed in SEEDS:
                e = load_epochs(horizon, n, condition, seed)
                e = e[e["epoch"] >= start]
                frac = e["actual_attendance"] / n
                corrs.append(float(np.corrcoef(e["town_broadcast_ratio"], frac)[0, 1]))
                shares.append(float(e["town_broadcast_ratio"].mean()))
                sds.append(float(e["actual_attendance"].std(ddof=1)))
                a = load_agents(horizon, n, condition, seed, ["epoch", "stated_intention", "in_bar"])
                a = a[a["epoch"] >= start]
                going = a["stated_intention"].astype(str).eq("going")
                at_bar = as_bool(a["in_bar"])
                bluffs.append(float((going & ~at_bar).mean()))
                coverts.append(float((~going & at_bar).mean()))
            stats[(n, condition)] = {"corr": corrs, "share": shares, "bluff": bluffs, "covert": coverts, "sd": sds}
    return stats


def main():
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(20261011)
    macros = {"nightsShort": str(SHORT), "nightsLong": str(LONG), "nightsLate": str(LATE), "lateStart": str(LONG - LATE + 1), "smooth": str(SMOOTH), "corrWindow": str(CORR_WINDOW)}
    reports = {view: {n: load_report(folder, n, name) for n in SIZES} for view, (folder, name) in VIEWS.items()}
    contrasts = {view: contrast_view(view, reports[view], macros) for view in VIEWS}
    for n in SIZES:
        ne = nash_equilibrium(n, THRESHOLD)
        macros[f"pstar{n}"] = f"{ne['mixed_go_probability']:.3f}"
        macros[f"mixedA{n}"] = f"{ne['mixed_expected_attendance']:.1f}"
        macros[f"optA{n}"] = str(ne["social_optimum_attendance"])
        macros[f"pureA{n}"] = ", ".join(str(a) for a in ne["pure_attendance"])

    claims = {"Short": claims_view(SHORT, 0), "Late": claims_view(LONG, LONG - LATE)}
    with open(os.path.join(OUT, "table_information.tex"), "w", encoding="utf-8") as f:
        for n in SIZES:
            for condition in CONDITIONS:
                s = claims["Short"][(n, condition)]
                l = claims["Late"][(n, condition)]
                f.write(
                    f"{n} & {LABEL[condition]} & {signed_tex(np.mean(s['corr']))} $\\pm$ {np.std(s['corr'], ddof=1):.3f} & "
                    f"{signed_tex(np.mean(l['corr']))} $\\pm$ {np.std(l['corr'], ddof=1):.3f} & {np.mean(l['share']):.3f} & "
                    f"{np.mean(l['bluff']):.3f} & {np.mean(l['covert']):.3f} & {np.mean(l['sd']):.2f} \\\\\n"
                )
        f.write("\\bottomrule\n")
    for view, table in claims.items():
        corr_means = []
        for n in SIZES:
            macros[f"corrP{view}{n}"] = fmt_p(sign_flip_p(np.array(table[(n, "delta2")]["corr"]) - np.array(table[(n, "control")]["corr"])))
            for condition, tag in (("control", "Ctl"), ("delta2", "Bc")):
                stat = table[(n, condition)]
                corr_means.append(abs(np.mean(stat["corr"])))
                macros[f"corr{tag}{view}{n}"] = signed(np.mean(stat["corr"]))
                macros[f"bluff{tag}{view}{n}"] = f"{np.mean(stat['bluff']):.3f}"
                macros[f"covert{tag}{view}{n}"] = f"{np.mean(stat['covert']):.3f}"
                macros[f"share{tag}{view}{n}"] = f"{np.mean(stat['share']):.3f}"
        macros[f"maxAbsCorr{view}"] = f"{max(corr_means):.3f}"

    curves = {}
    for n in SIZES:
        for condition in CONDITIONS:
            rows = {"attendance": [], "utility": [], "loss": [], "deception": [], "corr": []}
            for seed in SEEDS:
                e = load_epochs(LONG, n, condition, seed)
                rows["attendance"].append(e["actual_attendance"].to_numpy(dtype=float))
                rows["utility"].append(e["mean_utility"].to_numpy(dtype=float))
                rows["loss"].append(pd.to_numeric(e["td_loss"], errors="coerce").to_numpy(dtype=float))
                rows["deception"].append(e["deception_index"].to_numpy(dtype=float))
                rows["corr"].append(rolling_corr(e["town_broadcast_ratio"].to_numpy(dtype=float), (e["actual_attendance"] / n).to_numpy(dtype=float), CORR_WINDOW))
            curves[(n, condition)] = {k: np.array(v) for k, v in rows.items()}

    for n in SIZES:
        early, late = [], []
        for condition in CONDITIONS:
            gap = np.abs(curves[(n, condition)]["attendance"] - THRESHOLD)
            early.append(gap[:, :SHORT].mean())
            late.append(gap[:, -LATE:].mean())
            att = curves[(n, condition)]["attendance"]
            tag = "Ctl" if condition == "control" else "Bc"
            macros[f"attEarly{tag}{n}"] = f"{att[:, :SHORT].mean():.1f}"
            macros[f"attLate{tag}{n}"] = f"{att[:, -LATE:].mean():.1f}"
            loss = curves[(n, condition)]["loss"]
            macros[f"lossEarly{tag}{n}"] = f"{np.nanmean(loss[:, :SHORT]):.3f}"
            macros[f"lossLate{tag}{n}"] = f"{np.nanmean(loss[:, -LATE:]):.3f}"
            corr = curves[(n, condition)]["corr"]
            macros[f"rollCorrLate{tag}{n}"] = signed(float(np.nanmean(corr[:, -LATE:])))
        macros[f"gapEarly{n}"] = f"{np.mean(early):.2f}"
        macros[f"gapLate{n}"] = f"{np.mean(late):.2f}"
    late_roll = [abs(float(np.nanmean(curves[(n, c)]["corr"][:, -LATE:]))) for n in SIZES for c in CONDITIONS]
    macros["maxAbsRollCorrLate"] = f"{max(late_roll):.3f}"

    panels = (
        ("attendance", "Attendance", True),
        ("utility", "Mean utility", True),
        ("loss", "TD loss", True),
        ("deception", "Deception index", True),
        ("corr", f"$r(b_t, A_t/N)$, {CORR_WINDOW}-night", False),
    )
    fig, grid = plt.subplots(len(panels), len(SIZES), figsize=(7.0, 8.2), sharex=True)
    fig.subplots_adjust(hspace=0.28, wspace=0.28)
    x = np.arange(1, LONG + 1)
    for col, n in enumerate(SIZES):
        for row, (key, label, smooth) in enumerate(panels):
            ax = grid[row, col]
            for condition in CONDITIONS:
                matrix = curves[(n, condition)][key]
                if smooth:
                    matrix = np.array([rolling_mean(r, SMOOTH) for r in matrix])
                mean, lo, hi = seed_band(matrix, rng)
                ax.fill_between(x, lo, hi, color=COLOR[condition], alpha=0.2, linewidth=0)
                ax.plot(x, mean, color=COLOR[condition], linewidth=0.9, label=LABEL[condition])
            if key == "attendance":
                ax.axhline(THRESHOLD, color="0.2", linestyle="--", linewidth=0.7)
                ax.axhline(n * nash_equilibrium(n, THRESHOLD)["mixed_go_probability"], color="0.45", linestyle=":", linewidth=0.8)
            if key == "deception":
                ax.axhline(0.5, color="0.2", linestyle=":", linewidth=0.7)
            if key == "corr":
                ax.axhline(0.0, color="0.2", linestyle=":", linewidth=0.7)
            if key == "loss":
                ax.set_yscale("log")
            ax.axvspan(LONG - LATE, LONG, color="0.85", alpha=0.35, linewidth=0)
            ax.set_xlim(1, LONG)
            if row == 0:
                ax.set_title(f"$N = {n}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(label)
            if row == len(panels) - 1:
                ax.set_xlabel("Night")
    grid[0, 0].legend(loc="lower right", fontsize=7)
    fig.savefig(os.path.join(FIG, "learning.pdf"))
    plt.close(fig)

    fig, grid = plt.subplots(2, 4, figsize=(7.0, 3.5))
    fig.subplots_adjust(hspace=0.8, wspace=0.3)
    axes = grid.ravel()
    styles = {"Short": ("o", "0.55", 0.24), "Long": ("s", "0.35", 0.0), "Late": ("D", "0.05", -0.24)}
    rows_y = {n: i for i, n in enumerate(SIZES)}
    for ax, (group, key, label) in zip(axes, KEY_METRICS):
        for view, (marker, color, offset) in styles.items():
            for n in SIZES:
                v, _ = contrasts[view][(n, group, key)]
                scale = abs(v["control"]["mean"]) or 1.0
                est = v["estimate"] / scale * 100
                lo, hi = (c / scale * 100 for c in v["ci95"])
                y = rows_y[n] + offset
                ax.plot([lo, hi], [y, y], color=color, linewidth=0.8)
                ax.plot(est, y, marker, color=color, markersize=2.6, label=view if (ax is axes[0] and n == SIZES[0]) else None)
        ax.axvline(0, color="0.5", linewidth=0.6)
        ax.set_yticks(list(rows_y.values()))
        ax.set_yticklabels([str(n) for n in SIZES] if ax in (axes[0], axes[4]) else [])
        ax.set_ylim(-0.6, len(SIZES) - 0.4)
        ax.set_title(label, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("$N$")
    axes[4].set_ylabel("$N$")
    handles, labels = axes[0].get_legend_handles_labels()
    names = {"Short": f"{SHORT} nights", "Long": f"{LONG} nights", "Late": f"last {LATE} of {LONG}"}
    fig.legend(handles, [names[l] for l in labels], loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02, "Broadcast $-$ silent, % of the silent mean (95% hierarchical bootstrap CI)", ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG, "contrast.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    start = LONG - LATE
    for condition in CONDITIONS:
        xs, ys = [], []
        for seed in SEEDS:
            with open(os.path.join(trial_dir(LONG, 100, condition, seed), "minds.json"), encoding="utf-8") as f:
                minds = json.load(f)
            trace = load_trace(LONG, 100, condition, seed, ["epoch", "agent_id", "lied"])
            trace = trace[trace["epoch"] >= start]
            rates = as_bool(trace["lied"]).groupby(trace["agent_id"]).mean()
            for agent_id, rate in rates.items():
                xs.append(minds[str(agent_id)]["traits"]["honesty"])
                ys.append(rate)
        axes[0].scatter(xs, ys, s=3, alpha=0.35, color=COLOR[condition], linewidths=0, label=LABEL[condition])
    axes[0].set_xlabel("Honesty trait")
    axes[0].set_ylabel(f"Lie rate, last {LATE} nights")
    axes[0].set_title("$N = 100$: temperament drives lying", fontsize=9)
    axes[0].legend(loc="upper right", markerscale=3)
    for condition in CONDITIONS:
        xs, ys = [], []
        for seed in SEEDS:
            e = load_epochs(LONG, 100, condition, seed)
            e = e[e["epoch"] >= start]
            xs.extend(e["town_broadcast_ratio"].tolist())
            ys.extend((e["actual_attendance"] / 100).tolist())
        axes[1].scatter(xs, ys, s=3, alpha=0.3, color=COLOR[condition], linewidths=0, label=LABEL[condition])
    axes[1].plot([0.2, 0.8], [0.2, 0.8], color="0.4", linewidth=0.6, linestyle="--")
    axes[1].set_xlabel("Share of agents claiming “going”")
    axes[1].set_ylabel("Attendance share")
    axes[1].set_title("$N = 100$: claims do not predict the crowd", fontsize=9)
    fig.savefig(os.path.join(FIG, "minds.pdf"))
    plt.close(fig)

    with open(os.path.join(OUT, "results.tex"), "w", encoding="utf-8") as f:
        for key, value in sorted(macros.items()):
            name = macro_name(key)
            if any(ch.isdigit() for ch in name):
                raise ValueError(f"macro name {name} contains a digit")
            text = f"\\ensuremath{{-}}{value[1:]}" if value.startswith("-") else value
            f.write(f"\\newcommand{{\\r{name}}}{{{text}}}\n")
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
