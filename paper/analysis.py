import json
import math
import os
import sys

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
KEY_METRICS = (
    ("behavior", "attendance_mean", "Mean attendance"),
    ("behavior", "deception_mean", "Deception index"),
    ("behavior", "utility_mean", "Mean utility"),
    ("nash", "mean_abs_gap", r"$|A_t - A^*|$"),
    ("nash", "mean_regret", "Mean regret"),
    ("nash", "efficiency_mean", "Efficiency"),
    ("gini", "final", "Gini (final utility)"),
    ("drift", "entropy_mean", "Strategy entropy"),
)

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


def trial_dir(n, condition, seed):
    return os.path.join(DATA, f"n{n}", f"{condition}_seed_{seed}")


def load_report(n):
    with open(os.path.join(DATA, f"n{n}", "analytics_report.json"), encoding="utf-8") as f:
        return json.load(f)


def load_epochs(n, condition, seed):
    return pd.read_csv(os.path.join(trial_dir(n, condition, seed), "epoch_metrics.csv"))


def load_agents(n, condition, seed):
    return pd.read_csv(os.path.join(trial_dir(n, condition, seed), "agent_epoch_details.csv"))


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


def band(matrix):
    m = np.asarray(matrix, dtype=float)
    mean = m.mean(axis=0)
    half = 1.96 * m.std(axis=0, ddof=1) / math.sqrt(m.shape[0])
    return mean, mean - half, mean + half


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def signed_tex(value, digits=3):
    return f"$+${abs(value):.{digits}f}" if value >= 0 else f"$-${abs(value):.{digits}f}"


def fmt_p(value):
    if value is None:
        return "--"
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


def main():
    os.makedirs(FIG, exist_ok=True)
    reports = {n: load_report(n) for n in SIZES}
    macros = {}
    rows = []
    for n in SIZES:
        ne = nash_equilibrium(n, 60)
        macros[f"pstar{n}"] = f"{ne['mixed_go_probability']:.3f}"
        macros[f"mixedA{n}"] = f"{ne['mixed_expected_attendance']:.1f}"
        macros[f"optA{n}"] = str(ne["social_optimum_attendance"])
        macros[f"pureA{n}"] = ", ".join(str(a) for a in ne["pure_attendance"])
        metrics = reports[n]["contrast"]["metrics"]
        for group, ms in metrics.items():
            for key, value in ms.items():
                if value and value.get("p_value") is not None:
                    rows.append((n, group, key, value))
    adjusted = benjamini_hochberg([r[3]["p_value"] for r in rows])
    q_lookup = {(n, g, k): q for (n, g, k, _), q in zip(rows, adjusted)}
    nominal = sum(1 for r in rows if r[3]["p_value"] < 0.05)
    survivors = sum(1 for q in adjusted if q < 0.05)
    macros["nTests"] = str(len(rows))
    macros["nNominal"] = str(nominal)
    macros["nSurvive"] = str(survivors)
    macros["minQ"] = f"{float(np.min(adjusted)):.2f}"

    table = []
    for group, key, label in KEY_METRICS:
        cells = [label]
        for n in SIZES:
            v = reports[n]["contrast"]["metrics"][group][key]
            digits = 2 if group == "behavior" and key == "attendance_mean" or key == "mean_abs_gap" else 3
            cells.append(f"{fmt(v['control']['mean'], digits)} & {fmt(v['delta2']['mean'], digits)} & {'$+$' if v['estimate'] >= 0 else '$-$'}{fmt(abs(v['estimate']), digits)} & {fmt_p(v['p_value'])} & {fmt_p(q_lookup[(n, group, key)])}")
        table.append(" & ".join(cells) + r" \\")
    with open(os.path.join(OUT, "table_contrast.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(table) + "\n\\bottomrule\n")

    info_rows = []
    claim_corr = {c: {} for c in CONDITIONS}
    lies = {c: {} for c in CONDITIONS}
    late_sd = {c: {} for c in CONDITIONS}
    for n in SIZES:
        for condition in CONDITIONS:
            corrs, shares, bluffs, coverts, sds = [], [], [], [], []
            for seed in SEEDS:
                e = load_epochs(n, condition, seed)
                frac = e["actual_attendance"] / n
                corrs.append(float(np.corrcoef(e["town_broadcast_ratio"], frac)[0, 1]))
                shares.append(float(e["town_broadcast_ratio"].mean()))
                sds.append(float(e["actual_attendance"].iloc[-20:].std(ddof=1)))
                a = load_agents(n, condition, seed)
                going = a["stated_intention"].astype(str).eq("going")
                at_bar = as_bool(a["in_bar"])
                bluffs.append(float((going & ~at_bar).mean()))
                coverts.append(float((~going & at_bar).mean()))
            claim_corr[condition][n] = corrs
            lies[condition][n] = (bluffs, coverts)
            late_sd[condition][n] = sds
            info_rows.append((n, condition, np.mean(corrs), np.std(corrs, ddof=1), np.mean(shares), np.mean(bluffs), np.mean(coverts), np.mean(sds)))
    with open(os.path.join(OUT, "table_information.tex"), "w", encoding="utf-8") as f:
        for n, condition, c_mean, c_sd, share, bluff, covert, sd in info_rows:
            f.write(f"{n} & {LABEL[condition]} & {signed_tex(c_mean)} $\\pm$ {c_sd:.3f} & {share:.3f} & {bluff:.3f} & {covert:.3f} & {sd:.2f} \\\\\n")
        f.write("\\bottomrule\n")
    corr_pairs = {n: sign_flip_p(np.array(claim_corr["delta2"][n]) - np.array(claim_corr["control"][n])) for n in SIZES}
    for n in SIZES:
        macros[f"corrCtl{n}"] = f"{np.mean(claim_corr['control'][n]):+.3f}"
        macros[f"corrBc{n}"] = f"{np.mean(claim_corr['delta2'][n]):+.3f}"
        macros[f"corrP{n}"] = fmt_p(corr_pairs[n])
        for condition, tag in (("control", "Ctl"), ("delta2", "Bc")):
            b, c = lies[condition][n]
            macros[f"bluff{tag}{n}"] = f"{np.mean(b):.3f}"
            macros[f"covert{tag}{n}"] = f"{np.mean(c):.3f}"
        for condition, tag in (("control", "Ctl"), ("delta2", "Bc")):
            block = reports[n]["conditions"][condition]["metrics"]
            macros[f"att{tag}{n}"] = f"{block['behavior']['attendance_mean']['mean']:.1f}"
            macros[f"dec{tag}{n}"] = f"{block['behavior']['deception_mean']['mean']:.3f}"
            macros[f"util{tag}{n}"] = f"{block['behavior']['utility_mean']['mean']:.3f}"
            macros[f"eff{tag}{n}"] = f"{block['nash']['efficiency_mean']['mean']:.2f}"
            minds = reports[n]["conditions"][condition].get("minds") or {}
            rho = minds.get("honesty_lie_spearman", {})
            macros[f"rho{tag}{n}"] = f"{rho.get('mean'):+.2f}" if isinstance(rho, dict) and rho.get("mean") is not None else "--"
            gap = minds.get("reputation_gap_honest_minus_liars", {})
            macros[f"repgap{tag}{n}"] = f"{gap.get('mean'):.3f}" if isinstance(gap, dict) and gap.get("mean") is not None else "--"
            trust = minds.get("mean_trust_given_final", {})
            macros[f"trust{tag}{n}"] = f"{trust.get('mean'):.3f}" if isinstance(trust, dict) and trust.get("mean") is not None else "--"
            agree = minds.get("instinct_agreement", {})
            macros[f"agree{tag}{n}"] = f"{agree.get('mean'):.3f}" if isinstance(agree, dict) and agree.get("mean") is not None else "--"

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=False)
    for ax, n in zip(axes, SIZES):
        for condition in CONDITIONS:
            matrix = [load_epochs(n, condition, s)["actual_attendance"].to_numpy() for s in SEEDS]
            mean, lo, hi = band(matrix)
            x = np.arange(1, len(mean) + 1)
            ax.fill_between(x, lo, hi, color=COLOR[condition], alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=COLOR[condition], linewidth=1.0, label=LABEL[condition])
        ax.axhline(60, color="0.25", linestyle="--", linewidth=0.7)
        ax.set_title(f"$N = {n}$", fontsize=9)
        ax.set_xlabel("Night")
        ax.set_xlim(1, 100)
    axes[0].set_ylabel("Attendance")
    axes[0].legend(loc="lower right")
    fig.savefig(os.path.join(FIG, "attendance.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, n in zip(axes, SIZES):
        for condition in CONDITIONS:
            matrix = [load_epochs(n, condition, s)["deception_index"].to_numpy() for s in SEEDS]
            mean, lo, hi = band(matrix)
            x = np.arange(1, len(mean) + 1)
            ax.fill_between(x, lo, hi, color=COLOR[condition], alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=COLOR[condition], linewidth=1.0, label=LABEL[condition])
        ax.axhline(0.5, color="0.25", linestyle=":", linewidth=0.7)
        ax.set_title(f"$N = {n}$", fontsize=9)
        ax.set_xlabel("Night")
        ax.set_xlim(1, 100)
    axes[0].set_ylabel("Deception index")
    axes[0].legend(loc="lower right")
    fig.savefig(os.path.join(FIG, "deception.pdf"))
    plt.close(fig)

    fig, grid = plt.subplots(2, 4, figsize=(7.0, 3.3))
    fig.subplots_adjust(hspace=0.75, wspace=0.3)
    axes = grid.ravel()
    offsets = {80: -0.22, 100: 0.0, 150: 0.22}
    for ax, (group, key, label) in zip(axes, KEY_METRICS):
        for i, n in enumerate(SIZES):
            v = reports[n]["contrast"]["metrics"][group][key]
            scale = abs(v["control"]["mean"]) or 1.0
            est = v["estimate"] / scale * 100
            lo, hi = (c / scale * 100 for c in v["ci95"])
            ax.plot([lo, hi], [offsets[n], offsets[n]], color="0.35", linewidth=0.9)
            ax.plot(est, offsets[n], "o", color="0.1", markersize=2.8)
        ax.axvline(0, color="0.5", linewidth=0.6)
        ax.set_yticks([offsets[n] for n in SIZES])
        ax.set_yticklabels([f"{n}" for n in SIZES] if ax in (axes[0], axes[4]) else [])
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(label, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("$N$")
    axes[4].set_ylabel("$N$")
    fig.text(0.5, -0.02, "Broadcast $-$ silent, % of the silent mean (95% hierarchical bootstrap CI)", ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG, "contrast.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for condition in CONDITIONS:
        xs, ys = [], []
        for seed in SEEDS:
            with open(os.path.join(trial_dir(100, condition, seed), "minds.json"), encoding="utf-8") as f:
                minds = json.load(f)
            for mind in minds.values():
                xs.append(mind["traits"]["honesty"])
                ys.append(mind["private_lies"] / max(1, mind["decisions"]))
        axes[0].scatter(xs, ys, s=3, alpha=0.35, color=COLOR[condition], linewidths=0, label=LABEL[condition])
    axes[0].set_xlabel("Honesty trait")
    axes[0].set_ylabel("Agent lie rate")
    axes[0].set_title("$N = 100$: temperament drives lying", fontsize=9)
    axes[0].legend(loc="upper right", markerscale=3)
    for condition in CONDITIONS:
        xs, ys = [], []
        for seed in SEEDS:
            e = load_epochs(100, condition, seed)
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
            name = key.replace("150", "CL").replace("100", "C").replace("80", "LXXX")
            text = f"\\ensuremath{{-}}{value[1:]}" if value.startswith("-") else value
            f.write(f"\\newcommand{{\\r{name}}}{{{text}}}\n")
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
