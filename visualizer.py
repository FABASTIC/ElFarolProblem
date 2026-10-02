import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from analyze_results import (
    load_metrics_csv,
    compute_rolling_deception,
    detect_phase_transition,
    extract_cumulative_utilities,
)

COLOR_BG = (17 / 255, 17 / 255, 17 / 255)
COLOR_SPINE = (55 / 255, 55 / 255, 60 / 255)
COLOR_GRID = (28 / 255, 28 / 255, 32 / 255)
COLOR_TEXT = (235 / 255, 235 / 255, 235 / 255)
COLOR_MUTED = (135 / 255, 135 / 255, 142 / 255)
COLOR_CONTROL = (0 / 255, 229 / 255, 255 / 255)
COLOR_DELTA2 = (255 / 255, 0 / 255, 85 / 255)
COLOR_EMERALD = (0 / 255, 255 / 255, 163 / 255)
COLOR_THRESHOLD = (255 / 255, 215 / 255, 0 / 255)
COLOR_WHITE = (255 / 255, 255 / 255, 255 / 255)
COLOR_BOX_FILL = (26 / 255, 26 / 255, 30 / 255)
COLOR_LEGEND_BG = (22 / 255, 22 / 255, 25 / 255)


def apply_editorial_theme(fig, axes):
    fig.patch.set_facecolor(COLOR_BG)
    for ax in axes:
        ax.set_facecolor(COLOR_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_SPINE)
        ax.spines["bottom"].set_color(COLOR_SPINE)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(colors=COLOR_MUTED, labelsize=8, length=3.0, width=0.6)
        ax.xaxis.label.set_color(COLOR_TEXT)
        ax.yaxis.label.set_color(COLOR_TEXT)
        ax.xaxis.label.set_size(8.5)
        ax.yaxis.label.set_size(8.5)
        ax.title.set_color(COLOR_TEXT)
        ax.title.set_size(9.5)
        ax.title.set_weight("bold")
        ax.grid(True, linestyle=":", color=COLOR_GRID, linewidth=0.5, alpha=0.8)


def plot_simulation_results(
    control_df,
    delta2_df,
    control_cumulative_utility,
    delta2_cumulative_utility,
    phase_transition_epoch=None,
    output_path="el_farol_analysis.pdf",
    window=5,
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=300)
    apply_editorial_theme(fig, [ax1, ax2, ax3])

    ctrl_epochs = control_df["epoch"].values
    ctrl_att = control_df["bar_attendance"].values
    d2_epochs = delta2_df["epoch"].values
    d2_att = delta2_df["bar_attendance"].values

    ax1.plot(
        ctrl_epochs,
        ctrl_att,
        color=COLOR_CONTROL,
        linewidth=1.2,
        alpha=0.9,
        label="Control (No Broadcast)",
    )
    ax1.plot(
        d2_epochs,
        d2_att,
        color=COLOR_DELTA2,
        linewidth=1.2,
        alpha=0.95,
        label="Delta 2 (Strategic Deception)",
    )

    threshold_val = 60.0
    if "comfort_threshold" in delta2_df.columns and not delta2_df["comfort_threshold"].isna().all():
        threshold_val = float(delta2_df["comfort_threshold"].iloc[0])
    elif "comfort_threshold" in control_df.columns and not control_df["comfort_threshold"].isna().all():
        threshold_val = float(control_df["comfort_threshold"].iloc[0])

    ax1.axhline(
        threshold_val,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
        label=f"60% Threshold ({int(threshold_val)})",
    )
    ax1.set_title("01 // BAR ATTENDANCE DYNAMICS")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Bar Attendance (Agents)")
    ax1.set_ylim(bottom=0)
    leg1 = ax1.legend(
        facecolor=COLOR_LEGEND_BG,
        edgecolor=COLOR_SPINE,
        fontsize=7.5,
        loc="upper right",
        framealpha=0.9,
    )
    for text in leg1.get_texts():
        text.set_color(COLOR_TEXT)

    d2_raw_dec = delta2_df["deception_index"].values
    d2_rolling_dec = compute_rolling_deception(delta2_df, window=window).values

    ax2.plot(
        d2_epochs,
        d2_raw_dec,
        color=COLOR_DELTA2,
        linewidth=0.6,
        alpha=0.35,
        label="Raw Deception Index",
    )
    ax2.plot(
        d2_epochs,
        d2_rolling_dec,
        color=COLOR_EMERALD,
        linewidth=1.5,
        label=f"Rolling Mean (w={window})",
    )

    if phase_transition_epoch is not None:
        ax2.axvline(
            phase_transition_epoch,
            color=COLOR_THRESHOLD,
            linestyle=":",
            linewidth=1.0,
            alpha=0.9,
        )
        max_y = max(np.nanmax(d2_raw_dec) if len(d2_raw_dec) else 1.0, 0.5)
        ax2.text(
            phase_transition_epoch + 1.5,
            max_y * 0.85,
            f"CRITICAL TRANSITION [t={phase_transition_epoch}]",
            color=COLOR_THRESHOLD,
            fontsize=7.2,
            weight="bold",
        )

    ax2.set_title("02 // DECEPTION DYNAMICS & TRANSITION")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Deception Index")
    ax2.set_ylim(0.0, 1.05)
    leg2 = ax2.legend(
        facecolor=COLOR_LEGEND_BG,
        edgecolor=COLOR_SPINE,
        fontsize=7.5,
        loc="upper left",
        framealpha=0.9,
    )
    for text in leg2.get_texts():
        text.set_color(COLOR_TEXT)

    ctrl_u = np.asarray(control_cumulative_utility)
    d2_u = np.asarray(delta2_cumulative_utility)

    box_data = []
    for val in ctrl_u:
        box_data.append({"condition": "Control", "utility": float(val)})
    for val in d2_u:
        box_data.append({"condition": "Delta 2", "utility": float(val)})
    df_box = pd.DataFrame(box_data)

    sns.boxplot(
        ax=ax3,
        data=df_box,
        x="condition",
        y="utility",
        width=0.38,
        color=COLOR_BOX_FILL,
        medianprops=dict(color=COLOR_WHITE, linewidth=1.5),
        whiskerprops=dict(color=COLOR_MUTED, linewidth=0.7),
        capprops=dict(color=COLOR_MUTED, linewidth=0.7),
        flierprops=dict(marker=".", markersize=3, markerfacecolor=COLOR_MUTED, markeredgecolor="none"),
    )

    sns.stripplot(
        ax=ax3,
        data=df_box,
        x="condition",
        y="utility",
        hue="condition",
        legend=False,
        palette=[COLOR_CONTROL, COLOR_DELTA2],
        jitter=0.18,
        size=4.0,
        alpha=0.6,
    )

    for patch in ax3.patches:
        patch.set_facecolor(COLOR_BOX_FILL)
        patch.set_edgecolor(COLOR_SPINE)
        patch.set_linewidth(0.8)

    ax3.set_title("03 // CUMULATIVE UTILITY DISTRIBUTION")
    ax3.set_xlabel("Condition")
    ax3.set_ylabel("Cumulative Utility per Agent")

    plt.tight_layout(pad=2.2)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=COLOR_BG,
        edgecolor="none",
    )
    plt.close(fig)
    return output_path


def generate_figures_from_csv(
    control_csv,
    delta2_csv,
    output_path="el_farol_analysis.pdf",
    window=5,
    control_agent_csv=None,
    delta2_agent_csv=None,
):
    control_df = load_metrics_csv(control_csv)
    delta2_df = load_metrics_csv(delta2_csv)

    transition_info = detect_phase_transition(delta2_df, window=window)
    phase_epoch = transition_info["transition_epoch"]

    ctrl_u = extract_cumulative_utilities(control_csv, agent_filepath=control_agent_csv)
    d2_u = extract_cumulative_utilities(delta2_csv, agent_filepath=delta2_agent_csv)

    return plot_simulation_results(
        control_df=control_df,
        delta2_df=delta2_df,
        control_cumulative_utility=ctrl_u,
        delta2_cumulative_utility=d2_u,
        phase_transition_epoch=phase_epoch,
        output_path=output_path,
        window=window,
    )


TECHNICAL_FONTS = ("JetBrains Mono", "IBM Plex Mono", "Cascadia Mono", "Cascadia Code", "Consolas", "DejaVu Sans Mono")
COLOR_GROUND = (38 / 255, 38 / 255, 43 / 255)
COLOR_HONEST_BAR = (138 / 255, 138 / 255, 144 / 255)
COLOR_BLUFF = (255 / 255, 176 / 255, 0 / 255)
COLOR_NEUTRAL_MID = (46 / 255, 46 / 255, 51 / 255)
COLOR_RAMP_FLOOR = (24 / 255, 24 / 255, 28 / 255)
COLOR_RAMP_CEILING = (236 / 255, 236 / 255, 240 / 255)
PORTRAIT_SMOOTHING = 5
CONDITION_ORDER = ("control", "delta2")
CONDITION_COLORS = {"control": COLOR_CONTROL, "delta2": COLOR_DELTA2}
CONDITION_NAMES = {"control": "CONTROL // NO BROADCAST", "delta2": "DELTA 2 // BROADCAST + DECEPTION"}
STRATEGY_ORDER = ("honest_go", "honest_stay", "false_go", "false_stay")
STRATEGY_NAMES = {
    "honest_go": "HONEST · AT BAR",
    "honest_stay": "HONEST · HOME",
    "false_go": "BLUFF · SAID GO, STAYED",
    "false_stay": "COVERT · SAID STAY, WENT",
}
STRATEGY_COLORS = {
    "honest_go": COLOR_HONEST_BAR,
    "honest_stay": COLOR_GROUND,
    "false_go": COLOR_BLUFF,
    "false_stay": COLOR_DELTA2,
}
FIGURE_SCHEMA = "ELFAROL.FIGURES/1"


def _technical_font():
    try:
        available = {entry.name for entry in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    for name in TECHNICAL_FONTS:
        if name in available:
            return name
    return "DejaVu Sans Mono"


def _terminal_rc():
    primary = _technical_font()
    return {
        "font.family": [primary, "DejaVu Sans Mono"] if primary != "DejaVu Sans Mono" else ["DejaVu Sans Mono"],
        "font.monospace": [primary, "DejaVu Sans Mono"],
        "font.size": 7.0,
        "text.color": COLOR_TEXT,
        "axes.facecolor": COLOR_BG,
        "axes.edgecolor": COLOR_SPINE,
        "axes.linewidth": 0.5,
        "axes.labelcolor": COLOR_MUTED,
        "axes.labelsize": 6.2,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLOR_TEXT,
        "axes.unicode_minus": False,
        "figure.facecolor": COLOR_BG,
        "savefig.facecolor": COLOR_BG,
        "savefig.edgecolor": COLOR_BG,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.45,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }


def _arr(values):
    if values is None:
        return np.array([], dtype=float)
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.array([0.0, 1.0])
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate(([centers[0] - (mids[0] - centers[0])], mids, [centers[-1] + (centers[-1] - mids[-1])]))


def _present_conditions(report):
    conditions = report.get("conditions") or {}
    ordered = [c for c in CONDITION_ORDER if (conditions.get(c) or {}).get("series")]
    extras = sorted(c for c in conditions if c not in CONDITION_ORDER and (conditions.get(c) or {}).get("series"))
    return ordered + extras


def _trials_for(report, label):
    return [t for t in report.get("trials", []) if t.get("condition") == label]


def _changepoint(report, label, key):
    phase = ((report.get("conditions") or {}).get(label) or {}).get("phase") or {}
    return (phase.get("changepoints") or {}).get(key)


def _style_panel(ax, index, title, subtitle=None):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLOR_SPINE)
        ax.spines[side].set_linewidth(0.5)
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, color=COLOR_GRID, linewidth=0.45, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=COLOR_MUTED, labelsize=5.8, length=2.5, width=0.5, direction="in", top=True, right=True)
    ax.set_title(f"{index:02d} // {title}", loc="left", pad=13 if subtitle else 6, color=COLOR_TEXT, fontsize=7.5, fontweight="bold")
    if subtitle:
        ax.text(0.0, 1.018, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.3, color=COLOR_MUTED)


def _frame_figure(fig, heading, meta, footer):
    fig.text(0.014, 0.988, heading, ha="left", va="top", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.986, 0.986, meta, ha="right", va="top", fontsize=5.8, color=COLOR_MUTED)
    fig.add_artist(Line2D([0.014, 0.986], [0.958, 0.958], transform=fig.transFigure, color=COLOR_SPINE, linewidth=0.5))
    fig.text(0.014, 0.011, footer, ha="left", va="bottom", fontsize=4.8, color=COLOR_MUTED)
    fig.add_artist(Rectangle((0.005, 0.005), 0.99, 0.99, transform=fig.transFigure, fill=False, edgecolor=COLOR_SPINE, linewidth=0.5))


def _figure_legend(fig, handles, labels, y=0.95):
    legend = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.011, y),
        ncol=len(handles),
        frameon=False,
        fontsize=5.8,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=2.6,
    )
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)
    return legend


def _figure_meta(report):
    game = report.get("game") or {}
    nash = game.get("nash") or {}
    seeds = sorted({str(t.get("seed")) for t in report.get("trials", [])}, key=lambda s: (not s.lstrip("-").isdigit(), int(s) if s.lstrip("-").isdigit() else 0, s))
    epochs = max((t.get("epochs") or 0 for t in report.get("trials", [])), default=0)
    parts = []
    if game:
        parts.append(f"N={game.get('num_agents')}")
        parts.append(f"T={game.get('comfort_threshold')}")
        parts.append(f"A*={'/'.join(str(a) for a in nash.get('pure_attendance') or [])}")
        if nash.get("mixed_go_probability") is not None:
            parts.append(f"p*={nash['mixed_go_probability']:.3f}")
    parts.append(f"SEEDS {'/'.join(seeds) if seeds else '-'}")
    parts.append(f"{epochs} EPOCHS")
    return "  ·  ".join(parts)


def _figure_footer(report):
    source = (report.get("source") or {}).get("experiment_dir", "")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"SOURCE {source}  ·  RENDERED {stamp}  ·  {FIGURE_SCHEMA}"


def _rolling(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < 3:
        return values
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _identity_rule(ax, label):
    ax.spines["top"].set_color(CONDITION_COLORS.get(label, COLOR_TEXT))
    ax.spines["top"].set_linewidth(1.4)


def _reference_line(ax, y, label, color, style="-"):
    ax.axhline(y, color=color, linewidth=0.6, linestyle=style, alpha=0.85, zorder=1.5)
    ax.text(0.995, y, f"{label} ", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.2, color=COLOR_MUTED, zorder=6)


def _draw_condition_series(ax, report, labels, key, skip_first=False):
    peak = 0.0
    for label in labels:
        color = CONDITION_COLORS.get(label, COLOR_TEXT)
        start = 1 if skip_first else 0
        for trial in _trials_for(report, label):
            epochs = _arr(trial["series"].get("epoch"))[start:]
            values = _arr(trial["series"].get(key))[start:]
            valid = np.isfinite(values)
            if valid.any():
                ax.plot(epochs[valid], values[valid], color=color, linewidth=0.45, alpha=0.22, zorder=2)
                peak = max(peak, float(np.nanmax(values[valid])))
        series = report["conditions"][label]["series"]
        epochs = _arr(series.get("epoch"))[start:]
        mean = _arr(series.get(f"{key}_mean"))[start:]
        sd = _arr(series.get(f"{key}_sd"))[start:]
        valid = np.isfinite(mean)
        if not valid.any():
            continue
        spread = np.where(np.isfinite(sd), sd, 0.0)
        ax.fill_between(epochs[valid], (mean - spread)[valid], (mean + spread)[valid], color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(epochs[valid], mean[valid], color=color, linewidth=1.25, zorder=3)
        last = np.flatnonzero(valid)[-1]
        ax.scatter([epochs[last]], [mean[last]], s=15, color=color, edgecolors=COLOR_BG, linewidths=1.0, zorder=4)
        peak = max(peak, float(np.nanmax((mean + spread)[valid])))
    return peak


def _mark_changepoints(ax, report, labels, key):
    for row, label in enumerate(labels):
        cp = _changepoint(report, label, key)
        if not cp or cp.get("epoch") is None:
            continue
        color = CONDITION_COLORS.get(label, COLOR_TEXT)
        ax.axvline(cp["epoch"], color=color, linewidth=0.6, alpha=0.6, zorder=2)
        r2 = cp.get("r2")
        text = f" τ={cp['epoch']}" + (f"  R²={r2:.2f}" if r2 is not None else "")
        ax.text(cp["epoch"], 0.965 - 0.07 * row, text, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=5.2, color=COLOR_MUTED, zorder=6)


def _draw_phase_portrait(ax, report, labels, n_agents, nash):
    xs, ys = [], []
    for label in labels:
        series = report["conditions"][label]["series"]
        x = _rolling(_arr(series.get("deception_index_mean")), PORTRAIT_SMOOTHING)
        y = _rolling(_arr(series.get("bar_attendance_mean")) / float(n_agents), PORTRAIT_SMOOTHING)
        epochs = _arr(series.get("epoch"))
        valid = np.isfinite(x) & np.isfinite(y)
        x, y, epochs = x[valid], y[valid], epochs[valid]
        if x.size < 2:
            continue
        xs.append(x)
        ys.append(y)
        color = CONDITION_COLORS.get(label, COLOR_TEXT)
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.12, 1.0, len(segments))
        ax.add_collection(LineCollection(segments, colors=[to_rgba(color, a) for a in alphas], linewidths=1.2, capstyle="round", joinstyle="round", zorder=3))
        ax.scatter([x[0]], [y[0]], s=18, facecolors=COLOR_BG, edgecolors=color, linewidths=0.9, zorder=4)
        ax.scatter([x[-1]], [y[-1]], s=22, color=color, edgecolors=COLOR_BG, linewidths=1.0, zorder=5)
        ax.annotate(f"t{int(epochs[0])}", (x[0], y[0]), xytext=(5, -9), textcoords="offset points", fontsize=5.2, color=COLOR_MUTED)
        ax.annotate(f"t{int(epochs[-1])}", (x[-1], y[-1]), xytext=(5, 4), textcoords="offset points", fontsize=5.2, color=COLOR_MUTED)
        cp = _changepoint(report, label, "deception_index")
        if cp and cp.get("epoch") is not None:
            idx = int(np.argmin(np.abs(epochs - cp["epoch"])))
            ax.scatter([x[idx]], [y[idx]], marker="D", s=20, facecolors=COLOR_BG, edgecolors=color, linewidths=0.9, zorder=6)
            ax.annotate(f"τ={int(epochs[idx])}", (x[idx], y[idx]), xytext=(5, -9), textcoords="offset points", fontsize=5.2, color=COLOR_MUTED)
    if xs:
        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        ax.set_xlim(0.0, max(0.05, float(all_x.max()) * 1.18))
        ax.set_ylim(max(0.0, float(all_y.min()) - 0.08), 1.06)
    primary = nash.get("primary_attendance")
    if primary is not None and n_agents:
        _reference_line(ax, primary / float(n_agents), "NASH A*/N", COLOR_WHITE)


def plot_phase_transitions(report, output_base, formats=("png", "pdf"), dpi=300):
    labels = _present_conditions(report)
    game = report.get("game") or {}
    nash = game.get("nash") or {}
    n_agents = game.get("num_agents") or max((t.get("agents") or 0 for t in report.get("trials", [])), default=1) or 1
    threshold = game.get("comfort_threshold")
    with plt.rc_context(_terminal_rc()):
        fig = plt.figure(figsize=(15.0, 8.8))
        grid = fig.add_gridspec(2, 3, left=0.04, right=0.985, top=0.875, bottom=0.075, wspace=0.17, hspace=0.36)
        axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]

        ax = axes[0]
        _style_panel(ax, 1, "ATTENDANCE VS NASH EQUILIBRIUM", "MEAN ± 1 SD ACROSS SEEDS  ·  THIN TRACES = INDIVIDUAL SEEDS")
        peak = _draw_condition_series(ax, report, labels, "bar_attendance")
        ceiling = max(peak, float(n_agents), float(threshold or 0)) * 1.08
        if nash.get("primary_attendance") is not None:
            _reference_line(ax, nash["primary_attendance"], f"NASH A*={nash['primary_attendance']}", COLOR_WHITE)
        mixed = nash.get("mixed_expected_attendance")
        if mixed is not None and nash.get("primary_attendance") is not None and abs(mixed - nash["primary_attendance"]) > 0.5:
            _reference_line(ax, mixed, f"MIXED E[A]={mixed:.1f}", COLOR_WHITE, style=(0, (1.5, 2.5)))
        if threshold is not None:
            note = "  (UNREACHABLE: N<T)" if game.get("num_agents") and game["num_agents"] < threshold else ""
            _reference_line(ax, threshold, f"COMFORT T={threshold}{note}", COLOR_THRESHOLD, style=(0, (4, 3)))
        _mark_changepoints(ax, report, labels, "bar_attendance")
        ax.set_ylim(0.0, ceiling)
        ax.set_xlabel("EPOCH")
        ax.set_ylabel("AGENTS IN BAR")

        ax = axes[1]
        _style_panel(ax, 2, "DECEPTION INDEX", "SHARE OF AGENTS WHOSE STATED INTENTION ≠ REALISED LOCATION")
        peak = _draw_condition_series(ax, report, labels, "deception_index")
        _mark_changepoints(ax, report, labels, "deception_index")
        ax.set_ylim(0.0, min(1.0, max(0.1, peak * 1.25)))
        ax.set_xlabel("EPOCH")
        ax.set_ylabel("DECEPTION INDEX")

        ax = axes[2]
        _style_panel(ax, 3, "ε-NASH REGRET", "MEAN EX-POST UNILATERAL DEVIATION GAIN PER AGENT  ·  0 = EXACT NASH")
        peak = _draw_condition_series(ax, report, labels, "regret")
        _mark_changepoints(ax, report, labels, "nash_gap")
        ax.set_ylim(0.0, max(0.05, peak * 1.2))
        ax.set_xlabel("EPOCH")
        ax.set_ylabel("REGRET (UTILITY)")

        ax = axes[3]
        _style_panel(ax, 4, "STRATEGY DRIFT", "D_KL(p_t || p_t-1)  ·  NATS  ·  JEFFREYS α=0.5  ·  4-STRATEGY SPACE")
        peak = _draw_condition_series(ax, report, labels, "kl_step", skip_first=True)
        ax.set_ylim(0.0, max(0.01, peak * 1.2))
        ax.set_xlabel("EPOCH")
        ax.set_ylabel("KL DIVERGENCE (NATS)")

        ax = axes[4]
        _style_panel(ax, 5, "INEQUALITY OF SUCCESS", "MAX-NORMALISED GINI OF CUMULATIVE UTILITY")
        peak = _draw_condition_series(ax, report, labels, "gini")
        ax.set_ylim(0.0, max(0.05, peak * 1.25))
        ax.set_xlabel("EPOCH")
        ax.set_ylabel("GINI")

        ax = axes[5]
        _style_panel(ax, 6, "PHASE PORTRAIT", f"(DECEPTION, ATTENDANCE/N)  ·  {PORTRAIT_SMOOTHING}-EPOCH ROLLING MEAN  ·  HOLLOW = START  ·  FILLED = END  ·  DIAMOND = τ")
        _draw_phase_portrait(ax, report, labels, n_agents, nash)
        ax.set_xlabel("DECEPTION INDEX")
        ax.set_ylabel("ATTENDANCE / N")

        handles = [Line2D([0], [0], color=CONDITION_COLORS.get(l, COLOR_TEXT), linewidth=1.6) for l in labels]
        names = [CONDITION_NAMES.get(l, l.upper()) for l in labels]
        handles += [
            Line2D([0], [0], color=COLOR_WHITE, linewidth=0.8),
            Line2D([0], [0], color=COLOR_THRESHOLD, linewidth=0.8, linestyle=(0, (4, 3))),
        ]
        names += ["NASH REFERENCE", "COMFORT THRESHOLD"]
        _figure_legend(fig, handles, names)
        _frame_figure(fig, "EL FAROL // PHASE TRANSITION ATLAS", _figure_meta(report), _figure_footer(report))
        paths = _save_figure(fig, output_base, formats, dpi)
        plt.close(fig)
    return paths


def _style_colorbar(cbar, label):
    cbar.outline.set_edgecolor(COLOR_SPINE)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(colors=COLOR_MUTED, labelsize=5.4, length=2, width=0.5, direction="in")
    cbar.set_label(label, color=COLOR_MUTED, fontsize=5.4, labelpad=4)


def _strategy_axis(ax):
    ax.set_yticks(range(len(STRATEGY_ORDER)))
    ax.set_yticklabels([STRATEGY_NAMES[s] for s in STRATEGY_ORDER])
    ax.set_ylim(len(STRATEGY_ORDER) - 0.5, -0.5)
    ax.grid(False)


def _draw_share_heatmap(ax, report, label, index):
    series = report["conditions"][label]["series"]
    epochs = _arr(series.get("epoch"))
    shares = np.vstack([_arr(series["strategy_shares"][name]) for name in STRATEGY_ORDER])
    cmap = LinearSegmentedColormap.from_list("ramp_share", [COLOR_RAMP_FLOOR, COLOR_RAMP_CEILING])
    _style_panel(ax, index, f"STRATEGY SHARE // {CONDITION_NAMES.get(label, label.upper())}", "MEAN POPULATION SHARE PER EPOCH ACROSS SEEDS  ·  SHARED SCALE")
    _identity_rule(ax, label)
    mesh = ax.pcolormesh(_edges(epochs), np.arange(len(STRATEGY_ORDER) + 1) - 0.5, np.ma.masked_invalid(shares), cmap=cmap, vmin=0.0, vmax=1.0, shading="flat", edgecolors=COLOR_BG, linewidth=0.5)
    _strategy_axis(ax)
    ax.set_xlabel("EPOCH")
    cbar = ax.figure.colorbar(mesh, cax=ax.inset_axes([1.012, 0.0, 0.009, 1.0]))
    _style_colorbar(cbar, "SHARE")


def _draw_difference_heatmap(ax, report, index):
    _style_panel(ax, index, "STRATEGY SHIFT // DELTA 2 − CONTROL", "DIFFERENCE IN MEAN SHARE  ·  RED = MORE UNDER BROADCAST  ·  CYAN = MORE UNDER CONTROL")
    conditions = report.get("conditions") or {}
    control = (conditions.get("control") or {}).get("series")
    treatment = (conditions.get("delta2") or {}).get("series")
    if not control or not treatment:
        ax.text(0.5, 0.5, "NO CONTRAST // BOTH CONDITIONS REQUIRED", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=COLOR_MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        return
    shared = sorted(set(control["epoch"]) & set(treatment["epoch"]))
    c_pos = {e: i for i, e in enumerate(control["epoch"])}
    t_pos = {e: i for i, e in enumerate(treatment["epoch"])}
    diff = np.vstack([
        [(_arr(treatment["strategy_shares"][name])[t_pos[e]] - _arr(control["strategy_shares"][name])[c_pos[e]]) for e in shared]
        for name in STRATEGY_ORDER
    ])
    finite = diff[np.isfinite(diff)]
    limit = max(0.05, float(np.abs(finite).max())) if finite.size else 0.05
    cmap = LinearSegmentedColormap.from_list("diverge", [COLOR_CONTROL, COLOR_NEUTRAL_MID, COLOR_DELTA2])
    mesh = ax.pcolormesh(_edges(shared), np.arange(len(STRATEGY_ORDER) + 1) - 0.5, np.ma.masked_invalid(diff), cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), shading="flat", edgecolors=COLOR_BG, linewidth=0.5)
    _strategy_axis(ax)
    ax.set_xlabel("EPOCH")
    cbar = ax.figure.colorbar(mesh, cax=ax.inset_axes([1.006, 0.0, 0.0045, 1.0]))
    _style_colorbar(cbar, "Δ SHARE")


def _draw_strategy_raster(ax, raster, label, index):
    name = CONDITION_NAMES.get(label, label.upper())
    if raster is None:
        _style_panel(ax, index, f"AGENT RASTER // {name}")
        ax.text(0.5, 0.5, "NO TRACE", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=COLOR_MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        return
    codes = np.asarray(raster["codes"])
    final = np.asarray(raster["final_utility"], dtype=float)
    order = np.argsort(-np.nan_to_num(final, nan=-np.inf), kind="stable")
    data = np.ma.masked_less(codes[order], 0)
    _style_panel(ax, index, f"AGENT RASTER // {name}", f"{raster['trial'].upper()}  ·  ROWS SORTED BY FINAL CUMULATIVE UTILITY (DESC)  ·  TOP = {final[order][0]:+.1f}  ·  BOTTOM = {final[order][-1]:+.1f}")
    _identity_rule(ax, label)
    cmap = ListedColormap([STRATEGY_COLORS[s] for s in STRATEGY_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(STRATEGY_ORDER) + 0.5, 1.0), cmap.N)
    n_agents = codes.shape[0]
    ax.pcolormesh(_edges(raster["epochs"]), np.arange(n_agents + 1) - 0.5, data, cmap=cmap, norm=norm, shading="flat", edgecolors=COLOR_BG, linewidth=0.2)
    ax.set_ylim(n_agents - 0.5, -0.5)
    ticks = sorted(set([0] + list(range(9, n_agents, 10)) + [n_agents - 1]))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"#{t + 1}" for t in ticks])
    ax.grid(False)
    ax.set_xlabel("EPOCH")
    ax.set_ylabel("AGENT RANK")


def plot_strategy_heatmaps(report, rasters, output_base, formats=("png", "pdf"), dpi=300):
    labels = _present_conditions(report)
    pair = [l for l in CONDITION_ORDER if l in labels] or labels[:2]
    with plt.rc_context(_terminal_rc()):
        fig = plt.figure(figsize=(15.0, 11.6))
        grid = fig.add_gridspec(3, 2, height_ratios=[0.85, 0.85, 2.5], left=0.105, right=0.955, top=0.895, bottom=0.055, wspace=0.26, hspace=0.34)
        index = 1
        for column, label in enumerate(pair[:2]):
            _draw_share_heatmap(fig.add_subplot(grid[0, column]), report, label, index)
            index += 1
        _draw_difference_heatmap(fig.add_subplot(grid[1, :]), report, index)
        index += 1
        for column, label in enumerate(pair[:2]):
            _draw_strategy_raster(fig.add_subplot(grid[2, column]), (rasters or {}).get(label), label, index)
            index += 1
        handles = [Patch(facecolor=STRATEGY_COLORS[s], edgecolor=COLOR_SPINE if s == "honest_stay" else STRATEGY_COLORS[s], linewidth=0.5) for s in STRATEGY_ORDER]
        _figure_legend(fig, handles, [STRATEGY_NAMES[s] for s in STRATEGY_ORDER])
        _frame_figure(fig, "EL FAROL // STRATEGY HEATMAPS", _figure_meta(report), _figure_footer(report))
        paths = _save_figure(fig, output_base, formats, dpi)
        plt.close(fig)
    return paths


def _save_figure(fig, output_base, formats, dpi):
    directory = os.path.dirname(output_base)
    if directory:
        os.makedirs(directory, exist_ok=True)
    paths = []
    for fmt in formats:
        path = f"{output_base}.{str(fmt).lower().lstrip('.')}"
        fig.savefig(path, dpi=dpi, facecolor=COLOR_BG, edgecolor="none")
        paths.append(path)
    return paths


def _seed_sort_key(seed):
    text = str(seed)
    return (0, int(text), text) if text.lstrip("-").isdigit() else (1, 0, text)


def _select_raster_trials(trials, seed=None):
    by_condition = {}
    for trial in trials:
        if trial.get("status", "complete") != "complete":
            continue
        by_condition.setdefault(trial["condition"], []).append(trial)
    if seed is not None:
        return {label: next((t for t in members if str(t["seed"]) == str(seed)), None) for label, members in by_condition.items()}
    control_seeds = {str(t["seed"]) for t in by_condition.get("control", [])}
    treatment_seeds = {str(t["seed"]) for t in by_condition.get("delta2", [])}
    shared = sorted(control_seeds & treatment_seeds, key=_seed_sort_key)
    if shared:
        return {label: next((t for t in members if str(t["seed"]) == shared[0]), None) for label, members in by_condition.items()}
    return {label: sorted(members, key=lambda t: _seed_sort_key(t["seed"]))[0] for label, members in by_condition.items() if members}


def _load_rasters(selected):
    from analyzer import load_trial

    rasters = {}
    for label, trial in selected.items():
        if trial is None:
            rasters[label] = None
            continue
        try:
            data = load_trial(trial)
        except (ValueError, OSError, KeyError, IndexError):
            rasters[label] = None
            continue
        rasters[label] = {
            "trial": trial["trial"],
            "seed": trial["seed"],
            "codes": data["strategy"],
            "epochs": data["epochs"],
            "final_utility": data["cumulative"][:, -1],
        }
    return rasters


def generate_sweep_figures(experiment_dir="outputs/experiment", figures_dir=None, formats=("png", "pdf"), seed=None, dpi=300, bootstrap=200, editorial=True):
    from analyzer import build_report, discover_trials

    report = build_report(experiment_dir, bootstrap=bootstrap)
    if not report.get("trials"):
        raise RuntimeError(f"no completed trials with CSV traces under {experiment_dir}")
    figures_dir = figures_dir or os.path.join(experiment_dir, "figures")
    trials, _ = discover_trials(experiment_dir)
    selected = _select_raster_trials(trials, seed)
    rasters = _load_rasters(selected)
    paths = []
    paths += plot_phase_transitions(report, os.path.join(figures_dir, "phase_transitions"), formats, dpi)
    paths += plot_strategy_heatmaps(report, rasters, os.path.join(figures_dir, "strategy_heatmaps"), formats, dpi)
    control_trial = selected.get("control")
    treatment_trial = selected.get("delta2")
    if editorial and control_trial and treatment_trial:
        try:
            paths.append(
                generate_figures_from_csv(
                    control_csv=control_trial["epoch_csv"],
                    delta2_csv=treatment_trial["epoch_csv"],
                    output_path=os.path.join(figures_dir, f"editorial_seed_{control_trial['seed']}.pdf"),
                )
            )
        except Exception as exc:
            print(f"editorial figure skipped: {type(exc).__name__}: {exc}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=str, default=None)
    parser.add_argument("--delta2", type=str, default=None)
    parser.add_argument("--output", type=str, default="el_farol_analysis.pdf")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--control-agent-csv", type=str, default=None)
    parser.add_argument("--delta2-agent-csv", type=str, default=None)
    parser.add_argument("--experiment-dir", type=str, default="outputs/experiment")
    parser.add_argument("--figures-dir", type=str, default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-editorial", action="store_true")
    args = parser.parse_args()

    if args.control or args.delta2:
        if not (args.control and args.delta2):
            parser.error("--control and --delta2 must be supplied together")
        out_file = generate_figures_from_csv(
            control_csv=args.control,
            delta2_csv=args.delta2,
            output_path=args.output,
            window=args.window,
            control_agent_csv=args.control_agent_csv,
            delta2_agent_csv=args.delta2_agent_csv,
        )
        print(f"Publication-ready figure successfully generated: {out_file}")
        return

    try:
        paths = generate_sweep_figures(
            experiment_dir=args.experiment_dir,
            figures_dir=args.figures_dir,
            formats=args.formats,
            seed=args.seed,
            dpi=args.dpi,
            editorial=not args.no_editorial,
        )
    except RuntimeError as exc:
        print(f"visualizer: {exc}")
        sys.exit(1)
    for path in paths:
        print(f"figure written: {path}")


if __name__ == "__main__":
    main()
