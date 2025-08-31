#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import os

# Resolve project root (BASE_DIR) so paths are stable regardless of CWD
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent.parent  # project-root


# -----------------------------
# Dynamic config (resolved in main)
# -----------------------------
DEFAULT_INPUT: Path | None = None  # computed from selected_k
DEFAULT_PLOTS_DIR: Path | None = None
DEFAULT_CSV_DIR: Path | None = None

# Cluster names are derived from selected_k at runtime
CLUSTER_NAMES: Dict[int, str] = {}

def make_cluster_names(k: int) -> Dict[int, str]:
    """Return generic names Cluster 0..k-1."""
    return {i: f"Cluster {i}" for i in range(int(k))}

REQUIRED_COLS = [
    "cluster_label",
    "Matchweek",
    "minute_absolute",
    "XG",
    "goal_scored",
]

# Text output directory
TEXT_DIR = BASE_DIR / "outputs" / "text"

# Prompt text for AI-generated continuation of the visual analysis section
visuals_prompt = (
        "Continue writing Section 2 – Attacking Patterns Overview directly after the existing cluster descriptions.\n"
        "Use only the provided CSVs and charts for insights—do not invent data. Keep it concise (max 3–4 short paragraphs).\n\n"
        "Files & visuals to use (place where noted):\n"
        "\t1. summary_by_cluster.csv → insert here as a compact table when comparing cluster outputs.\n"
        "\t2. fig_goals_vs_high_threat.png → insert when comparing goal counts and % high-threat attacks.\n"
        "\t3. fig_high_threat_timing.png → insert when explaining when each cluster creates high-threat attacks.\n"
        "\t4. fig_attack_distribution_over_time.png → insert when showing overall attack timing profiles.\n\n"
        "Writing focus:\n"
        "• Compare efficiency and threat creation between clusters (link to playstyle traits from earlier in Section 2).\n"
        "• Highlight standout patterns (e.g., clusters with high threat but low goals, or peaks in specific match phases).\n"
        "• Keep sentences tight and focused—avoid long statistical explanations.\n"
        "• Maintain a professional, coach-friendly tone consistent with the earlier text.\n"
        "• For each paragraph, finish with one short tactical takeaway line beginning with phrases like ‘This suggests that…’ or ‘This indicates…’.\n"
    )


def call_gemini_with_prompt_for_visuals(prompt_text: str, k_value: int) -> str:
    """Load VisualAnalysis artifacts for the selected K, build a prompt, call Gemini, return text."""
    from google import genai
    import os

    # NOTE: Uses env var if present; otherwise falls back to the provided key.
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyD3Y6wabXe5rMidJTpmk6785rSI4K2gjvk")

    vis_csv_dir = f"outputs/CSV-JSON/VisualAnalysis/KMeans_k{k_value}"
    vis_plots_dir = f"outputs/plots/VisualAnalysis/KMeans_k{k_value}"

    summary_fp = os.path.join(vis_csv_dir, "summary_by_cluster.csv")
    goals_fp = os.path.join(vis_plots_dir, "fig_goals_vs_high_threat.png")
    timing_fp = os.path.join(vis_plots_dir, "fig_high_threat_timing.png")
    distrib_fp = os.path.join(vis_plots_dir, "fig_attack_distribution_over_time.png")

    def read_file_trim(path: str, max_chars: int = 50_000) -> str:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        return data if len(data) <= max_chars else data[:max_chars] + "\n... [TRIMMED]"

    # Ingest compact CSV text for the model context
    summary_txt = read_file_trim(summary_fp, 20_000)

    full_prompt = f"""{prompt_text}\n\n--- summary_by_cluster.csv (compact table) ---\n{summary_txt}\n\n--- Visuals (use where noted) ---\n[IMAGE: {goals_fp}]\n[IMAGE: {timing_fp}]\n[IMAGE: {distrib_fp}]\n"""

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=full_prompt,
    )
    text = getattr(resp, "text", None)
    if text is None:
        print("[GEMINI][WARN] Empty response.text in call_gemini_with_prompt_for_visuals; returning empty string.")
        text = ""
    return text


PHASE_BINS = [0, 15, 30, 45, 60, 75, 105]
PHASE_LABELS = ["0–15", "16–30", "31–45", "46–60", "61–75", "76–90+"]

# Radar chart configuration
SELECTED_RADAR_FEATURES = [
    "duration",
    "num_pass",
    "number_of_players_involved",
    "distance_covered",
    "num_long_passes",
]
RADAR_PRETTY_LABELS = {
    "duration": "Duration",
    "num_pass": "Pass",
    "number_of_players_involved": "Players",
    "distance_covered": "Dist",
    "num_long_passes": "Long pass",
}

# -----------------------------
# Utilities
# -----------------------------

def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_to_report(text: str, path: Path) -> None:
    """Append the given text to the report file, ensuring the directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _coerce_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    cols_num = ["cluster_label", "Matchweek", "minute_absolute", "XG", "goal_scored"]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows missing any of the above
    df = df.dropna(subset=cols_num).copy()
    # Cast types
    df["cluster_label"] = df["cluster_label"].astype(int)
    df["Matchweek"] = df["Matchweek"].astype(int)
    df["minute_absolute"] = df["minute_absolute"].astype(float)
    df["XG"] = df["XG"].astype(float)
    df["goal_scored"] = df["goal_scored"].astype(int)
    # Logical minutes (allow up to 105 for added time)
    df = df[(df["minute_absolute"] >= 0) & (df["minute_absolute"] <= 105)].copy()
    # Cluster names
    df["cluster_name"] = df["cluster_label"].map(CLUSTER_NAMES)
    return df


def _time_bin_5m(minute: pd.Series) -> pd.Series:
    # Bin to 5-min starts: 0,5,10,...,90. Values >90 are clipped to 90.
    start = (minute // 5) * 5
    start = start.clip(lower=0, upper=90)
    return start.astype(int)


def _bar_group_positions(n_groups: int, n_bars: int, width: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(n_groups)
    bar_width = width / max(1, n_bars)
    offsets = -width / 2 + bar_width / 2 + np.arange(n_bars) * bar_width
    return x, offsets


# -----------------------------
# Plots (matplotlib only)
# -----------------------------

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# Steps 0–6
# -----------------------------

def preprocess(path_csv: Path) -> Tuple[pd.DataFrame, float]:
    if not path_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path_csv}. "
            "Ensure clustering has produced clustered_attacks.csv under outputs/CSV-JSON/clustering/KMeans_k{selected_k} "
            "or pass --input pointing to the correct file."
        )
    df = pd.read_csv(path_csv)
    # Ensure we have the expected columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = _coerce_and_clean(df)
    median_xg = float(df["XG"].median())
    df["high_threat"] = (df["XG"] > median_xg).astype(int)
    return df, median_xg


def step1_summary_by_cluster(df: pd.DataFrame, csv_dir: Path, plots_dir: Path) -> pd.DataFrame:
    # Compute totals
    grp = df.groupby(["cluster_label", "cluster_name"], as_index=False)
    agg = grp.agg(
        total_attacks=("cluster_label", "size"),
        goals_scored=("goal_scored", "sum"),
        high_threat_attacks=("high_threat", "sum"),
    )
    # Rates
    agg["high_threat_rate_pct"] = (agg["high_threat_attacks"] / agg["total_attacks"] * 100).replace([np.inf, -np.inf], np.nan)
    agg["success_metric"] = (agg["goals_scored"] / agg["high_threat_attacks"]).replace([np.inf, -np.inf], np.nan)
    out_csv = csv_dir / "summary_by_cluster.csv"
    agg.rename(columns={"cluster_label": "cluster_label", "cluster_name": "cluster_name"}, inplace=True)
    agg.to_csv(out_csv, index=False)

    # Plot: Goals vs % High-Threat
    clusters = [CLUSTER_NAMES.get(i, str(i)) for i in sorted(df["cluster_label"].unique())]
    n = len(clusters)
    x, offsets = _bar_group_positions(n_groups=n, n_bars=2)

    plt.figure(figsize=(10, 6))
    # Align data to cluster order
    agg_ordered = agg.set_index("cluster_name").reindex(clusters)
    plt.bar(x + offsets[0], agg_ordered["goals_scored"].values, width=0.8 / 2, label="Goals Scored")
    plt.bar(x + offsets[1], agg_ordered["high_threat_rate_pct"].values, width=0.8 / 2, label="% High-Threat Attacks")
    plt.xticks(x, [f"Cluster {i} – {name}" for i, name in enumerate(clusters)])
    plt.ylabel("Count / Percent")
    plt.title("Goals vs High-Threat Attacks by Cluster")
    plt.legend()
    _save_fig(plots_dir / "fig_goals_vs_high_threat.png")

    return agg


def step3_high_threat_timing(df: pd.DataFrame, csv_dir: Path, plots_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["time_bin_start"] = _time_bin_5m(df["minute_absolute"])  # 0..90
    ht = df[df["high_threat"] == 1]
    timing = (
        ht.groupby(["time_bin_start", "cluster_label", "cluster_name"], as_index=False)
          .size()
          .rename(columns={"size": "high_threat_attacks"})
    )
    timing.to_csv(csv_dir / "high_threat_timing.csv", index=False)

    # Line plot per cluster
    clusters = [CLUSTER_NAMES.get(i, str(i)) for i in sorted(df["cluster_label"].unique())]
    plt.figure(figsize=(10, 6))
    for cname in clusters:
        sub = timing[timing["cluster_name"] == cname]
        plt.plot(sub["time_bin_start"], sub["high_threat_attacks"], marker='o', label=cname)
    plt.xlabel("Match minute (5-min bins)")
    plt.ylabel("High-threat attacks")
    plt.title("Timing of High-Threat Attacks by Cluster")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_fig(plots_dir / "fig_high_threat_timing.png")

    return timing


def step4_attack_distribution(df: pd.DataFrame, csv_dir: Path, plots_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["time_bin_start"] = _time_bin_5m(df["minute_absolute"])  # 0..90
    dist = (
        df.groupby(["time_bin_start", "cluster_label", "cluster_name"], as_index=False)
          .size()
          .rename(columns={"size": "num_attacks"})
    )
    dist.to_csv(csv_dir / "attack_distribution_over_time.csv", index=False)

    clusters = [CLUSTER_NAMES.get(i, str(i)) for i in sorted(df["cluster_label"].unique())]
    plt.figure(figsize=(10, 6))
    for cname in clusters:
        sub = dist[dist["cluster_name"] == cname]
        plt.plot(sub["time_bin_start"], sub["num_attacks"], marker='o', label=cname)
    plt.xlabel("Match minute (5-min bins)")
    plt.ylabel("Number of attacks")
    plt.title("Distribution of Attacking Sequences over Match Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_fig(plots_dir / "fig_attack_distribution_over_time.png")

    return dist


def step5_phase_kpis(df: pd.DataFrame, csv_dir: Path, plots_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["phase"] = pd.cut(
        df["minute_absolute"],
        bins=PHASE_BINS,
        labels=PHASE_LABELS,
        include_lowest=True,
        right=True,
        ordered=True,
    )
    # Ensure no unused categorical levels remain (avoids length mismatches in groupby)
    if str(df["phase"].dtype) == "category":
        df["phase"] = df["phase"].cat.remove_unused_categories()

    # KPIs per phase x cluster
    kpi = (
        df.groupby(["phase", "cluster_label", "cluster_name"], observed=True, as_index=False)
          .agg(
              attacks=("cluster_label", "size"),
              goals_scored=("goal_scored", "sum"),
              high_threat_rate_pct=("high_threat", "mean"),
          )
    )
    kpi["high_threat_rate_pct"] = kpi["high_threat_rate_pct"] * 100.0

    # Share of phase attacks
    phase_totals = (
        kpi.groupby("phase", observed=True, as_index=False)["attacks"]
          .sum()
          .rename(columns={"attacks": "attacks_in_phase"})
    )
    kpi = kpi.merge(phase_totals, on="phase", how="left")
    kpi["share_of_phase_attacks_pct"] = np.where(
        kpi["attacks_in_phase"] > 0,
        kpi["attacks"] / kpi["attacks_in_phase"] * 100.0,
        0.0,
    )
    kpi = kpi.drop(columns=["attacks_in_phase"])  # keep only needed columns

    out_csv = csv_dir / "phase_kpis.csv"
    kpi.to_csv(out_csv, index=False)

    # Grouped bars: high_threat_rate_pct by phase
    phases = PHASE_LABELS
    clusters = [CLUSTER_NAMES.get(i, str(i)) for i in sorted(df["cluster_label"].unique())]
    n_bars = len(clusters)
    x, offsets = _bar_group_positions(n_groups=len(phases), n_bars=n_bars)

    plt.figure(figsize=(12, 6))
    for idx, cname in enumerate(clusters):
        yvals = []
        for ph in phases:
            val = kpi[(kpi["phase"] == ph) & (kpi["cluster_name"] == cname)]["high_threat_rate_pct"].mean()
            yvals.append(0.0 if pd.isna(val) else float(val))
        plt.bar(np.arange(len(phases)) + offsets[idx], yvals, width=0.8 / max(1, n_bars), label=cname)

    plt.xticks(np.arange(len(phases)), phases)
    plt.xlabel("Match phase (15-min blocks)")
    plt.ylabel("% of high-threat attacks")
    plt.title("High-Threat Attack Rate by Phase and Cluster")
    plt.legend()
    _save_fig(plots_dir / "fig_phase_high_threat_rate.png")

    return kpi


def step6_coach_digest(df_cluster: pd.DataFrame, timing: pd.DataFrame, kpi: pd.DataFrame, median_xg: float, text_dir: Path) -> None:
    # Most goals (absolute)
    most_goals_row = df_cluster.sort_values(["goals_scored", "cluster_label"], ascending=[False, True]).head(1)
    most_goals_cluster = most_goals_row.iloc[0]["cluster_name"] if not most_goals_row.empty else "N/A"

    # Highest success metric
    best_success_row = df_cluster.sort_values(["success_metric", "cluster_label"], ascending=[False, True]).head(1)
    best_success_cluster = best_success_row.iloc[0]["cluster_name"] if not best_success_row.empty else "N/A"

    # Best time windows per cluster (top bins by high_threat_attacks)
    lines = []
    lines.append(f"median_xg: {median_xg:.4f}")
    lines.append(f"most_goals_cluster: {most_goals_cluster}")
    lines.append(f"best_success_cluster: {best_success_cluster}")
    lines.append("")

    for clabel, cname in CLUSTER_NAMES.items():
        sub_t = timing[timing["cluster_label"] == clabel]
        top_bins = (
            sub_t.sort_values("high_threat_attacks", ascending=False)
                 .head(3)["time_bin_start"].tolist()
        )
        best_windows = ", ".join([f"{int(b)}–{int(b)+5}" for b in top_bins]) if len(top_bins) else "None"

        sub_k = kpi[kpi["cluster_label"] == clabel]
        idxmax = sub_k["high_threat_rate_pct"].astype(float).idxmax() if not sub_k.empty else None
        best_phase = str(sub_k.loc[idxmax, "phase"]) if idxmax is not None else "None"

        lines.append(f"{cname} — best_time_windows: {best_windows}; best_phase: {best_phase}")

    (text_dir / "coach_digest.txt").write_text("\n".join(lines), encoding="utf-8")



# -----------------------------
# Radar chart step
# -----------------------------

def step7_radar_selected_features(df: pd.DataFrame, csv_dir: Path, plots_dir: Path) -> None:
    """Create radar charts for the five selected features using Matplotlib+Seaborn.

    Scaling rule: min–max per feature using the overall distribution from
    clustered_attacks (not just cluster means). This avoids polygons collapsing
    and keeps visual differences realistic.
    """
    # --- DEBUG Radar Features ---
    expected = [
        "duration",
        "num_pass",
        "number_of_players_involved",
        "distance_covered",
        "num_long_passes",
    ]
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    print("[RADAR] present features:", present)
    print("[RADAR] missing  features:", missing)
    # Guard: features present
    features = [f for f in SELECTED_RADAR_FEATURES if f in df.columns]
    if not features:
        return

    # Compute global min/max from all attacks
    mins = df[features].min()
    maxs = df[features].max()
    ranges = (maxs - mins).replace(0, np.nan)

    # Cluster means (raw)
    means_raw = (
        df.groupby(["cluster_label", "cluster_name"], as_index=False)[features]
          .mean()
    )

    # Scaled means 0–1 (safe where range==0)
    means_scaled = means_raw.copy()
    for f in features:
        if pd.isna(ranges[f]):
            means_scaled[f] = 0.5
        else:
            means_scaled[f] = (means_raw[f] - mins[f]) / ranges[f]

    # Persist scaling/meta (for auditability)
    scale_meta = pd.DataFrame({
        "feature": features,
        "global_min": [float(mins[f]) for f in features],
        "global_max": [float(maxs[f]) for f in features],
    })
    scale_meta.to_csv(csv_dir / "radar_feature_scaling_info.csv", index=False)

    means_joined = means_raw.merge(means_scaled, on=["cluster_label", "cluster_name"], suffixes=("", "_scaled"))
    means_joined.to_csv(csv_dir / "radar_cluster_means_raw_and_scaled.csv", index=False)

    # --- Radar plotting helpers (matplotlib polar) ---
    labels = [RADAR_PRETTY_LABELS.get(f, f) for f in features]
    n_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    def _plot_one(ax: plt.Axes, df_scaled: pd.DataFrame, title: str, show_legend: bool = True) -> None:
        sns.set(style="whitegrid")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.set_title(title, y=1.08, fontsize=14)

        # Ordered clusters
        ordered = (
            df_scaled[["cluster_label", "cluster_name"]]
            .drop_duplicates()
            .sort_values("cluster_label")
        )
        palette = sns.color_palette("Set2", len(ordered))
        color_map = {int(row.cluster_label): palette[i % len(palette)] for i, row in enumerate(ordered.itertuples(index=False))}

        for row in ordered.itertuples(index=False):
            clabel = int(row.cluster_label)
            cname = row.cluster_name
            vals_series = df_scaled[df_scaled["cluster_label"] == clabel][features].mean()
            vals = vals_series.tolist() + [vals_series.tolist()[0]]
            color = color_map[clabel]
            ax.plot(angles, vals, color=color, linewidth=2.5, marker="o", label=f"Cluster {clabel} – {cname}")
            ax.fill(angles, vals, color=color, alpha=0.25)

        if show_legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), frameon=False, fontsize=10)

    # Combined chart
    fig, ax = plt.subplots(figsize=(8.8, 8.8), subplot_kw=dict(polar=True))
    _plot_one(ax, means_scaled, "Cluster Profiles – Selected Features (min–max scaled)")
    # Reposition legend to top-right of figure and keep polar plot centered
    try:
        handles, legend_labels = ax.get_legend_handles_labels()
        if ax.legend_:
            ax.legend_.remove()
    except Exception:
        handles, legend_labels = [], []
    # Reserve space on the right for the legend
    fig.subplots_adjust(left=0.08, right=0.78, top=0.88, bottom=0.08)
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=10,
            ncol=1,
        )
    _save_fig(plots_dir / "fig_radar_selected_features_combined.png")

    # Individual cluster charts with raw-value annotations
    for clabel in sorted(means_raw["cluster_label"].unique()):
        fig, ax = plt.subplots(figsize=(8.8, 8.8), subplot_kw=dict(polar=True))
        _plot_one(
            ax,
            means_scaled[means_scaled["cluster_label"] == clabel],
            f"Cluster {clabel} – {CLUSTER_NAMES.get(clabel)}",
            show_legend=False,
        )
        # annotate raw means near vertices
        row_raw = means_raw[means_raw["cluster_label"] == clabel][features].mean()
        row_scaled = means_scaled[means_scaled["cluster_label"] == clabel][features].mean()
        for ang, v_scaled, (f, v_raw) in zip(angles[:-1], row_scaled.tolist(), row_raw.items()):
            r = min(max(float(v_scaled) + 0.06, 0.05), 0.98)
            ax.text(ang, r, f"{v_raw:.1f}", fontsize=9, ha="center", va="center")
        _save_fig(plots_dir / f"fig_radar_selected_features_cluster_{clabel}.png")

def main():
    parser = argparse.ArgumentParser(description="Generate temporal analysis visuals and tables for clustered attacks.")
    parser.add_argument("--input", help="Path to clustered_attacks.csv")
    parser.add_argument("--plots_dir", help="Directory to save plot outputs")
    parser.add_argument("--csv_dir", help="Directory to save CSV outputs")
    args = parser.parse_args()

    # Resolve selected_k first
    selected_k_path = TEXT_DIR / "selected_k.txt"
    try:
        selected_k = (selected_k_path.read_text(encoding="utf-8").strip())
        if not selected_k.isdigit():
            raise ValueError
    except Exception:
        selected_k = "3"  # safe default

    # Build defaults based on selected_k (CSV lives under CSV-JSON/clustering)
    default_input = BASE_DIR / f"outputs/CSV-JSON/clustering/KMeans_k{selected_k}/clustered_attacks.csv"
    default_plots_dir = BASE_DIR / f"outputs/plots/VisualAnalysis/KMeans_k{selected_k}"
    default_csv_dir = BASE_DIR / f"outputs/CSV-JSON/VisualAnalysis/KMeans_k{selected_k}"

    input_path = Path(args.input) if args.input else default_input
    plots_dir = Path(args.plots_dir) if args.plots_dir else default_plots_dir
    csv_dir = Path(args.csv_dir) if args.csv_dir else default_csv_dir

    # Normalize any provided relative paths to project BASE_DIR
    if not input_path.is_absolute():
        input_path = (BASE_DIR / input_path).resolve()
    if not plots_dir.is_absolute():
        plots_dir = (BASE_DIR / plots_dir).resolve()
    if not csv_dir.is_absolute():
        csv_dir = (BASE_DIR / csv_dir).resolve()

    # Ensure directories exist
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] selected_k={selected_k}")
    print(f"[INFO] input_path={input_path}")
    print(f"[INFO] plots_dir={plots_dir}")
    print(f"[INFO] csv_dir={csv_dir}")

    # Set cluster names dynamically
    global CLUSTER_NAMES
    try:
        k_int = int(selected_k)
    except Exception:
        k_int = 3
    CLUSTER_NAMES = make_cluster_names(k_int)

    # Step 0: preprocess
    df, median_xg = preprocess(input_path)

    summary_by_cluster = step1_summary_by_cluster(df, csv_dir, plots_dir)
    high_threat_timing = step3_high_threat_timing(df, csv_dir, plots_dir)
    attack_distribution = step4_attack_distribution(df, csv_dir, plots_dir)
    phase_kpis = step5_phase_kpis(df, csv_dir, plots_dir)
    step7_radar_selected_features(df, csv_dir, plots_dir)
    step6_coach_digest(summary_by_cluster, high_threat_timing, phase_kpis, median_xg, TEXT_DIR)

    # --- AI-generated visual analysis continuation ---
    try:
        visuals_report_text = call_gemini_with_prompt_for_visuals(visuals_prompt, int(k_int))
        final_report_path = TEXT_DIR / "FinalReport.txt"
        section_header = f"\n## Section 2 – Attacking Patterns Overview (continued, K={k_int})\n\n"
        visuals_report_text = visuals_report_text or ""
        append_to_report(section_header + visuals_report_text + "\n", final_report_path)
        print(f"[INFO] Gemini visual analysis appended to {final_report_path}")
    except Exception as e:
        print("[WARN] Gemini visual analysis generation skipped:", e)

    print("Artifacts written to:\n - plots:", plots_dir, "\n - csv:", csv_dir, "\n - text:", TEXT_DIR)

def run():
    """Entry point wrapper so that other modules can call VisualAnalysis."""
    main()

if __name__ == "__main__":
    main()