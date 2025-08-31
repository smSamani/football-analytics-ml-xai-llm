from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.k_selector_gui import show_k_selection_gui
# KneeLocator from kneed is optional but provides automatic elbow detection
try:
    from kneed import KneeLocator  # type: ignore
except ImportError:  # graceful fallback if kneed not installed
    KneeLocator = None  # type: ignore
def append_to_report(file_path: str, content: str) -> None:
    """Append text content to a report file (creates file if missing)."""
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)

# ----------------------------- CONFIGURATION (Updated for project-root) ----------------------------- #
DATA_PATH = Path("outputs/CSV-JSON/BaseFeatures.csv")

# Feature list
FEATURES: List[str] = [
    "duration",
    "distance_covered",
    "num_pass",
    "velocity",
    "number_of_players_involved",
    "num_dribbles",
    "num_duels",
    "num_crosses",
    "num_events",
    "num_long_passes",
    "zone_transition_count",
    "forward_movement_count",
    "backward_movement_count",
    "net_progress_ratio",
] + [f"zone_{i}_count" for i in range(1, 25)]


# ----------------------------- HELPER FUNCTIONS ----------------------------- #

def determine_optimal_k(inertias: List[float], k_range: List[int]) -> int:
    """Return optimal k using the ‘elbow’ point in the inertia curve.

    If kneed is available, use KneeLocator; otherwise, fall back to the point with the
    largest discrete second derivative (heuristic). If that also fails, return 3.
    """

    if KneeLocator is not None:
        try:
            kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
            if kl.elbow is not None:
                return int(kl.elbow)
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[WARN] KneeLocator failed: {exc}", file=sys.stderr)

    # Fallback: naive second-derivative heuristic
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    if len(second_diffs):
        elbow_idx = np.argmin(second_diffs) + 2  # +2 to align with k index (start 2)
        return k_range[elbow_idx]

    return 3  # sensible default


# ----------------------------- MAIN WORKFLOW ----------------------------- #

def run() -> None:  # noqa: C901 — complex but cohesive
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(
            f"[WARN] The following requested features are missing in the CSV and will be filled with 0: {missing_features}",
            file=sys.stderr,
        )
        for col in missing_features:
            df[col] = 0

    # Extract and preprocess features
    X = df[FEATURES].fillna(0).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Run KMeans clustering for the explicitly specified k values
    # ------------------------------------------------------------------
    ks_to_run = [3,4, 5]

    def run_kmeans_clustering(k: int) -> None:
        """Fit KMeans with k clusters and persist all requested outputs."""
        metrics_dir = Path("outputs/metrics/clustering") / f"KMeans_k{k}"
        plots_dir = Path("outputs/plots/clustering") / f"KMeans_k{k}"
        processed_dir = Path("outputs/CSV-JSON/clustering") / f"KMeans_k{k}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=12)
        labels = kmeans.fit_predict(X_scaled)

        # Add labels to a copy of the original dataframe to avoid contamination
        df_clustered = df.copy()
        df_clustered["cluster_label"] = labels

        # 1. Save clustered data
        df_clustered.to_csv(processed_dir / "clustered_attacks.csv", index=False)

        # 2. Save cluster centres (inverse-transformed)
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers_original, columns=FEATURES)
        centers_df.insert(0, "cluster_label", range(k))
        centers_df.to_csv(metrics_dir / "cluster_centers.csv", index=False)
        centers_df.to_csv(processed_dir / "cluster_centers.csv", index=False)

        # 3. Summary statistics
        summary_df = df_clustered.groupby("cluster_label")[FEATURES].mean().reset_index()
        summary_df.to_csv(metrics_dir / "cluster_summary.csv", index=False)
        summary_df.to_csv(processed_dir / "cluster_summary.csv", index=False)

        # 4. Pie chart of cluster distribution
        try:
            plt.figure(figsize=(6, 6))
            df_clustered["cluster_label"].value_counts().sort_index().plot.pie(
                autopct="%1.1f%%", startangle=90, cmap="tab20"
            )
            plt.ylabel("")
            plt.title(f"Cluster Distribution (k={k})")
            plt.tight_layout()
            plt.savefig(plots_dir / "cluster_distribution.png", dpi=300)
            plt.close()
        except Exception as exc:
            print(f"[WARN] Pie chart failed for k={k}: {exc}", file=sys.stderr)

        # 5. 2-D PCA scatter plot
        try:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                hue=labels,
                palette="tab10",
                legend="full",
                alpha=0.7,
            )

            # Overlay centroids projected into PCA space
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            # Draw centroids as white circles with black borders
            plt.scatter(
                centroids_pca[:, 0],
                centroids_pca[:, 1],
                c="white",
                edgecolors="black",
                s=250,
                marker="o",
                linewidth=1.5,
                label="Centroids",
                zorder=10
            )

            plt.title(f"PCA Cluster Visualization (k={k})")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "cluster_visualization.png", dpi=300)
            plt.close()
        except Exception as exc:
            print(f"[WARN] PCA visualization failed for k={k}: {exc}", file=sys.stderr)

        print(f"[INFO] Finished KMeans clustering for k={k}. Results saved to respective directories.")

    # Execute clustering for each k
    for k in ks_to_run:
        run_kmeans_clustering(k)

    print("[SUCCESS] All clustering runs completed. Outputs saved to respective directories.")

 # --------------------- Step 2.5: GUI for K Selection --------------------- #
    

    selected_k = show_k_selection_gui()
    print(f"✅ User selected K = {selected_k}")

    with open("outputs/text/selected_k.txt", "w") as f:
        f.write(str(selected_k))

    # Inline prompt for clustering analysis (no file read)
    base_prompt = (
        "You are acting as a professional football performance analyst tasked with reviewing "
        "clustering results for a full season's attacking sequences of Bayer Leverkusen.\n"
        "Objective\n"
        "Your job is to:\n"
        "Assign a short, sharp, football-coach-friendly name to each cluster (e.g., “Fast Direct Transitions”, “Structured Positional Build-Up”).\n"
        "Write a vivid, tactically rich paragraph for each cluster that:\n"
        "Clearly paints the style of play in football terms.\n"
        "Highlights what makes this cluster different from the others.\n"
        "References real match-context triggers (e.g., recoveries, pressing phases, final-third dominance).\n"
        "Provide a “Portfolio Summary” that explains:\n"
        "The share of each style across the season.\n"
        "The tactical value of each style and when it is most effective.\n"
        "How the combination of styles gives the team flexibility.\n\n"
        "Data Provided\n"
        "cluster_summary.csv – Summary stats per cluster (duration, passes, players involved, dribbles, crosses, long passes, zone transitions, net progress ratio, etc.).\n"
        "cluster_centers.csv – Numerical centroids for each feature in each cluster.\n"
        "clustered_attacks.csv – Each attack's assigned cluster, match week, goals scored, goals conceded.\n\n"
        "Writing Guidelines\n"
        "Avoid heavy statistical jargon — convert numbers into meaningful football insights.\n"
        "Focus on differences in tempo, width usage, build-up style, and shot creation approach.\n"
        "Identify likely tactical contexts (e.g., counterattack, sustained pressure, positional play).\n"
        "Always explain why the style is effective or risky.\n"
        "Present data points as supporting details (e.g., “~20 passes over 60 seconds” instead of raw column names).\n"
        "Explicitly compare clusters so that distinctions are obvious.\n"
        "Output Format\n"
        "Cluster X – [Name]\n"
        "Description: [Tactical narrative paragraph with differences from other clusters]\n"
        "Key traits:\n"
        "Bullet points with main statistical and tactical features (derived from data, written in coach-friendly terms)\n"
        "Portfolio Summary: [3–4 sentences connecting all clusters, their balance, and tactical implications]"
    )

    # --------------------- Step 2.6: Call Gemini with K-specific files --------------------- #
    from google import genai
    import textwrap

    def call_gemini_with_prompt_for_k(prompt_text: str, k_value: int) -> str:
        """Reads K-specific CSVs, composes a prompt, calls Gemini, and returns the text."""
        # Configure API key (prefer env var; fallback to provided key)
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyD3Y6wabXe5rMidJTpmk6785rSI4K2gjvk")

        # Resolve paths for selected K
        cluster_dir = f"outputs/CSV-JSON/clustering/KMeans_k{k_value}"
        summary_fp = os.path.join(cluster_dir, "cluster_summary.csv")
        centers_fp = os.path.join(cluster_dir, "cluster_centers.csv")
        attacks_fp = os.path.join(cluster_dir, "clustered_attacks.csv")

        # Read CSV contents (lightly trimmed to avoid overly long prompts)
        def read_file_trim(path: str, max_chars: int = 50_000) -> str:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            return data if len(data) <= max_chars else data[:max_chars] + "\n... [TRIMMED]"

        summary_txt = read_file_trim(summary_fp, 20_000)
        centers_txt = read_file_trim(centers_fp, 20_000)
        attacks_txt = read_file_trim(attacks_fp, 20_000)

        full_prompt = f"""{prompt_text}\n\n--- Cluster Summary (CSV) ---\n{summary_txt}\n\n--- Cluster Centers (CSV) ---\n{centers_txt}\n\n--- Clustered Attacks (CSV) ---\n{attacks_txt}\n"""

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=full_prompt
        )
        text = getattr(resp, "text", None)
        if text is None:
            print("[GEMINI][WARN] Empty response.text in call_gemini_with_prompt_for_k; returning empty string.")
            text = ""
        return text


    # Call Gemini and persist output
    report_text = call_gemini_with_prompt_for_k(base_prompt, selected_k)

    final_report_path = "outputs/text/FinalReport.txt"
    section_header = f"# Attacking Patterns Overview (K={selected_k})\n\n"
    report_text = report_text or ""
    append_to_report(final_report_path, section_header + report_text + "\n")
    print(f"✅ Gemini analysis appended → {final_report_path}")

if __name__ == "__main__":
    run()
