import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mplsoccer import Pitch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUT_DIR / "CSV-JSON"
PLOTS_DIR = OUTPUT_DIR / "plots"

# ----------------------------- Config ---------------------------------------
CSV_PATH = CSV_DIR / "classification_median/explainability/extended_all/cls_shap_meanabs.csv"
OUT_PATH = PLOTS_DIR / "ZoneAnalysis" / "zone_shap_heatmap.png"

CSV_PATH_DEF = CSV_DIR / "DefenceAnalysis/classification_median/explainability/extended_all/cls_shap_meanabs.csv"
OUT_PATH_DEF = PLOTS_DIR / "DefenceAnalysis" / "zone_shap_heatmap.png"

# StatsBomb pitch size
PITCH_LENGTH = 120
PITCH_WIDTH = 80

# Grid configuration (global 6×4 of 20×20 covering whole pitch)
COLS = 6
ROWS = 4
CELL_W = PITCH_LENGTH // COLS   # 20
CELL_H = PITCH_WIDTH // ROWS    # 20

# Zone numbering matrix (bottom→top rows, left→right cols)
# Matches your established convention:
#   [[4, 8, 12, 16, 20, 24],
#    [3, 7, 11, 15, 19, 23],
#    [2, 6, 10, 14, 18, 22],
#    [1, 5,  9, 13, 17, 21]]
ZONE_MATRIX = [
    [4, 8, 12, 16, 20, 24],
    [3, 7, 11, 15, 19, 23],
    [2, 6, 10, 14, 18, 22],
    [1, 5,  9, 13, 17, 21],
]

TARGET_MIN, TARGET_MAX = 13, 24
TARGET_ZONES = set(range(TARGET_MIN, TARGET_MAX + 1))

# ----------------------------- Helpers --------------------------------------

def parse_zone_from_feature(feat: str):
    """
    Extract zone number from patterns like 'zone_14_count'.
    Returns int or None.
    """
    m = re.search(r"zone[_\s]?(\d+)_count$", str(feat), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def load_zone_shap(csv_path: str):
    """
    Read CSV and return dict {zone_id: mean_raw_shap} for zones 13..24.
    """
    df = pd.read_csv(str(csv_path))
    if "feature" not in df.columns or "mean_raw_shap" not in df.columns:
        raise ValueError("CSV must contain 'feature' and 'mean_raw_shap' columns.")

    zone_vals = {}
    for _, row in df.iterrows():
        z = parse_zone_from_feature(row["feature"])
        if z is None or z not in TARGET_ZONES:
            continue
        zone_vals[z] = float(row["mean_raw_shap"])

    if not zone_vals:
        raise ValueError("No matching zone_*_count rows found for zones 13..24.")

    return zone_vals

def zone_at(row: int, col: int) -> int:
    return ZONE_MATRIX[row][col]

def draw_heatmap_for_zone_values(zone_vals: dict, out_path: str, title: str):
    # Compute normalization for opacity
    max_abs = max(abs(v) for v in zone_vals.values()) if zone_vals else 1.0
    if max_abs == 0:
        max_abs = 1.0

    # Set up pitch
    pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white")
    fig, ax = pitch.draw(figsize=(10, 7))

    # Draw global grid, but paint only zones 13..24
    for r in range(ROWS):
        for c in range(COLS):
            zid = zone_at(r, c)
            x0 = c * CELL_W
            y0 = r * CELL_H

            if zid in TARGET_ZONES:
                val = zone_vals.get(zid, 0.0)
                alpha = min(abs(val) / max_abs, 1.0)
                face = (1.0, 0.0, 0.0) if val >= 0 else (0.0, 0.2, 1.0)
                rect = Rectangle(
                    (x0, y0), CELL_W, CELL_H,
                    linewidth=0.8,
                    edgecolor="black",
                    facecolor=face,
                    alpha=alpha,
                    zorder=3,
                )
                ax.add_patch(rect)
                ax.text(
                    x0 + CELL_W / 2.0,
                    y0 + CELL_H / 2.0,
                    str(zid),
                    ha="center", va="center",
                    fontsize=9, color="black",
                    fontweight="bold", zorder=5
                )

    ax.set_title(title)
    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap → {out_path}")

# ----------------------------- Main -----------------------------------------

def main():
    # Heatmap 1 — CLresults (attack classification)
    zone_vals = load_zone_shap(CSV_PATH)
    draw_heatmap_for_zone_values(zone_vals, OUT_PATH, "SHAP Zone Impact Heatmap (Opponent Half)")
    
    # Heatmap 2 — Defence analysis (optional)
    if CSV_PATH_DEF.exists():
        zone_vals_def = load_zone_shap(CSV_PATH_DEF)
        draw_heatmap_for_zone_values(zone_vals_def, OUT_PATH_DEF, "SHAP Zone Impact Heatmap (Opponent Half) — Defence")
    else:
        print(f"[WARN] Defence CSV not found: {CSV_PATH_DEF} — skipping defence heatmap.")

def run():
    """Entry point wrapper for main.py compatibility."""
    main()

if __name__ == "__main__":
    main()