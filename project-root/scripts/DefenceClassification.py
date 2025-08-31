from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from xgboost import XGBClassifier
import shap

# ----------------------------- Configuration ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUT_DIR / "CSV-JSON" / "DefenceAnalysis"
PLOTS_DIR = OUTPUT_DIR / "plots" / "DefenceAnalysis"
METRICS_DIR = OUTPUT_DIR / "metrics" / "DefenceAnalysis"
TEXT_DIR = OUTPUT_DIR / "text" / "DefenceAnalysis"

DATA_PATH = OUTPUT_DIR / "CSV-JSON" / "DefenceBaseFeatures.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2  # proportion of **groups** held out for final test
N_JOBS = -1
N_ITER_SEARCH = 50

# Grouping preference for leakage-safe splits (first existing column will be used)
GROUP_CANDIDATES = ["Match Name", "Matchweek"]

# Columns to always exclude from features (IDs, outcomes, or obvious leakage)
EXCLUDE_COLS = [
    "Case number",  # row identifier
    "Match Name",   # grouping key / metadata
    "HorW",         # already in Match Name or elsewhere
    "end_point_id",
    "start_point_id",
    "start_index",
    "end_index",
    "XG",           # target for regression    
    "goal_scored",  # leakage (outcome)
    "Matchweek",    # grouping key / metadata
    "Leverkusen Goals",  # leakage w.r.t. match outcome context
    "Opponent Goals",    # leakage w.r.t. match outcome context
]

# Targets & labeling options
TARGET_REG = "XG"
CLASS_LABEL_MODE = "median"  # "median" or "q75"
Q_HIGH = 0.75                 # used when CLASS_LABEL_MODE == "q75"

# SHAP sampling (to keep plots readable when data is large)

MAX_SHAP_SAMPLES = 3000


# ----------------------------- Utilities -------------------------------------

def friendly_print(msg: str) -> None:
    line = "=" * 80
    print(f"{line}\n{msg}\n{line}")


def save_json(data: dict, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        json.dump(data, f, indent=4)


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def get_group_column(df: pd.DataFrame) -> str:
    for c in GROUP_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"None of the group columns {GROUP_CANDIDATES} found in dataset columns: {list(df.columns)}"
    )


def split_groups(df: pd.DataFrame, group_col: str, test_size: float, random_state: int):
    """Group-aware single split for final test; returns train_idx, test_idx."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col]
    (train_idx, test_idx) = next(gss.split(df, groups=groups))
    return train_idx, test_idx


def build_label_series(df: pd.DataFrame, group_col: str, target_col: str,
                       mode: str = CLASS_LABEL_MODE, q_high: float = Q_HIGH) -> pd.Series:
    """Return 0/1 danger label per row using per-group thresholding."""
    if mode not in {"median", "q75"}:
        raise ValueError("mode must be 'median' or 'q75'")
    thresholds = (
        df.groupby(group_col)[target_col].median()
        if mode == "median"
        else df.groupby(group_col)[target_col].quantile(q_high)
    )
    # Map threshold back to rows
    thr = df[group_col].map(thresholds)
    return (df[target_col] >= thr).astype(int)


def detect_feature_types(df: pd.DataFrame, exclude: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) after excluding given columns."""
    candidate_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in candidate_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    try:
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        cat = OneHotEncoder(handle_unknown="ignore", sparse=False)        # sklearn < 1.2
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", cat, categorical_cols),
        ]
    )
    return pre


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if trans == "passthrough":
            names.extend(list(cols))
        else:
            try:
                # OneHotEncoder provides get_feature_names_out
                ohe_names = trans.get_feature_names_out(cols)
                names.extend(list(ohe_names))
            except Exception:
                names.extend(list(cols))
    return names


def generate_required_shap_artifacts(
    model,
    preprocessor,
    X_source: pd.DataFrame,
    feature_names: list[str],
    out_dir: Path,
    random_state: int = RANDOM_STATE,
    max_samples: int | None = None,
    row_ids: Optional[pd.Series] = None,
    row_id_name: str = "Case number",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Also ensure plots directory exists for SHAP plots
    # For .png outputs, create corresponding plots dir
    mode_match = re.search(r"classification_(\w+)", str(out_dir))
    mode_tag = mode_match.group(1) if mode_match else "custom"
    plots_dir = PLOTS_DIR / f"classification_{mode_tag}" / out_dir.relative_to(out_dir.parents[1])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Transform features
    Xt = preprocessor.transform(X_source)

    # Align optional row IDs to X_source index
    if row_ids is not None:
        # Ensure it's aligned to X_source row order
        row_ids_aligned = row_ids.loc[X_source.index]
    else:
        row_ids_aligned = None

    # Optional downsampling
    limit = MAX_SHAP_SAMPLES if max_samples is None else max_samples
    rng = np.random.RandomState(random_state)
    if Xt.shape[0] > limit:
        idx = rng.choice(Xt.shape[0], limit, replace=False)
        Xt_plot = Xt[idx]
        if row_ids_aligned is not None:
            row_ids_plot = row_ids_aligned.iloc[idx].reset_index(drop=True)
        else:
            row_ids_plot = None
    else:
        Xt_plot = Xt
        row_ids_plot = row_ids_aligned.reset_index(drop=True) if row_ids_aligned is not None else None

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_ex = explainer(Xt_plot)
    sv = shap_ex.values if hasattr(shap_ex, 'values') else shap_ex

    # 1) Save raw SHAP values array
    np.save(out_dir / 'cls_shap_values.npy', sv)
    # Also save as CSV for readability (include case_id if provided)
    try:
        df_sv = pd.DataFrame(sv, columns=feature_names)
        if row_ids_plot is not None:
            df_sv.insert(0, row_id_name, row_ids_plot.values)
        df_sv.to_csv(out_dir / 'cls_shap_values.csv', index=False)
    except Exception as _e:
        with open(out_dir / 'WARN_shap_values_csv.txt', 'w') as f:
            f.write(str(_e))

    # 2) Save mean absolute SHAP per feature
    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_raw = np.mean(sv, axis=0)
    pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs,
        'mean_raw_shap': mean_raw
    }).sort_values('mean_abs_shap', ascending=False).to_csv(out_dir / 'cls_shap_meanabs.csv', index=False)

    # 3) SHAP summary plot
    shap.summary_plot(sv, pd.DataFrame(Xt_plot, columns=feature_names), show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / 'cls_shap_summary.png', dpi=300)
    plt.close()

    # 4) Interaction matrix (if supported)
    inter_dir = out_dir / 'interaction'
    inter_dir.mkdir(exist_ok=True)
    plots_inter_dir = plots_dir / 'interaction'
    plots_inter_dir.mkdir(exist_ok=True)
    try:
        if hasattr(explainer, 'shap_interaction_values'):
            inter_vals = explainer.shap_interaction_values(Xt_plot)
            M = np.mean(np.abs(inter_vals), axis=0)
            np.save(inter_dir / 'interaction_matrix.npy', M)
            pd.DataFrame(M, index=feature_names, columns=feature_names).to_csv(inter_dir / 'interaction_matrix.csv')
        else:
            # Fallback proxy using correlation of SHAP values
            corr = np.corrcoef(sv, rowvar=False)
            pd.DataFrame(corr, index=feature_names, columns=feature_names).to_csv(inter_dir / 'interaction_matrix_proxy.csv')
    except Exception as e:
        with open(inter_dir / 'ERROR.txt', 'w') as f:
            f.write(str(e))

    # 5) Only two specific dependence plots
    targeted_dir = out_dir / 'targeted'
    targeted_dir.mkdir(exist_ok=True)
    plots_targeted_dir = plots_dir / 'targeted'
    plots_targeted_dir.mkdir(exist_ok=True)
    feats_df = pd.DataFrame(Xt_plot, columns=feature_names)
    available = set(feature_names)
    pairs = [
        ("shoot_angle", "pressure_on_shooter"),
        ("shoot_angle", "num_opponents_front"),
    ]
    with open(targeted_dir / '_available_features.txt', 'w') as f:
        f.write("\n".join(sorted(available)))
    for main_f, inter_f in pairs:
        if main_f not in available or inter_f not in available:
            with open(targeted_dir / f"WARN_missing_{main_f}__{inter_f}.txt", 'w') as f:
                f.write(f"Missing: main={main_f in available}, interaction={inter_f in available}\n")
            continue
        try:
            plt.figure(figsize=(6, 5))
            shap.dependence_plot(
                ind=main_f,
                shap_values=sv,
                features=feats_df,
                interaction_index=inter_f,
                show=False,
            )
            plt.title(f"Dependence: {main_f} by {inter_f}")
            plt.tight_layout()
            plt.savefig(plots_targeted_dir / f"dep_{main_f}__by__{inter_f}.png", dpi=300)
            plt.close()
        except Exception as e:
            with open(targeted_dir / f"ERROR_{main_f}__{inter_f}.txt", 'w') as f:
                f.write(str(e))


# ---------------------- Zone SHAP heatmap utilities -------------------------

def _color_from_value(v: float) -> tuple:
    """
    Map v in [-1, 1] to a blue-white-red scale:
    -1 -> blue, 0 -> white, +1 -> red.
    """
    v = float(np.clip(v, -1.0, 1.0))
    if v >= 0:
        # 0..+1: white -> red
        r = 1.0
        g = 1.0 - v
        b = 1.0 - v
    else:
        # -1..0: blue -> white
        r = 1.0 + v
        g = 1.0 + v
        b = 1.0
    return (r, g, b)

def plot_zone_shap_heatmap(zone_values: dict[int, float], save_path: Path) -> None:
    """
    Draw a simple opponent-half (zones 13..24) heatmap on a 120x80 pitch using matplotlib only.
    zone_values: mapping {zone -> normalized value in [-1,1]}.
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect('equal')
    ax.axis('off')

    # Pitch outline (simple)
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color="black", linewidth=2)
    # Halfway line
    ax.plot([60, 60], [0, 80], color="black", linewidth=2)
    # Penalty box (attacking right)
    ax.add_patch(patches.Rectangle((102, 18), 18, 44, fill=False, edgecolor="black", linewidth=2))
    # 6-yard box
    ax.add_patch(patches.Rectangle((114, 30), 6, 20, fill=False, edgecolor="black", linewidth=2))
    # Goal
    ax.add_patch(patches.Rectangle((120, 36), 2, 8, fill=False, edgecolor="black", linewidth=2))
    # Centre circle
    circ = patches.Circle((60, 40), radius=10, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(circ)

    # Zone grid (6 columns x 4 rows) for opponent half = x in [60,120]
    # Numbering pattern (bottom->top rows):
    grid = [
        [13, 17, 21],  # bottom row, x bins: [60-80], [80-100], [100-120]
        [14, 18, 22],
        [15, 19, 23],
        [16, 20, 24],
    ]
    # Each cell size
    cell_w = 20
    cell_h = 20
    # Bottom row y=0..20, top row y=60..80; x from 60..120
    for r in range(4):
        for c in range(3):
            z_left = 60 + c * cell_w
            z_bottom = r * cell_h
            zone_id = grid[r][c]
            val = float(zone_values.get(zone_id, 0.0))
            col = _color_from_value(val)
            rect = patches.Rectangle((z_left, z_bottom), cell_w, cell_h, linewidth=1.0, edgecolor="black", facecolor=col, alpha=0.9)
            ax.add_patch(rect)
            ax.text(z_left + cell_w/2, z_bottom + cell_h/2, f"{zone_id}", ha="center", va="center", fontsize=10, color="black")

    ax.set_title("SHAP Zone Impact Heatmap (Opponent Half)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # If saving a .png, ensure plots dir exists (handled by caller)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def compute_zone_shap_map(
    df_all: pd.DataFrame,
    shap_values_csv: Path,
    feature_names: list[str],
    zone_col_name_candidates: tuple = ("End_Zone", "End Zone", "EndZone"),
) -> dict[int, float]:
    """
    Build {zone -> mean raw SHAP} map.
    Priority 1: read mean_raw_shap from extended_all/cls_shap_meanabs.csv and
                parse one-hot End_Zone features into numeric zone ids.
    Priority 2: if not found, fall back to per-row SHAP values (cls_shap_values.csv).
    Returns normalized values in [-1, 1].
    """
    zone_map: dict[int, float] = {}
    # ---------- Try Priority 1: mean_raw_shap table ----------
    try:
        meanabs_path = shap_values_csv.parent / "cls_shap_meanabs.csv"
        if meanabs_path.exists():
            df_mean = pd.read_csv(meanabs_path)
            if {"feature", "mean_raw_shap"}.issubset(df_mean.columns):
                # Accept several OHE name patterns, pull trailing integer even if written "13.0"
                pat = re.compile(r"(?:^|[_\s])(?:End[_\s]?Zone)_(\d+(?:\.0)?)$|(?:^|[_\s])cat__End[_\s]?Zone_(\d+(?:\.0)?)$")
                for _, row in df_mean.iterrows():
                    feat = str(row["feature"])
                    m = pat.search(feat)
                    if m:
                        num_str = m.group(1) or m.group(2)
                        try:
                            z = int(float(num_str))
                            zone_map[z] = float(row["mean_raw_shap"])
                        except Exception:
                            continue
    except Exception:
        pass
    # ---------- If Priority 1 failed, try Priority 2: per-row shap csv ----------
    if not zone_map:
        try:
            shap_df = pd.read_csv(shap_values_csv)
            # Gather OHE columns for zones
            ohe_cols = []
            for col in shap_df.columns:
                m = re.search(r"(?:^|[_\s])(?:End[_\s]?Zone)_(\d+(?:\.0)?)$|(?:^|[_\s])cat__End[_\s]?Zone_(\d+(?:\.0)?)$", col)
                if m:
                    try:
                        z = int(float(m.group(1) or m.group(2)))
                        ohe_cols.append((z, col))
                    except Exception:
                        pass
            if ohe_cols:
                for z, col in ohe_cols:
                    zone_map[z] = shap_df[col].mean()
            else:
                # Fallback to numeric End_Zone column if present in both dfs
                shap_numeric = None
                for cand in zone_col_name_candidates:
                    if cand in shap_df.columns and cand in df_all.columns:
                        shap_numeric = cand
                        break
                if shap_numeric:
                    tmp = pd.DataFrame({"zone": df_all[shap_numeric].astype(int), "sv": shap_df[shap_numeric]})
                    zone_map = tmp.groupby("zone")["sv"].mean().to_dict()
        except Exception:
            pass
    # ---------- Normalize & restrict to opponent-half zones ----------
    if not zone_map:
        return {}
    # Keep only zones 13..24 if present
    zone_map = {z: v for z, v in zone_map.items() if 13 <= int(z) <= 24}
    if not zone_map:
        return {}
    max_abs = max(abs(v) for v in zone_map.values()) or 1.0
    for k in list(zone_map.keys()):
        zone_map[k] = float(np.clip(zone_map[k] / max_abs, -1.0, 1.0))
    return zone_map

# ----------------------------- Main pipeline ---------------------------------

def main() -> None:
    # 1) Load data
    friendly_print("Loading data…")
    df = pd.read_csv(DATA_PATH)

    # basic cleaning of target
    if TARGET_REG not in df.columns:
        raise ValueError(f"Target column '{TARGET_REG}' not found in dataset.")

    # Drop rows with missing/invalid target
    df = df[np.isfinite(df[TARGET_REG])].copy()

    # 2) Determine group column
    group_col = get_group_column(df)
    friendly_print(f"Using group column for leakage-safe splits: {group_col}")

    # 3) Feature type detection
    friendly_print("Detecting feature types…")
    num_cols, cat_cols = detect_feature_types(df, exclude=EXCLUDE_COLS)
    if len(num_cols) + len(cat_cols) == 0:
        raise ValueError("No features left after exclusions. Check EXCLUDE_COLS.")

    # 4) Group-aware train/test split
    friendly_print("Creating group-aware train/test split…")
    train_idx, test_idx = split_groups(df, group_col, TEST_SIZE, RANDOM_STATE)
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    # 5) Build preprocessing and feature matrices
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train = train_df[num_cols + cat_cols]
    X_test = test_df[num_cols + cat_cols]
    # Keep regression target only for labeling, no regression model used

    # Keep groups for CV
    train_groups = train_df[group_col]

    # ------------------------ Classification (XGBoost) ------------------------
    friendly_print("Building danger labels and training XGBClassifier…")
    # Build labels per-group
    y_train_cls = build_label_series(train_df, group_col, TARGET_REG, mode=CLASS_LABEL_MODE, q_high=Q_HIGH)
    y_test_cls = build_label_series(test_df, group_col, TARGET_REG, mode=CLASS_LABEL_MODE, q_high=Q_HIGH)

    xgb_cls = XGBClassifier(
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_estimators=300,
        n_jobs=N_JOBS,
        # scale_pos_weight set inside search via lambda
    )

    pipe_cls = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_cls)])

    # Handle class imbalance via scale_pos_weight = neg/pos in train fold (approx via global rate)
    pos_rate = max(y_train_cls.mean(), 1e-6)
    spw = (1 - pos_rate) / pos_rate

    param_distributions_cls = {
        "model__n_estimators": np.arange(200, 901, 50),
        "model__learning_rate": np.linspace(0.02, 0.25, 24),
        "model__max_depth": np.arange(3, 11),
        "model__subsample": np.linspace(0.6, 1.0, 9),
        "model__colsample_bytree": np.linspace(0.6, 1.0, 9),
        "model__min_child_weight": np.arange(1, 11),
        "model__gamma": np.linspace(0, 1.0, 11),
        "model__reg_alpha": np.linspace(0, 1.0, 11),
        "model__reg_lambda": np.linspace(0.5, 2.0, 16),
        "model__scale_pos_weight": [spw, max(spw * 0.5, 0.1), spw * 2, 1.0],
    }

    cv_cls = GroupKFold(n_splits=5)

    search_cls = RandomizedSearchCV(
        pipe_cls,
        param_distributions=param_distributions_cls,
        n_iter=N_ITER_SEARCH,
        cv=cv_cls,
        scoring="roc_auc",
        verbose=1,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )

    search_cls.fit(X_train, y_train_cls, groups=train_groups)

    best_cls = search_cls.best_estimator_
    best_cls_params = convert_to_serializable(search_cls.best_params_)

    # Evaluate classification on held-out test
    friendly_print("Evaluating classifier on test set…")
    y_prob = best_cls.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cls_metrics = {
        "Accuracy": accuracy_score(y_test_cls, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_test_cls, y_pred),
        "ROC_AUC": roc_auc_score(y_test_cls, y_prob),
        "PR_AUC": average_precision_score(y_test_cls, y_prob),
        "BrierScore": brier_score_loss(y_test_cls, y_prob),
        "BestParameters": best_cls_params,
        "ScalePosWeight_used": best_cls.named_steps["model"].get_xgb_params().get("scale_pos_weight", None),
    }

    mode_tag = CLASS_LABEL_MODE if CLASS_LABEL_MODE in {"median", "q75"} else "custom"
    out_cls = CSV_DIR / f"classification_{mode_tag}"
    out_cls.mkdir(parents=True, exist_ok=True)
    out_metrics = METRICS_DIR / f"classification_{mode_tag}"
    out_metrics.mkdir(parents=True, exist_ok=True)
    # Save model, predictions
    joblib.dump(best_cls, out_cls / "xgb_cls.joblib")
    pd.DataFrame({
        "y_true": y_test_cls,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }).to_csv(out_cls / "test_predictions.csv", index=False)
    # Save metrics.json in metrics directory
    save_json(cls_metrics, out_metrics / "metrics.json")

    # ---------------- Required SHAP artifacts on ALL data ----------------
    X_all = df[num_cols + cat_cols]
    f_names_all = transformed_feature_names(best_cls.named_steps["preprocessor"])  # ensure order
    ext_all_dir = out_cls / "explainability" / "extended_all"
    generate_required_shap_artifacts(
        model=best_cls.named_steps["model"],
        preprocessor=best_cls.named_steps["preprocessor"],
        X_source=X_all,
        feature_names=f_names_all,
        out_dir=ext_all_dir,
        random_state=RANDOM_STATE,
        max_samples=X_all.shape[0],
        row_ids=df["Case number"],
        row_id_name="Case number",
    )


    # --------------- Friendly terminal summary ---------------
    friendly_print("Training & evaluation complete!")
    print("\n[Classification] Best hyperparameters:")
    for k, v in best_cls_params.items():
        print(f"  {k}: {v}")
    print("[Classification] Metrics:")
    for k, v in cls_metrics.items():
        if k not in {"BestParameters", "ScalePosWeight_used"}:
            print(f"  {k}: {v:.4f}")
    print(f"  ScalePosWeight_used: {cls_metrics['ScalePosWeight_used']}")

    print("\nArtifacts saved to:")
    print(f"  Classification dir : {out_cls}")



# --- main() wrapper for external run ---
def run():
    """Entry point wrapper for main.py compatibility."""
    # silence shap deprecation chatter on some versions
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

if __name__ == "__main__":
    run()