from __future__ import annotations

import json
from pathlib import Path
import warnings
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# Targets & labeling options
CLASS_LABEL_MODE = "median"  # "median" or "q75"
Q_HIGH = 0.75
# ----------------------------- Configuration ---------------------------------
DATA_PATH = Path(
    "outputs/CSV-JSON/BaseFeatures.csv"
)
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.2  # proportion of **groups** held out for final test
N_JOBS = -1
N_ITER_SEARCH = 50

# --------- Output directories per mode and artifact type ---------
MODE_TAG = f"classification_{CLASS_LABEL_MODE}"
DIR_METRICS = OUTPUT_DIR / "metrics" / MODE_TAG
DIR_TABLES  = OUTPUT_DIR / "CSV-JSON" / MODE_TAG
DIR_PLOTS   = OUTPUT_DIR / "plots" / MODE_TAG
for d in [DIR_METRICS, DIR_TABLES, DIR_PLOTS]:
    d.mkdir(parents=True, exist_ok=True)

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

# Text outputs directory
TEXT_DIR = OUTPUT_DIR / "text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

def append_to_report(text: str, file_path: Path) -> None:
    """Append text to a UTF-8 report file, creating it if missing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(text)


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
    import sklearn
    from packaging import version

    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat = OneHotEncoder(handle_unknown="ignore", sparse=False)
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
    out_dir_tables: Path,
    out_dir_plots: Path,
    random_state: int = RANDOM_STATE,
    max_samples: int | None = None,
    row_ids: Optional[pd.Series] = None,
    row_id_name: str = "Case number",
) -> None:
    out_dir_tables = Path(out_dir_tables)
    out_dir_tables.mkdir(parents=True, exist_ok=True)
    out_dir_plots = Path(out_dir_plots)
    out_dir_plots.mkdir(parents=True, exist_ok=True)

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

    # 1) Save raw SHAP values array (tables)
    np.save(out_dir_tables / 'cls_shap_values.npy', sv)
    # Also save as CSV for readability (include case_id if provided)
    try:
        df_sv = pd.DataFrame(sv, columns=feature_names)
        if row_ids_plot is not None:
            df_sv.insert(0, row_id_name, row_ids_plot.values)
        df_sv.to_csv(out_dir_tables / 'cls_shap_values.csv', index=False)
    except Exception as _e:
        with open(out_dir_tables / 'WARN_shap_values_csv.txt', 'w') as f:
            f.write(str(_e))

    # 2) Save mean absolute SHAP per feature (tables)
    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_raw = np.mean(sv, axis=0)
    pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs,
        'mean_raw_shap': mean_raw
    }).sort_values('mean_abs_shap', ascending=False).to_csv(out_dir_tables / 'cls_shap_meanabs.csv', index=False)

    # 3) SHAP summary plot (plots)
    shap.summary_plot(sv, pd.DataFrame(Xt_plot, columns=feature_names), show=False)
    plt.tight_layout()
    plt.savefig(out_dir_plots / 'cls_shap_summary.png', dpi=300)
    plt.close()

    # 4) Interaction matrix (tables)
    inter_dir = out_dir_tables / 'interaction'
    inter_dir.mkdir(exist_ok=True)
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

    # 5) Only two specific dependence plots (plots)
    targeted_dir = out_dir_plots / 'targeted'
    targeted_dir.mkdir(exist_ok=True)
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
            plt.savefig(targeted_dir / f"dep_{main_f}__by__{inter_f}.png", dpi=300)
            plt.close()
        except Exception as e:
            with open(targeted_dir / f"ERROR_{main_f}__{inter_f}.txt", 'w') as f:
                f.write(str(e))

#
# ---------------- Gemini caller for SHAP dependence plot insights ----------------
def call_gemini_for_dependence_plots(prompt_text: str, image_paths: list[str]) -> str:
    """Build a prompt with one or more dependence-plot images and call Gemini."""
    from google import genai

    # Use env var if present; otherwise fallback to provided key for parity with VisualAnalysis.py
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyD3Y6wabXe5rMidJTpmk6785rSI4K2gjvk")

    # Compose prompt with image references
    img_block = "\n".join(f"[IMAGE: {p}]" for p in image_paths)
    full_prompt = f"{prompt_text}\n\n--- Charts ---\n{img_block}\n"

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=full_prompt,
    )
    text = getattr(resp, "text", None)
    if text is None:
        print("[GEMINI][WARN] Empty response.text in call_gemini_for_dependence_plots; returning empty string.")
        text = ""
    return text

DEPENDENCE_PROMPT = (
    "Role & Goal:\n"
    "You are a football performance analyst whose job is to turn complex data visualizations into short, coach-friendly tactical insights. The target audience is a football coach with no data science background.\n\n"
    "Instructions:\n"
    "\t1.\tYou will be given one or more SHAP dependence plots in PNG format.\n"
    "\t2.\tFor each chart, do the following:\n"
    "## Chart [Feature name]\n"
    "**Simple Insight:**  \n"
    "[2–3 sentence summary]\n\n"
    "**In-Game Decision Checklist:**  \n"
    "- [Bullet 1]  \n"
    "- [Bullet 2]  \n"
    "- [Bullet 3]  \n\n"
    "**Training Recommendations:**  \n"
    "- [Drill 1]  \n"
    "- [Drill 2]  \n"
    "**KPIs:**  \n"
    "- [KPI 1]  \n"
    "- [KPI 2]  \n"
    "- [KPI 3]  \n"
    "\t•\tStep 1 – Simple Insight (2–3 sentences): Summarise what the chart says in plain football language. Avoid any data science jargon (no “SHAP”, “correlation”, or “feature importance” terms).\n"
    "\t•\tStep 2 – In-Game Decision Checklist: Write a 3–4 bullet-point list of clear in-match decisions a coach can tell players, based on what the chart shows. Each bullet must be short and actionable (“If angle < 20° → pass instead of shooting”).\n"
    "\t•\tStep 3 – Training Recommendations: Suggest 2–3 specific drills or KPIs that can be used in training to improve team performance related to the insight. Include 2–3 bullet-point KPIs (measurable targets) that a coach can track.\n\n"
    "Tone & Style:\n"
    "\t•\tUse short sentences, active voice.\n"
    "\t•\tAvoid technical words.\n"
    "\t•\tSpeak as if explaining to a player in a locker room — clear, direct, no fluff.\n\n"
    "Example:\n"
    "Input: dep_shoot_angle__by__pressure_on_shooter.png\n"
    "Output:\n"
    "## Chart shoot_angle by pressure_on_shooter\n"
    "**Simple Insight:**  \n"
    "When the shooting angle is over 20°, the chance of scoring increases. But if the shooter is under heavy pressure, even a good angle loses its advantage. Reducing pressure before the shot makes a big difference.\n\n"
    "**In-Game Decision Checklist:**  \n"
    "- If angle &lt; 20° → pass or move to open the angle.  \n"
    "- If angle &gt; 20° and pressure is low → shoot.  \n"
    "- If angle &gt; 20° and pressure is high → fake or play a quick pass to reduce pressure, then shoot.\n\n"
    "**Training Recommendations:**  \n"
    "- Practice fakes and quick releases under pressure.  \n"
    "- Train switching the ball quickly to reduce defender pressure.  \n"
    "**KPIs:**  \n"
    "- % of shots with angle &gt; 20° and low pressure.  \n"
    "- Average pressure on shooter at the moment of shot (trend: decreasing).  \n"
    "- Goal/shot rate from reduced-pressure situations.\n"
)

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

    # Save model, predictions, metrics
    joblib.dump(best_cls, DIR_METRICS / "xgb_cls.joblib")
    pd.DataFrame({
        "y_true": y_test_cls,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }).to_csv(DIR_TABLES / "test_predictions.csv", index=False)
    save_json(cls_metrics, DIR_METRICS / "metrics.json")

    # ---------------- Required SHAP artifacts on ALL data ----------------
    X_all = df[num_cols + cat_cols]
    f_names_all = transformed_feature_names(best_cls.named_steps["preprocessor"])  # ensure order
    ext_all_tables = DIR_TABLES / "explainability" / "extended_all"
    ext_all_plots  = DIR_PLOTS  / "explainability" / "extended_all"
    generate_required_shap_artifacts(
        model=best_cls.named_steps["model"],
        preprocessor=best_cls.named_steps["preprocessor"],
        X_source=X_all,
        feature_names=f_names_all,
        out_dir_tables=ext_all_tables,
        out_dir_plots=ext_all_plots,
        random_state=RANDOM_STATE,
        max_samples=X_all.shape[0],
        row_ids=df["Case number"],
        row_id_name="Case number",
    )

    # --- AI-generated insights for SHAP dependence plots (classification) ---
    targeted_dir = DIR_PLOTS / "explainability" / "extended_all" / "targeted"
    img_paths = [
        str(targeted_dir / "dep_shoot_angle__by__num_opponents_front.png"),
        str(targeted_dir / "dep_shoot_angle__by__pressure_on_shooter.png"),
    ]
    try:
        ai_text = call_gemini_for_dependence_plots(DEPENDENCE_PROMPT, img_paths)
        final_report_path = TEXT_DIR / "FinalReport.txt"
        section_header = "\n## Section – Dependence Plot Insights (Classification)\n\n"
        append_to_report(section_header + (ai_text or "") + "\n", final_report_path)
        print(f"[INFO] Gemini dependence-plot insights appended to {final_report_path}")
    except Exception as e:
        print("[WARN] Gemini dependence-plot analysis skipped:", e)

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
    print(f"  Metrics : {DIR_METRICS}")
    print(f"  Tables  : {DIR_TABLES}")
    print(f"  Plots   : {DIR_PLOTS}")



# Public entry point for orchestration

def run() -> None:
    """Entry point wrapper so other modules (e.g., main.py) can call Classification."""
    # keep runtime behavior consistent with direct execution
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

if __name__ == "__main__":
    # silence shap deprecation chatter on some versions
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()