from pathlib import Path
import json
import pandas as pd
from typing import List, Tuple, Dict, DefaultDict

from collections import defaultdict
import importlib.util
from typing import Callable, Optional, Dict, DefaultDict

# ======= CONFIG =======
# Project root (parent of scripts folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Root folder that contains all match folders under data/raw/Leverkusen_Matches
BASE_DIR = PROJECT_ROOT / "data" / "raw" / "Leverkusen_Matches"
# You can lock by team_id instead of name as well (StatsBomb example: 904 for Bayer Leverkusen in your data).
TARGET_TEAM_NAME = "Bayer Leverkusen"
TARGET_TEAM_ID = 904   # If you're unsure, set to None to filter by team name only.

# --- Unified output roots by artifact type (create subfolder PlayerZoneAnalysis under each) ---
ROOT_OUTPUT = PROJECT_ROOT / "outputs"
OUT_CSVJSON_DIR = ROOT_OUTPUT / "CSV-JSON" / "PlayerZoneAnalysis"
OUT_PLOTS_DIR   = ROOT_OUTPUT / "plots" / "PlayerZoneAnalysis"
OUT_METRICS_DIR = ROOT_OUTPUT / "metrics" / "PlayerZoneAnalysis"

# Output CSV path (players list)
OUTPUT_CSV = OUT_CSVJSON_DIR / "leverkusen_season_players.csv"

# === NEW: paths used for case/zone counting ===
# Path to the BaseFeatures CSV produced by CaseDetection.py
BASE_FEATURES_CSV = PROJECT_ROOT / "outputs" / "CSV-JSON" / "BaseFeatures.csv"
# Where to save outputs for player x zone counts
OUTPUT_PLAYER_ZONE_LONG = OUT_CSVJSON_DIR / "player_zone_counts_long.csv"
OUTPUT_PLAYER_ZONE_WIDE = OUT_CSVJSON_DIR / "player_zone_counts_wide.csv"
# Outputs for finishing shots in zones 22 & 23
OUTPUT_FINISHING_SHOTS_LONG = OUT_CSVJSON_DIR / "finishing_shots_by_player_long.csv"
OUTPUT_FINISHING_SHOTS_SUMMARY = OUT_CSVJSON_DIR / "finishing_shots_by_player_summary.csv"
# SHAP mean-abs file (season-level) for wing analysis (Module 1)
SHAP_MEANABS_CSV = PROJECT_ROOT / "outputs" / "CSV-JSON" / "classification_median" / "explainability" / "extended_all" / "cls_shap_meanabs.csv"
# Output path for Module 1 JSON
OUTPUT_WING_JSON = OUT_CSVJSON_DIR / "wing_analysis.json"
# Outputs for Module 2 (finishing zones JSON) and case-level shooter mapping
OUTPUT_CASE_FINISHING_SHOTS = OUT_CSVJSON_DIR / "case_finishing_shots.csv"
OUTPUT_FINISHING_JSON = OUT_CSVJSON_DIR / "finishing_analysis.json"
# Candidate paths to CaseDetection.py (we try these in order); adjust if needed
CASE_DETECTION_CANDIDATES = [
    PROJECT_ROOT / "scripts" / "CaseDetection.py"
]
# Whether to also count the end_location of carries as a second touch in that zone
COUNT_CARRY_END = False
# ======================

def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None

def is_target_team(team_obj: dict) -> bool:
    # Flexible: match by name and, if provided, by team ID
    name_ok = (str(team_obj.get("team_name", "")).strip().lower() 
               == TARGET_TEAM_NAME.strip().lower())
    id_ok = (TARGET_TEAM_ID is None) or (int(team_obj.get("team_id", -1)) == TARGET_TEAM_ID)
    # If ID is set, require both; if ID is None, use name only
    return (name_ok and id_ok) if TARGET_TEAM_ID is not None else name_ok

# ---------- Utilities to load CaseDetection zone function ----------

def _load_case_detection_module() -> Optional[object]:
    """Attempt to import the user's CaseDetection.py dynamically from known locations.
    Returns the loaded module object if found; otherwise None.
    """
    for p in CASE_DETECTION_CANDIDATES:
        try:
            if p.exists():
                spec = importlib.util.spec_from_file_location("CaseDetection", str(p))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    print(f"[INFO] Loaded CaseDetection from: {p}")
                    return mod
        except Exception as e:
            print(f"[WARN] Could not import CaseDetection at {p}: {e}")
    print("[WARN] CaseDetection.py not found in candidates; falling back to default zone mapping.")
    return None


def _zone_from_xy_fallback(x: float, y: float) -> int:
    """Fallback 6x4 (120x80) grid mapping that matches Soroush's matrix:
    Columns left→right (x):
      col1: [1,2,3,4], col2: [5,6,7,8], col3: [9,10,11,12],
      col4: [13,14,15,16], col5: [17,18,19,20], col6: [21,22,23,24]
    Rows bottom→top (y):
      row0: y∈[0,20) → 1..6 per column base
      row1: y∈[20,40)
      row2: y∈[40,60)
      row3: y∈[60,80]

    This yields the top row (y=60→80): [4, 8, 12, 16, 20, 24]
    and the bottom row (y=0→20): [1, 5, 9, 13, 17, 21].
    """
    # Clamp to field
    if x < 0: x = 0.0
    if y < 0: y = 0.0
    if x > 120: x = 120.0
    if y > 80:  y = 80.0
    # Compute column (0..5) and row (0..3)
    col = int(min(5, x // 20))  # 6 columns across 120
    row = int(min(3, y // 20))  # 4 rows across 80, counted bottom→top
    # Column-wise numbering: base index per column is col*4, then add row offset
    return col * 4 + row + 1

# --- Validator to ensure external zone function matches the expected matrix ---
_EXPECTED_MATRIX = [
    [4, 8, 12, 16, 20, 24],  # top row (y=60–80)
    [3, 7, 11, 15, 19, 23],  # y=40–60
    [2, 6, 10, 14, 18, 22],  # y=20–40
    [1, 5, 9, 13, 17, 21],   # bottom row (y=0–20)
]

def _zone_fn_matches_expected(zone_fn) -> bool:
    """Sample the center of each cell and compare to _EXPECTED_MATRIX.
    Returns True when numbering is identical to the requested scheme.
    """
    # centers for each cell
    ys = [70, 50, 30, 10]  # top→bottom rows
    xs = [10, 30, 50, 70, 90, 110]  # left→right columns
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            expected = _EXPECTED_MATRIX[r][c]
            got = int(zone_fn(x, y))
            if got != expected:
                return False
    return True


def get_zone_function() -> Callable[[float, float], int]:
    """Prefer the exact zone function from CaseDetection.py to avoid mismatch.
    Supported names (first match wins): xy_to_zone, point_to_zone, zone_index_from_xy
    Fallback: internal 6x4 mapping above.
    """
    mod = _load_case_detection_module()
    if mod is not None:
        for fname in ("xy_to_zone", "point_to_zone", "zone_index_from_xy"):
            f = getattr(mod, fname, None)
            if callable(f):
                # Validate numbering against the expected matrix
                try:
                    if _zone_fn_matches_expected(f):
                        print(f"[INFO] Using zone function from CaseDetection.py: {fname}() (validated)")
                        return f  # type: ignore[return-value]
                    else:
                        print(f"[WARN] CaseDetection.{fname}() numbering does not match expected matrix; using internal fallback.")
                except Exception as e:
                    print(f"[WARN] Could not validate CaseDetection.{fname}(): {e}; using internal fallback.")
        print("[WARN] No valid zone function found in CaseDetection.py; using fallback mapping.")
    return _zone_from_xy_fallback

# ---------- Safe parsing helpers ----------

def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None

# ---------- Resolve event.json path for each BaseFeatures row ----------

def _resolve_events_path(row: pd.Series, matches_root: Path) -> Optional[Path]:
    """Best-effort resolution of the event.json path for a given row.
    Priority:
      1) Any column that already contains a path ending with 'event.json' or 'events.json'.
      2) A folder-ish column (e.g., match_folder, match_dir, folder, path) joined with 'event.json'.
      3) Last resort: None (caller must handle).
    """
    # 0) Strong hint: many pipelines store the match folder name in 'Match Name'
    #    e.g., 'LeverkusenVSAugsburg_Away' → BASE_DIR / that / 'event.json'
    match_name_cols = ["Match Name", "match_name", "match", "match_code"]
    for key in match_name_cols:
        if key in row and isinstance(row[key], str) and row[key].strip():
            candidate = matches_root / row[key].strip() / "event.json"
            if candidate.exists():
                return candidate
            # also try events.json just in case
            candidate2 = matches_root / row[key].strip() / "events.json"
            if candidate2.exists():
                return candidate2

    # 1) Existing absolute/relative path in any string-like column
    for col in row.index:
        val = row[col]
        if isinstance(val, str):
            s = val.strip()
            low = s.lower()
            if low.endswith("event.json") or low.endswith("events.json"):
                p = Path(s)
                return p if p.is_absolute() else (matches_root / p)
    # 2) Folder-style columns
    for key in ("match_folder", "match_dir", "folder", "path", "relative_dir"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return matches_root / row[key].strip() / "event.json"

    print(f"[DBG] Could not resolve path from row keys; tried match_name and folder-like columns. Row keys: {list(row.index)}")
    return None

# ---------- Event iteration within [start_index, end_index] ----------

def _iter_events_between(events: list, start_idx: int, end_idx: int):
    """Yield events whose 'index' field is within [start_idx, end_idx] inclusive.
    Works even if the list order differs from the 'index' numeric order.
    """
    lo, hi = (start_idx, end_idx) if start_idx <= end_idx else (end_idx, start_idx)
    for ev in events:
        idx = ev.get("index")
        if isinstance(idx, int) and lo <= idx <= hi:
            yield ev

def extract_players_from_lineup(lineup_json) -> List[Tuple[int, str]]:
    """
    Input: lineup.json structure (a list of two team objects)
    Output: list of (player_id, player_name) tuples for the target team only
    """
    players = []
    if not isinstance(lineup_json, list):
        return players
    for team in lineup_json:
        if is_target_team(team):
            for p in team.get("lineup", []):
                pid = p.get("player_id")
                pname = p.get("player_name")
                if pid is not None and pname:
                    players.append((int(pid), str(pname)))
    return players

# ---------- Core aggregation: count player touches per zone during detected attacks ----------

def build_player_zone_counts(
    base_features_csv: Path,
    matches_root: Path,
    target_team_id: int = TARGET_TEAM_ID,
    count_carry_end: bool = COUNT_CARRY_END,
) -> pd.DataFrame:
    """Create a long-form DataFrame with columns: player_id, player_name, zone, count.

    Contractual correctness:
      • Match alignment: we never guess an order. For each case-row we resolve the exact
        event.json path from columns in BaseFeatures. This guarantees start/end indices
        refer to the correct match file that CaseDetection.py used.
      • Zone indices: we import the zone function from CaseDetection.py (if available)
        to ensure the exact same 6x4 mapping and numbering.

    Robustness:
      • If BaseFeatures contains cases for both teams, we filter events by team.id==target_team_id.
      • If an events path cannot be resolved for a row, we skip that row but log a warning.
    """
    # 0) Load BaseFeatures and sanity-log its shape & columns
    bf = pd.read_csv(base_features_csv)
    print(f"[INFO] BaseFeatures loaded: {bf.shape[0]} rows, {bf.shape[1]} cols")
    print("[INFO] BaseFeatures columns:", list(bf.columns))
    print("[INFO] Sample rows:\n", bf.head(3))

    # 1) Resolve zone function
    zone_fn = get_zone_function()

    # 2) Cache events per match path to avoid re-loading
    events_cache: Dict[Path, List[dict]] = {}

    # 3) Counts: {player_id: {zone: count}}
    counts: Dict[int, DefaultDict[int, int]] = {}
    player_names: Dict[int, str] = {}

    # Try to detect an explicit column that groups cases by match (so we can reuse paths)
    # Not required, but helps to log.
    match_hint_col = None
    for k in ("match_id", "match_folder", "match_dir", "match_date", "events_path", "event_path"):
        if k in bf.columns:
            match_hint_col = k
            break

    for i, row in bf.iterrows():
        s_idx = _safe_int(row.get("start_index"))
        e_idx = _safe_int(row.get("end_index"))
        if s_idx is None or e_idx is None:
            print(f"[WARN] Row {i}: missing start/end_index; skipping.")
            continue

        ev_path = _resolve_events_path(row, matches_root)
        if ev_path is None:
            print(f"[WARN] Row {i}: could not resolve event.json path; skipping.")
            continue
        if ev_path not in events_cache:
            try:
                with open(ev_path, "r", encoding="utf-8") as f:
                    events_cache[ev_path] = json.load(f)
                print(f"[INFO] Loaded events: {ev_path}")
            except Exception as e:
                print(f"[WARN] Cannot load events {ev_path}: {e}")
                events_cache[ev_path] = []

        for ev in _iter_events_between(events_cache[ev_path], s_idx, e_idx):
            # Filter to target team if present
            team = ev.get("team") or {}
            if target_team_id is not None and int(team.get("id", -1)) != int(target_team_id):
                continue
            # We only count when a player and a location exist
            player = ev.get("player") or {}
            loc = ev.get("location")
            if player and isinstance(loc, (list, tuple)) and len(loc) >= 2:
                pid = _safe_int(player.get("id"))
                pname = str(player.get("name")) if player.get("name") else None
                if pid is None:
                    continue
                x, y = float(loc[0]), float(loc[1])
                z = int(zone_fn(x, y))
                counts.setdefault(pid, defaultdict(int))[z] += 1
                if pname:
                    player_names.setdefault(pid, pname)
            # Optionally count carry end-locations to reflect movement
            if count_carry_end and "carry" in ev and isinstance(ev["carry"], dict):
                end_loc = ev["carry"].get("end_location")
                player = ev.get("player") or {}
                if end_loc and isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2 and player:
                    pid = _safe_int(player.get("id"))
                    pname = str(player.get("name")) if player.get("name") else None
                    if pid is not None:
                        x2, y2 = float(end_loc[0]), float(end_loc[1])
                        z2 = int(zone_fn(x2, y2))
                        counts.setdefault(pid, defaultdict(int))[z2] += 1
                        if pname:
                            player_names.setdefault(pid, pname)

    # 4) Build long-form DataFrame
    records = []
    for pid, zone_map in counts.items():
        for z, c in sorted(zone_map.items()):
            records.append({
                "player_id": pid,
                "player_name": player_names.get(pid, ""),
                "zone": z,
                "count": int(c),
            })

    # Guard: no records collected → return empty, typed frames to avoid KeyError on sort
    if not records:
        df_long = pd.DataFrame(columns=["player_id", "player_name", "zone", "count"])
        df_wide = pd.DataFrame(columns=["player_id", "player_name"] + [f"zone_{i}" for i in range(1, 25)])
        return df_long, df_wide

    df_long = pd.DataFrame(records).sort_values(["player_name", "player_id", "zone"]).reset_index(drop=True)

    # 5) Also create a wide pivot (one row per player, columns zone_1..zone_24)
    if not df_long.empty:
        df_wide = df_long.pivot_table(index=["player_id", "player_name"], columns="zone", values="count", aggfunc="sum", fill_value=0)
        # Flatten columns to zone_X names
        df_wide.columns = [f"zone_{int(c)}" for c in df_wide.columns]
        df_wide = df_wide.reset_index().sort_values(["player_name", "player_id"]).reset_index(drop=True)
    else:
        df_wide = pd.DataFrame(columns=["player_id", "player_name"] + [f"zone_{i}" for i in range(1, 25)])

    return df_long, df_wide

 # ---------- Helpers for Module 1 (wing analysis) ----------
def _read_shap_mean_table(path: Path) -> pd.DataFrame:
    """
    Strict reader for cls_shap_meanabs.csv.
    Rules:
      - MUST read SHAP magnitudes from column 'mean_raw_shap' (no fallbacks).
      - Preferred zone source: column 'feature' with exact pattern 'zone_<id>_count' (case-insensitive).
      - Fallback zone source: integer column 'zone'.
      - Only zones in 1..24 are kept.
    Returns a DataFrame with columns ['zone','mean_raw_shap'] where 'zone' is int.
    """
    df = pd.read_csv(path)

    if "mean_raw_shap" not in df.columns:
        raise ValueError(f"'mean_raw_shap' column not found in {path}. Found columns: {list(df.columns)}")

    # Preferred: parse from 'feature' like 'zone_14_count'
    zone_series = None
    if "feature" in df.columns:
        import re
        pat = re.compile(r"^zone_(\d+)_count$", re.IGNORECASE)
        extracted = df["feature"].astype(str).str.extract(pat)
        if extracted.shape[1] == 1:
            zone_series = extracted[0]

    if zone_series is None and "zone" in df.columns:
        zone_series = df["zone"]

    if zone_series is None:
        raise ValueError(f"Neither a parsable 'feature' (zone_XX_count) nor an integer 'zone' column found in {path}.")

    out = pd.DataFrame({
        "zone": pd.to_numeric(zone_series, errors="coerce"),
        "mean_raw_shap": df["mean_raw_shap"]
    })
    # keep only valid pitch zones 1..24
    out = out[(out["zone"] >= 1) & (out["zone"] <= 24)].dropna(subset=["zone"]).copy()
    out["zone"] = out["zone"].astype(int)
    return out[["zone", "mean_raw_shap"]]

def _map_zone_columns(cols, zones):
    """
    Try to find column names for requested zones in a wide player table.
    Accepts variants like 'zone_13','Zone13','Z13','13'.
    Returns dict {zone:int -> column_name:str}
    """
    mapping = {}
    import re
    lower = {str(c).lower(): c for c in cols}
    for z in zones:
        candidates = [
            f"zone_{z}", f"zone{z}", f"z{z}", f"{z}"
        ]
        found = None
        for cand in candidates:
            if cand in lower:
                found = lower[cand]
                break
        if found is None:
            # regex boundary match to avoid 2 vs 20
            pat = re.compile(rf"(^|[^0-9])0*{z}([^0-9]|$)")
            for lc, orig in lower.items():
                if pat.search(lc):
                    found = orig
                    break
        if found:
            mapping[z] = found
    return mapping

def analyze_wings_to_json(
    shap_csv: Path,
    player_wide_csv: Path,
    out_json_path: Path,
    right_zones = (13,14,17,18,21),
    left_zones = (15,16,19,20,24),
    topk_players_per_zone: int = 3,
    min_presence_per_player: int = 1
) -> dict:
    """
    Module 1:
      - Reads season-level mean shap per zone
      - Scores right/left wings by sum of mean shap across their zones
      - Picks the 3 weakest zones from the weaker wing (lowest mean shap)
      - For each weak zone, selects top-K players by presence count from the wide table
      - Exports a JSON ready for AI consumption
    """
    shap_df = _read_shap_mean_table(shap_csv)
    # score wings
    right_score = shap_df.loc[shap_df["zone"].isin(right_zones), "mean_raw_shap"].sum()
    left_score  = shap_df.loc[shap_df["zone"].isin(left_zones),  "mean_raw_shap"].sum()
    stronger = "right" if right_score >= left_score else "left"
    weaker  = "left" if stronger == "right" else "right"
    weak_zones_list = list(right_zones if weaker == "right" else left_zones)
    # pick worst 3 zones on weaker side
    weak_side_vals = shap_df[shap_df["zone"].isin(weak_zones_list)].copy()
    weak_side_vals = weak_side_vals.sort_values("mean_raw_shap", ascending=True)
    worst3 = weak_side_vals.head(3)[["zone","mean_raw_shap"]].to_dict(orient="records")

    # read player wide table
    pw = pd.read_csv(player_wide_csv)
    # map zone columns
    zone_cols_map = _map_zone_columns(pw.columns, [rec["zone"] for rec in worst3])
    missing = [rec["zone"] for rec in worst3 if rec["zone"] not in zone_cols_map]
    if missing:
        raise ValueError(f"Missing zone columns in player wide csv for zones: {missing}. Columns={list(pw.columns)}")
    # locate player id/name columns
    name_col = None
    for cand in ["player_name","Player","name","player"]:
        if cand in pw.columns:
            name_col = cand; break
    if name_col is None:
        raise ValueError("player name column not found in player_zone_counts_wide.csv")
    id_col = None
    for cand in ["player_id","id","PlayerId","Player_ID"]:
        if cand in pw.columns:
            id_col = cand; break

    top_players = []
    for rec in worst3:
        z = rec["zone"]
        zcol = zone_cols_map[z]
        tmp = pw[[name_col, zcol] + ([id_col] if id_col else [])].rename(columns={zcol: "count"}).copy()
        tmp = tmp[tmp["count"] >= min_presence_per_player]
        tmp = tmp.sort_values("count", ascending=False).head(topk_players_per_zone)
        players = []
        for _, r in tmp.iterrows():
            entry = {"player_name": r[name_col], "count": int(r["count"])}
            if id_col:
                entry["player_id"] = int(r[id_col])
            players.append(entry)
        top_players.append({"zone": int(z), "players": players})

    payload = {
        "version": "1.0",
        "scope": "season_aggregate",
        "team_id": TARGET_TEAM_ID,
        "wing_comparison": {
            "right_score": float(right_score),
            "left_score": float(left_score),
            "stronger_wing": stronger,
            "weaker_wing": weaker,
            "weak_zones": [{"zone": int(x["zone"]), "mean_raw_shap": float(x["mean_raw_shap"])} for x in worst3],
            "top_players_in_weak_zones": top_players
        }
    }
    # save json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wing analysis JSON saved → {out_json_path}")
    return payload

def build_case_finishing_shots(
    base_features_csv: Path,
    matches_root: Path,
    target_team_id: int = TARGET_TEAM_ID,
    finishing_zones: Tuple[int, int] = (22, 23),
) -> pd.DataFrame:
    """
    Build a case-level table with the shooter for finishing zones, using a light and strict path:
      - Trust 'end_index' as the exact shot event index for the case.
      - Trust 'End_Zone' as the final zone of that shot.
      - Read only the single event whose 'index' == end_index to fetch shooter/xG.
    Requirements in BaseFeatures:
      • Columns: "Case number", "end_index", "End_Zone", and a resolvable match path (via _resolve_events_path()).
    Output columns:
      ["Case number", "zone", "player_id", "player_name", "event_index", "xg"]
    """
    bf = pd.read_csv(base_features_csv)
    req_cols = ["Case number", "end_index", "End_Zone"]
    for c in req_cols:
        if c not in bf.columns:
            raise ValueError(f"BaseFeatures must contain column '{c}'. Available: {list(bf.columns)}")

    # Keep only cases whose End_Zone is finishing_zones
    bm = bf["End_Zone"].isin(list(finishing_zones))
    bf = bf[bm].copy()
    if bf.empty:
        return pd.DataFrame(columns=["Case number", "zone", "player_id", "player_name", "event_index", "xg"])

    events_cache: Dict[Path, List[dict]] = {}
    rows = []

    for i, row in bf.iterrows():
        e_idx = _safe_int(row.get("end_index"))
        z = _safe_int(row.get("End_Zone"))
        if e_idx is None or z is None:
            continue
        ev_path = _resolve_events_path(row, matches_root)
        if ev_path is None:
            continue

        # Load events once per match
        if ev_path not in events_cache:
            try:
                with open(ev_path, "r", encoding="utf-8") as f:
                    events_cache[ev_path] = json.load(f)
            except Exception as e:
                print(f"[WARN] (CaseFin-lite) Cannot load events {ev_path}: {e}")
                continue

        # Direct lookup: the exact event with index == end_index
        ev = next((ev for ev in events_cache[ev_path] if isinstance(ev.get("index"), int) and ev["index"] == e_idx), None)
        if ev is None:
            print(f"[WARN] (CaseFin-lite) end_index {e_idx} not found in events for case {row['Case number']}")
            continue

        # Optional safety checks
        team = ev.get("team") or {}
        if target_team_id is not None and int(team.get("id", -1)) != int(target_team_id):
            # Different team shot at end_index → skip this case to avoid cross-team contamination
            print(f"[DBG] (CaseFin-lite) end_index belongs to other team for case {row['Case number']}; skipping.")
            continue

        t = (ev.get("type") or {}).get("name", "")
        if str(t).strip().lower() != "shot" and "shot" not in ev:
            # If end_index is not a shot (data drift), skip conservatively
            print(f"[DBG] (CaseFin-lite) end_index event is not a Shot for case {row['Case number']}; skipping.")
            continue

        player = ev.get("player") or {}
        pid = _safe_int(player.get("id"))
        pname = str(player.get("name")) if player.get("name") else ""
        if pid is None:
            continue

        # try xg
        xg = None
        shot_data = ev.get("shot") or {}
        for k in ("statsbomb_xg", "xg", "expected_goals"):
            if k in shot_data:
                try:
                    xg = float(shot_data[k])
                    break
                except Exception:
                    pass

        rows.append({
            "Case number": row["Case number"],
            "zone": int(z),
            "player_id": pid,
            "player_name": pname,
            "event_index": int(e_idx),
            "xg": xg
        })

    if not rows:
        return pd.DataFrame(columns=["Case number", "zone", "player_id", "player_name", "event_index", "xg"])

    df_cases = pd.DataFrame(rows).sort_values(["Case number", "zone"]).reset_index(drop=True)
    return df_cases

def analyze_finishing_to_json(
    shap_mean_csv: Path,
    shap_values_csv: Path,
    case_finishing_csv: Path,
    out_json_path: Path,
    min_cases_per_player: int = 5,
    topk_players: int = 5,
) -> dict:
    """
    Module 2 (no negativity requirement):
      1) Decide weaker finishing zone by comparing season-level mean_raw_shap for 'zone_22_count' vs 'zone_23_count'.
      2) Join case-level SHAP values with case-level shooter mapping in that zone.
      3) Rank players by lower average SHAP contribution in the weaker zone (lower=worse), with tie-breakers.
    Strict column rules:
      - shap_mean_csv must have: 'feature' with 'zone_22_count'/'zone_23_count' and 'mean_raw_shap'.
      - shap_values_csv must have: 'Case number', 'zone_22_count', 'zone_23_count'.
      - case_finishing_csv must have: 'Case number', 'zone', 'player_id', 'player_name'.
    """
    # 1) weaker zone from season-level means
    sm = pd.read_csv(shap_mean_csv)
    if "feature" not in sm.columns or "mean_raw_shap" not in sm.columns:
        raise ValueError("shap_mean_csv must contain 'feature' and 'mean_raw_shap'.")
    sm = sm[sm["feature"].isin(["zone_22_count", "zone_23_count"])]
    if sm.empty or sm["feature"].nunique() < 2:
        raise ValueError("Could not find both 'zone_22_count' and 'zone_23_count' in shap_mean_csv.")
    mean22 = float(sm.loc[sm["feature"]=="zone_22_count","mean_raw_shap"].iloc[0])
    mean23 = float(sm.loc[sm["feature"]=="zone_23_count","mean_raw_shap"].iloc[0])
    weaker_zone = 22 if mean22 <= mean23 else 23

    # 2) read shap values (cases)
    sv = pd.read_csv(shap_values_csv)
    needed_cols = ["Case number", "zone_22_count", "zone_23_count"]
    for c in needed_cols:
        if c not in sv.columns:
            raise ValueError(f"shap_values_csv missing required column: {c}")

    # keep only the shap column for the weaker zone
    shap_col = f"zone_{weaker_zone}_count"
    keep = sv[["Case number", shap_col]].rename(columns={shap_col: "shap_value"})

    # 3) read case→shooter mapping
    cf = pd.read_csv(case_finishing_csv)
    req_cf = ["Case number", "zone", "player_id", "player_name"]
    for c in req_cf:
        if c not in cf.columns:
            raise ValueError(f"case_finishing_csv missing required column: {c}")
    cf = cf[cf["zone"] == weaker_zone]

    # join on Case number
    joined = keep.merge(cf, on="Case number", how="inner")
    if joined.empty:
        payload = {
            "version": "1.0",
            "scope": "season_aggregate",
            "team_id": TARGET_TEAM_ID,
            "finishing_analysis": {
                "zone_means": {"22": mean22, "23": mean23},
                "weaker_finishing_zone": weaker_zone,
                "note": "No cases with a finishing shot in the weaker zone were found.",
                "players": []
            }
        }
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Finishing analysis JSON saved → {out_json_path}")
        return payload

    # 4) build ranking metrics (lower SHAP = worse contribution)
    zone_median = float(joined["shap_value"].median())
    grp = (joined.groupby(["player_id","player_name"], as_index=False)
                 .agg(n_cases=("Case number","nunique"),
                      mean_shap=("shap_value","mean"),
                      p25_shap=("shap_value", lambda s: s.quantile(0.25)),
                      share_below_median=("shap_value", lambda s: float((s <= zone_median).mean()))))
    grp = grp[grp["n_cases"] >= min_cases_per_player].copy()
    grp = grp.sort_values(["mean_shap","n_cases"], ascending=[True, False]).reset_index(drop=True)
    top = grp.head(topk_players)

    payload = {
        "version": "1.0",
        "scope": "season_aggregate",
        "team_id": TARGET_TEAM_ID,
        "finishing_analysis": {
            "zone_means": {"22": mean22, "23": mean23},
            "weaker_finishing_zone": weaker_zone,
            "ranking_metric": "mean_shap (lower=worse), tie → n_cases desc",
            "zone_median_shap": zone_median,
            "players": top.to_dict(orient="records")
        }
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Finishing analysis JSON saved → {out_json_path}")
    return payload

# ---------- Finishing analysis: shots per player in zones 22 & 23 ----------
def build_finishing_shots_summary(
    base_features_csv: Path,
    matches_root: Path,
    target_team_id: int = TARGET_TEAM_ID,
    finishing_zones: Tuple[int, int] = (22, 23),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      - df_long: columns = [player_id, player_name, zone, shots]
      - df_summary: columns = [player_id, player_name, shots_z22, shots_z23, shots_finishing_total]
    Policy:
      - Uses the same path resolution, zone function validation, and team-id filtering
        as the rest of the pipeline to avoid mixing wrong matches/teams.
      - Only counts events of type 'Shot' (StatsBomb) whose location maps into finishing_zones.
    """
    # Load BaseFeatures
    bf = pd.read_csv(base_features_csv)
    print(f"[INFO] (Finishing) BaseFeatures loaded: {bf.shape[0]} rows, {bf.shape[1]} cols")

    # Resolve zone function (validated against expected 6x4 numbering)
    zone_fn = get_zone_function()

    # Cache events by path
    events_cache: Dict[Path, List[dict]] = {}

    # Accumulators
    shots_per_player_zone: Dict[int, DefaultDict[int, int]] = {}
    player_names: Dict[int, str] = {}

    def _is_shot(ev: dict) -> bool:
        t = ev.get("type") or {}
        tname = (t.get("name") or "").strip().lower()
        if tname == "shot":
            return True
        # Fallback: some feeds include a 'shot' object
        return "shot" in ev

    for i, row in bf.iterrows():
        s_idx = _safe_int(row.get("start_index"))
        e_idx = _safe_int(row.get("end_index"))
        if s_idx is None or e_idx is None:
            print(f"[WARN] (Finishing) Row {i}: missing start/end_index; skipping.")
            continue

        ev_path = _resolve_events_path(row, matches_root)
        if ev_path is None:
            print(f"[WARN] (Finishing) Row {i}: could not resolve event.json path; skipping.")
            continue

        if ev_path not in events_cache:
            try:
                with open(ev_path, "r", encoding="utf-8") as f:
                    events_cache[ev_path] = json.load(f)
                print(f"[INFO] (Finishing) Loaded events: {ev_path}")
            except Exception as e:
                print(f"[WARN] (Finishing) Cannot load events {ev_path}: {e}")
                events_cache[ev_path] = []

        for ev in _iter_events_between(events_cache[ev_path], s_idx, e_idx):
            # Team filter
            team = ev.get("team") or {}
            if target_team_id is not None and int(team.get("id", -1)) != int(target_team_id):
                continue
            # Must be a shot with valid location
            if not _is_shot(ev):
                continue
            loc = ev.get("location")
            player = ev.get("player") or {}
            if not (isinstance(loc, (list, tuple)) and len(loc) >= 2 and player):
                continue

            pid = _safe_int(player.get("id"))
            pname = str(player.get("name")) if player.get("name") else None
            if pid is None:
                continue

            x, y = float(loc[0]), float(loc[1])
            z = int(zone_fn(x, y))
            if z not in finishing_zones:
                continue

            shots_per_player_zone.setdefault(pid, defaultdict(int))[z] += 1
            if pname:
                player_names.setdefault(pid, pname)

    # Build long-form
    long_records = []
    for pid, zmap in shots_per_player_zone.items():
        for z in sorted(zmap.keys()):
            long_records.append({
                "player_id": pid,
                "player_name": player_names.get(pid, ""),
                "zone": int(z),
                "shots": int(zmap[z]),
            })

    if not long_records:
        df_long = pd.DataFrame(columns=["player_id", "player_name", "zone", "shots"])
        df_summary = pd.DataFrame(columns=["player_id", "player_name", "shots_z22", "shots_z23", "shots_finishing_total"])
        return df_long, df_summary

    df_long = (
        pd.DataFrame(long_records)
        .sort_values(["player_name", "player_id", "zone"])
        .reset_index(drop=True)
    )

    # Summary wide
    df_summary = df_long.pivot_table(
        index=["player_id", "player_name"],
        columns="zone",
        values="shots",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # Ensure columns for 22 and 23 exist
    for z in finishing_zones:
        if z not in df_summary.columns:
            df_summary[z] = 0

    # Rename to shots_z22 / shots_z23
    rename_map = {22: "shots_z22", 23: "shots_z23"}
    df_summary = df_summary.rename(columns=rename_map)
    for old, new in list(rename_map.items()):
        if old in df_summary.columns:
            # if numeric column name persists, drop it after renaming
            pass

    # If numeric zone columns remain, map them too
    for col in list(df_summary.columns):
        if isinstance(col, int) and col in rename_map:
            df_summary = df_summary.rename(columns={col: rename_map[col]})

    # Add total
    if "shots_z22" not in df_summary.columns:
        df_summary["shots_z22"] = 0
    if "shots_z23" not in df_summary.columns:
        df_summary["shots_z23"] = 0
    df_summary["shots_finishing_total"] = df_summary["shots_z22"] + df_summary["shots_z23"]

    # Sort by total desc, then by z23 then z22
    df_summary = df_summary.sort_values(
        by=["shots_finishing_total", "shots_z23", "shots_z22", "player_name", "player_id"],
        ascending=[False, False, False, True, True]
    ).reset_index(drop=True)

    return df_long, df_summary

def collect_unique_players(base_dir: Path) -> pd.DataFrame:
    lineup_files = sorted(base_dir.rglob("lineup.json"))
    seen = {}
    total_files = 0
    hit_files = 0

    # If no lineup.json files are present, fall back to scanning event files
    if not lineup_files:
        event_files = sorted(list(base_dir.rglob("event.json")) + list(base_dir.rglob("events.json")))
        if not event_files:
            print(f"[WARN] No lineup.json or event.json found under {base_dir}. Returning empty player list.")
            return pd.DataFrame(columns=["player_id", "player_name"])

        print(f"[INFO] No lineup.json found. Falling back to scan {len(event_files)} event file(s) for players of the target team.")
        for fp in event_files:
            total_files += 1
            data = read_json(fp)
            if not isinstance(data, list):
                continue
            # Collect any player ids/names from events that belong to the target team
            file_hit = False
            for ev in data:
                team = ev.get("team") or {}
                # Match by team id when available; otherwise by name
                id_ok = (TARGET_TEAM_ID is None) or (int(team.get("id", -1)) == int(TARGET_TEAM_ID))
                name_ok = True
                tname = (team.get("name") or "").strip().lower()
                if TARGET_TEAM_NAME:
                    name_ok = (tname == TARGET_TEAM_NAME.strip().lower())
                if not (id_ok and name_ok):
                    continue
                player = ev.get("player") or {}
                pid = player.get("id")
                pname = player.get("name")
                if pid is None or not pname:
                    continue
                try:
                    pid = int(pid)
                except Exception:
                    continue
                seen.setdefault(pid, str(pname))
                file_hit = True
            if file_hit:
                hit_files += 1

        df = pd.DataFrame(
            [{"player_id": pid, "player_name": pname} for pid, pname in seen.items()]
        ).sort_values(["player_name", "player_id"]).reset_index(drop=True)

        print(f"Scanned event files: {total_files} | files with target team players: {hit_files}")
        print(f"Unique players found (from events): {len(df)}")
        return df

    for fp in lineup_files:
        total_files += 1
        data = read_json(fp)
        if data is None:
            continue
        players = extract_players_from_lineup(data)
        if players:
            hit_files += 1
        for pid, pname in players:
            # If a player appears with different names, keep the first seen name
            seen.setdefault(pid, pname)

    df = pd.DataFrame(
        [{"player_id": pid, "player_name": pname} for pid, pname in seen.items()]
    ).sort_values(["player_name", "player_id"]).reset_index(drop=True)

    print(f"Scanned lineup files: {total_files} | files with target team: {hit_files}")
    print(f"Unique players found: {len(df)}")
    return df


# ============= GEMINI AI INTEGRATION =============
import os

def call_gemini_for_zone_analysis(prompt_text: str, image_path: str, json_files: list[str]) -> str:
    """Call Gemini with zone heatmap image and JSON analysis files."""
    from google import genai

    # Use env var if present; otherwise fallback to provided key
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyD3Y6wabXe5rMidJTpmk6785rSI4K2gjvk")

    # Read JSON files and include their content in the prompt
    json_content = ""
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_content += f"\n--- {Path(json_file).name} ---\n{json.dumps(data, indent=2)}\n"
        except Exception as e:
            json_content += f"\n--- {Path(json_file).name} (ERROR) ---\nCould not read file: {e}\n"

    # Compose full prompt with image and JSON data
    full_prompt = f"{prompt_text}\n\n=== JSON DATA FOR VERIFICATION ===\n{json_content}\n\n=== IMAGE ===\n[IMAGE: {image_path}]\n"

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=full_prompt,
    )
    text = getattr(resp, "text", None)
    if text is None:
        print("[GEMINI][WARN] Empty response.text in call_gemini_for_zone_analysis; returning empty string.")
        text = ""
    return text

# The complete prompt for Attacking Zone & Player Analysis
ZONE_ANALYSIS_PROMPT = """
Attacking Zone & Player Analysis (Season-Level)

Context:
All inputs are season-aggregated. Keep the analysis at season level; prioritise stable patterns over single-match noise.
Write in coach-friendly football language – no data-science terms, no technical metrics. Think of it as talking to a head coach during a tactical briefing: simple, direct, and actionable.

Important:
The JSON files are provided for you to verify your claims internally. Do not mention SHAP values, scores, or raw numbers in the output. Instead, express the insights as football takeaways (e.g., "the left wing struggled to create" instead of "left_score = -0.0143").

Part 1 – Zone-Level Performance (zones only)

Use the zone heatmap to explain which areas of the opponent half helped create dangerous attacks (red) and which struggled (blue).
Stay focused on zones only (no players yet).
Use the JSONs only to cross-check accuracy:

wing_analysis.json → verify which wing is stronger/weaker.
finishing_analysis.json → verify which finishing zone (22 vs 23) is weaker.
Output: 1 clear, short paragraph, written like tactical feedback.
After the text, reference the zone heatmap image.

Part 2 – Wing & Player Analysis

Use wing_analysis.json 
Compare right vs left effectiveness in plain terms (stronger vs weaker).
For the weaker wing, highlight its problem zones (from weak_zones).
Mention the players most active in those weak zones (top_players_in_weak_zones).
Combine tactical insight + player mentions in 1–2 short paragraphs.
After the text, reference the wing visuals.

Part 3 – Finishing Analysis (zones 22 & 23)

Use finishing_analysis.json.
Clearly state which finishing zone is weaker (22 vs 23).
Highlight the players who shoot most often in the weaker zone and explain the tactical problem (e.g., "too many rushed shots" / "low-quality positions").
Write 1–2 short paragraphs in simple, coach-friendly language.
After the text, reference the finishing visuals.

End

Finish with one one-line tactical recommendation that ties together all three parts (wing usage + finishing focus).

Important Number Rules:
• Use football-friendly counts (e.g., number of shots, actions, cases, % of weak outcomes).
• Write them in brackets right after the player's name.
Example: "Alejandro Grimaldo (24 shots), Patrik Schick (14), Amine Adli (12)."
• Percentages are allowed only if they describe share of weak vs strong outcomes.
Example: "Over half of Adli's shots (58%) in this zone were below-average quality."
• Do NOT mention SHAP values, model coefficients, or technical impact metrics.
• Do NOT say "impact score" or "model importance."
• Keep numbers strictly tied to football events (shots, actions, cases).
"""

def generate_zone_analysis_report() -> None:
    """Generate zone analysis report using Gemini AI."""
    
    # Define file paths based on existing OUTPUT paths
    image_path = str(ROOT_OUTPUT / "plots" / "ZoneAnalysis" / "zone_shap_heatmap.png")
    json_files = [
        str(OUTPUT_FINISHING_JSON),  # finishing_analysis.json
        str(OUTPUT_WING_JSON)        # wing_analysis.json
    ]
    
    # Check if required files exist
    missing_files = []
    if not Path(image_path).exists():
        missing_files.append(image_path)
    for json_file in json_files:
        if not Path(json_file).exists():
            missing_files.append(json_file)
    
    if missing_files:
        print(f"[WARN] Missing files for zone analysis: {missing_files}")
        return
    
    try:
        # Call Gemini AI
        ai_text = call_gemini_for_zone_analysis(ZONE_ANALYSIS_PROMPT, image_path, json_files)
        
        # Create text output directory and append to final report
        text_dir = ROOT_OUTPUT / "text"
        text_dir.mkdir(parents=True, exist_ok=True)
        final_report_path = text_dir / "FinalReport.txt"
        
        section_header = "\n## Section – Attacking Zone & Player Analysis (Season-Level)\n\n"
        
        # Helper function for appending to report
        def append_to_report(text: str, file_path: Path) -> None:
            """Append text to a UTF-8 report file, creating it if missing."""
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("a", encoding="utf-8") as f:
                f.write(text)
        
        append_to_report(section_header + (ai_text or "") + "\n", final_report_path)
        
        print(f"[INFO] Gemini zone analysis insights appended to {final_report_path}")
        
    except Exception as e:
        print(f"[WARN] Gemini zone analysis skipped: {e}")

# ============= END GEMINI INTEGRATION =============



# Entry point wrapper for main.py compatibility
def run():
    """Entry point wrapper for main.py compatibility."""
    # Ensure output directories exist
    OUT_CSVJSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CFG] BASE_DIR = {BASE_DIR}")
    print(f"[CFG] BASE_FEATURES_CSV = {BASE_FEATURES_CSV}")
    print(f"[CFG] SHAP_MEANABS_CSV = {SHAP_MEANABS_CSV}")
    # 1) Build/refresh the unique players list from lineups (season scope)
    df_players = collect_unique_players(BASE_DIR)
    print(df_players)
    df_players.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved player list to: {OUTPUT_CSV}")

    # 2) Build player x zone counts during detected attacking cases (from BaseFeatures)
    try:
        df_long, df_wide = build_player_zone_counts(
            base_features_csv=BASE_FEATURES_CSV,
            matches_root=BASE_DIR,
            target_team_id=TARGET_TEAM_ID,
            count_carry_end=COUNT_CARRY_END,
        )
        print("[INFO] Player-zone counts (long):\n", df_long.head())
        print("[INFO] Player-zone counts (wide):\n", df_wide.head())
        # Save
        df_long.to_csv(OUTPUT_PLAYER_ZONE_LONG, index=False, encoding="utf-8")
        df_wide.to_csv(OUTPUT_PLAYER_ZONE_WIDE, index=False, encoding="utf-8")
        print(f"Saved long counts to: {OUTPUT_PLAYER_ZONE_LONG}")
        print(f"Saved wide counts to: {OUTPUT_PLAYER_ZONE_WIDE}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Hint: check BASE_FEATURES_CSV path and that CaseDetection.py produced it.")

    # 3) Finishing shots in zones 22 & 23 (season-level, within case windows)
    try:
        df_fin_long, df_fin_summary = build_finishing_shots_summary(
            base_features_csv=BASE_FEATURES_CSV,
            matches_root=BASE_DIR,
            target_team_id=TARGET_TEAM_ID,
            finishing_zones=(22, 23),
        )
        print("[INFO] Finishing shots (long):\n", df_fin_long.head())
        print("[INFO] Finishing shots (summary):\n", df_fin_summary.head())
        df_fin_long.to_csv(OUTPUT_FINISHING_SHOTS_LONG, index=False, encoding="utf-8")
        df_fin_summary.to_csv(OUTPUT_FINISHING_SHOTS_SUMMARY, index=False, encoding="utf-8")
        print(f"Saved finishing shots (long) to: {OUTPUT_FINISHING_SHOTS_LONG}")
        print(f"Saved finishing shots (summary) to: {OUTPUT_FINISHING_SHOTS_SUMMARY}")
    except FileNotFoundError as e:
        print(f"[ERROR] (Finishing) {e}")
        print("Hint: verify BASE_FEATURES_CSV path and events resolution for finishing shots.")

    # 4) Module 1: Right vs Left wing analysis → JSON for AI
    try:
        analyze_wings_to_json(
            shap_csv=SHAP_MEANABS_CSV,
            player_wide_csv=OUTPUT_PLAYER_ZONE_WIDE,
            out_json_path=OUTPUT_WING_JSON,
            right_zones=(13,14,17,18,21),
            left_zones=(15,16,19,20,24),
            topk_players_per_zone=3,
            min_presence_per_player=1
        )
    except Exception as e:
        print(f"[ERROR] (Wing JSON) {e}")

    # === Visual B: Grouped bar chart for top 3 players per weak wing zone ===
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        # Load wing_analysis.json
        with open(OUTPUT_WING_JSON, "r", encoding="utf-8") as f:
            wing_data = json.load(f)
        weak_zones = wing_data["wing_comparison"]["weak_zones"]
        top_players_in_weak_zones = wing_data["wing_comparison"]["top_players_in_weak_zones"]
        # Extract zone numbers
        zone_numbers = [z["zone"] for z in weak_zones]
        # For each zone, top 3 players and their counts
        player_names = []
        counts_by_zone_and_player = []
        for zrec in top_players_in_weak_zones:
            players = zrec["players"]
            # Pad to 3 if less than 3 players
            while len(players) < 3:
                players.append({"player_name": "", "count": 0})
            counts_by_zone_and_player.append([p["count"] for p in players])
            player_names.append([p["player_name"] for p in players])
        # Transpose for grouping: 3 players x N zones
        counts_arr = np.array(counts_by_zone_and_player).T  # shape: (3, N)
        player_name_labels = []
        # Use the first non-empty name for each player position
        for i in range(3):
            name = ""
            for names in player_names:
                if names[i]:
                    name = names[i]
                    break
            player_name_labels.append(name if name else f"Player {i+1}")
        x = np.arange(len(zone_numbers))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(3):
            ax.bar(x + (i-1)*width, counts_arr[i], width, label=player_name_labels[i])
        ax.set_xlabel('Zone')
        ax.set_ylabel('Number of Actions/Shots')
        ax.set_xticks(x)
        ax.set_xticklabels([str(z) for z in zone_numbers])
        ax.set_title("Top 3 Players per Weak Wing Zone")
        ax.legend()
        output_folder = OUT_PLOTS_DIR
        plt.tight_layout()
        plt.savefig(str(output_folder / "zone_weak_player_bar.png"))
        plt.close()
        print(f"[INFO] Saved Visual B to {output_folder / 'zone_weak_player_bar.png'}")
    except Exception as e:
        print(f"[ERROR] (Visual B) {e}")

   
    # 5) Build case-level finishing shooter mapping (zones 22 & 23)
    try:
        df_case_fin = build_case_finishing_shots(
            base_features_csv=BASE_FEATURES_CSV,
            matches_root=BASE_DIR,
            target_team_id=TARGET_TEAM_ID,
            finishing_zones=(22, 23),
        )
        df_case_fin.to_csv(OUTPUT_CASE_FINISHING_SHOTS, index=False, encoding="utf-8")
        print(f"Saved case finishing shots to: {OUTPUT_CASE_FINISHING_SHOTS}")
    except Exception as e:
        print(f"[ERROR] (CaseFin) {e}")

    # 6) Module 2: Finishing zones analysis → JSON for AI (no negativity requirement)
    try:
        analyze_finishing_to_json(
            shap_mean_csv=SHAP_MEANABS_CSV,
            shap_values_csv=PROJECT_ROOT / "outputs" / "CSV-JSON" / "classification_median" / "explainability" / "extended_all" / "cls_shap_values.csv",
            case_finishing_csv=OUTPUT_CASE_FINISHING_SHOTS,
            out_json_path=OUTPUT_FINISHING_JSON,
            min_cases_per_player=5,
            topk_players=5
        )
    except Exception as e:
        print(f"[ERROR] (Finishing JSON) {e}")
    # === Visual C: Horizontal bar chart for finishing in weak zone (Top 5 Players, SHAP blue overlay) ===
    try:
        # Load finishing_analysis.json
        with open(OUTPUT_FINISHING_JSON, "r", encoding="utf-8") as f:
            finishing_data = json.load(f)
        fin = finishing_data.get("finishing_analysis", {})
        weak_zone = fin.get("weaker_finishing_zone")
        if weak_zone is None:
            raise ValueError("No weaker_finishing_zone in finishing_analysis.json")
        # Load case-level shooter mapping and SHAP values
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        case_finishing_csv = OUTPUT_CASE_FINISHING_SHOTS
        shap_values_csv = PROJECT_ROOT / "outputs" / "CSV-JSON" / "classification_median" / "explainability" / "extended_all" / "cls_shap_values.csv"
        df_cases = pd.read_csv(case_finishing_csv)
        df_cases = df_cases[df_cases["zone"] == weak_zone]
        shap_col = f"zone_{weak_zone}_count"
        df_shap = pd.read_csv(shap_values_csv)
        if shap_col not in df_shap.columns:
            raise ValueError(f"SHAP values file missing {shap_col}")
        merged = df_cases.merge(df_shap[["Case number", shap_col]], on="Case number", how="inner")
        merged = merged.rename(columns={shap_col: "shap_value"})
        # For each player, count total shots and number of weak finishing shots (shap_value < 0)
        summary = (
            merged.groupby("player_name", as_index=False)
            .agg(total_shots=("Case number", "count"),
                 weak_shots=("shap_value", lambda s: (s < 0).sum()))
        )
        # Sort by total shots desc, take top 5
        summary = summary.sort_values("total_shots", ascending=False).head(5)
        # Prepare data for plotting
        players = summary["player_name"].tolist()
        total_shots = summary["total_shots"].tolist()
        weak_shots = summary["weak_shots"].tolist()
        y = np.arange(len(players))
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot total shots in grey
        ax.barh(y, total_shots, color="lightgrey", label="Total Shots")
        # Overlay weak finishing shots in a single SHAP-blue tone (no gradient)
        cmap = plt.cm.Blues
        weak_color = cmap(0.75)  # fixed blue
        for i, (ws, ts) in enumerate(zip(weak_shots, total_shots)):
            ax.barh(
                y[i],
                ws,
                color=weak_color,
                label="Finishing performance drops (shots)" if i == 0 else ""
            )
        ax.set_yticks(y)
        ax.set_yticklabels(players)
        ax.set_xlabel("Number of Shots")
        ax.set_title("Finishing Performance in Weak Zone (Top 5 Players)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        output_folder = OUT_PLOTS_DIR
        plt.savefig(str(output_folder / "finishing_zone_bar.png"))
        plt.close()
        print(f"[INFO] Saved Visual C to {output_folder / 'finishing_zone_bar.png'}")
    except Exception as e:
        print(f"[ERROR] (Visual C) {e}")

    # 7) GEMINI AI ANALYSIS – append narrative to FinalReport.txt
    generate_zone_analysis_report()

if __name__ == "__main__":
    run()