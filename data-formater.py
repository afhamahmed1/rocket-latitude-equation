#!/usr/bin/env python3
"""
build_apogee_lat_dataset.py

Creates a one-row-per-flight dataset for the task:
    TARGET = latitude at apogee

Works with the file formats:
- Raven .csv (with columns like "Time@[Altitude (Baro-Ft-AGL)]" and "[Altitude (Baro-Ft-AGL)]")
- Featherweight GPS .csv/.xlsx (columns like UNIXTIME, ALT, LAT, LON, ...)
- Trimmed .TXT (OPTIONAL: if it contains actual GPS lines; many Trimmed files do NOT)

Recommended folder layout
------------------------
project/
  build_apogee_lat_dataset.py
  data/
    altimeter/
      <Raven .csv and/or *_Trimmed.TXT>
    gps/
      <Featherweight on-board GPS .csv or edited .xlsx>
    metadata/
      manifest.csv           # recommended explicit pairing (see below)
      motors_manual.csv      # optional motor stats (no .eng needed)
      rockets_manual.csv     # optional rocket constants
      flights_metadata.csv   # optional per-flight extras (wind, RH, etc.)
  out/
    flights_dataset.csv
    cleaned_tracks/
      <flight_id>_raven_alt.csv
      <flight_id>_gps.csv
      <flight_id>_txtgps.csv (only if TXT contains GPS)

manifest.csv format
-------------------
flight_id,rocket_id,motor_name,altimeter_file,gps_file
20210417_Adventurer_J510W_1,Madcow Adventurer,J510W,data/altimeter/20210417_Adventurer_J510W_1_Raven.csv,data/gps/20210417_Adventurer_J510W_1_GPS.csv
...

motor_name should be the BASE motor identifier for thrust stats.
Example: use "1115J530-15A" (not "1115J530-15A-15").

motors_manual.csv format (optional)
-----------------------------------
motor_name,manufacturer,avg_thrust_N,initial_thrust_N,max_thrust_N,total_impulse_Ns,burn_time_s
1115J530-15A,Cesaroni,531.2,723.2,828.8,1115.5,2.1

Notes on apogee latitude computation
------------------------------------
We compute (when possible):
1) GPS-peak method: apogee index = max GPS altitude after GPS-liftoff.
2) Altimeter-timed method: apogee time (relative to liftoff) from Raven baro AGL,
   then interpolate GPS latitude at that relative time (GPS time also relative to its liftoff).

Final target selection:
- Prefer GPS-peak if GPS altitude exists and shows an ascent.
- Otherwise use altimeter-timed (if Raven exists and GPS covers that time).

Install deps:
  pip install pandas numpy openpyxl python-dateutil
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


# -----------------------------
# Logging
# -----------------------------
def log(msg: str, log_path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# -----------------------------
# Motor stats (manual)
# -----------------------------
@dataclass
class MotorStats:
    motor_name: str
    manufacturer: str = ""
    avg_thrust_N: float = float("nan")
    initial_thrust_N: float = float("nan")
    max_thrust_N: float = float("nan")
    total_impulse_Ns: float = float("nan")
    burn_time_s: float = float("nan")


def normalize_motor_name(name: str) -> str:
    """
    Normalize motor identifiers so that delay suffixes don't break matching:
      - "1115J530-15A-15" -> "1115J530-15A"
    """
    s = (name or "").strip().upper()
    s = re.sub(r"-\d+$", "", s)  # strip trailing "-15" delay if present
    return s


def load_motors_manual(meta_dir: Path) -> Dict[str, MotorStats]:
    motor_map: Dict[str, MotorStats] = {}
    p = meta_dir / "motors_manual.csv"
    if not p.exists():
        return motor_map

    df = pd.read_csv(p)
    for _, r in df.iterrows():
        m = normalize_motor_name(str(r.get("motor_name", "")))
        if not m:
            continue
        motor_map[m] = MotorStats(
            motor_name=m,
            manufacturer=str(r.get("manufacturer", "") or ""),
            avg_thrust_N=float(r.get("avg_thrust_N", float("nan"))),
            initial_thrust_N=float(r.get("initial_thrust_N", float("nan"))),
            max_thrust_N=float(r.get("max_thrust_N", float("nan"))),
            total_impulse_Ns=float(r.get("total_impulse_Ns", float("nan"))),
            burn_time_s=float(r.get("burn_time_s", float("nan"))),
        )
    return motor_map


# -----------------------------
# Rocket metadata (optional)
# -----------------------------
@dataclass
class RocketMeta:
    rocket_id: str
    dry_mass_kg: float = float("nan")
    diameter_m: float = float("nan")
    length_m: float = float("nan")
    cg_from_nose_m: float = float("nan")  # no-motor CG if that's what you have
    cp_from_nose_m: float = float("nan")  # optional
    stability_cal: float = float("nan")   # optional


def load_rockets_manual(meta_dir: Path) -> Dict[str, RocketMeta]:
    """
    Optional metadata file: rockets_manual.csv

    Expected columns (any subset is fine):
      rocket_id,dry_mass_kg,diameter_m,length_m,cg_from_nose_m,cp_from_nose_m,stability_cal
    """
    rockets: Dict[str, RocketMeta] = {}
    p = meta_dir / "rockets_manual.csv"
    if not p.exists():
        return rockets

    df = pd.read_csv(p)
    for _, r in df.iterrows():
        rid = str(r.get("rocket_id", "") or "").strip()
        if not rid:
            continue
        rockets[rid] = RocketMeta(
            rocket_id=rid,
            dry_mass_kg=float(r.get("dry_mass_kg", float("nan"))),
            diameter_m=float(r.get("diameter_m", float("nan"))),
            length_m=float(r.get("length_m", float("nan"))),
            cg_from_nose_m=float(r.get("cg_from_nose_m", float("nan"))),
            cp_from_nose_m=float(r.get("cp_from_nose_m", float("nan"))),
            stability_cal=float(r.get("stability_cal", float("nan"))),
        )
    return rockets


# -----------------------------
# Time series helpers
# -----------------------------
def smooth(y: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    w = window if window % 2 == 1 else window + 1
    k = np.ones(w) / w
    ypad = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def detect_liftoff(
    t: np.ndarray,
    alt_m: np.ndarray,
    alt_rise_m: float = 5.0,
    slope_mps: float = 2.0,
    consec: int = 3,
) -> float:
    """
    Liftoff heuristic:
      first time altitude > baseline + alt_rise_m AND d(alt)/dt > slope_mps
      for 'consec' consecutive samples.
    """
    if len(t) < 10:
        return float(t[0])

    t0 = t[0]
    base_mask = t <= (t0 + 5.0)
    if base_mask.sum() < 5:
        base_mask = np.arange(len(t)) < max(5, int(0.1 * len(t)))
    baseline = float(np.nanmedian(alt_m[base_mask]))

    alt_s = smooth(np.nan_to_num(alt_m, nan=baseline), window=9)
    dt = np.diff(t)
    da = np.diff(alt_s)
    slope = np.zeros_like(t, dtype=float)
    slope[1:] = np.divide(da, dt, out=np.zeros_like(da, dtype=float), where=dt > 0)

    cond = (alt_s >= baseline + alt_rise_m) & (slope >= slope_mps)
    run = 0
    for i, ok in enumerate(cond):
        run = run + 1 if ok else 0
        if run >= consec:
            return float(t[i - consec + 1])

    idx = np.where(alt_s >= baseline + alt_rise_m)[0]
    return float(t[idx[0]]) if len(idx) else float(t[0])


def interp_1d(df: pd.DataFrame, x: str, y: str, xq: float, allow_extrapolation: bool = False) -> float:
    d = df[[x, y]].dropna().sort_values(x)
    xs = d[x].to_numpy(float)
    ys = d[y].to_numpy(float)
    if len(xs) < 2:
        return float("nan")
    if xq < xs[0]:
        return float(ys[0]) if allow_extrapolation else float("nan")
    if xq > xs[-1]:
        return float(ys[-1]) if allow_extrapolation else float("nan")
    return float(np.interp(xq, xs, ys))


def delta_north_m(delta_lat_deg: float) -> float:
    R = 6371000.0
    return R * math.radians(delta_lat_deg)


def resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p

    candidates = [base_dir / p, PROJECT_ROOT / p, p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_dir / p


# -----------------------------
# Raven parsing
# -----------------------------
def read_raven_baro_agl_or_asl(path: Path) -> pd.DataFrame:
    """
    Extract Raven baro altitude series.
    Prefers [Altitude (Baro-Ft-AGL)] with matching time column.
    Falls back to [Altitude (Baro-Ft-ASL)].

    Returns: columns t_s, alt_m
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    def find_pair(label: str) -> Optional[Tuple[str, str]]:
        time_col = f"Time@[Altitude ({label})]"
        val_col = f"[Altitude ({label})]"
        if time_col in cols and val_col in cols:
            return time_col, val_col
        return None

    pair = find_pair("Baro-Ft-AGL")
    if pair is None:
        pair = find_pair("Baro-Ft-ASL")

    if pair is None:
        # fallback: any bracketed Baro altitude column
        val_candidates = [c for c in cols if "Altitude" in c and "Baro" in c and c.strip().startswith("[")]
        for vcol in val_candidates:
            inside = vcol.strip()[1:-1]
            tcol = f"Time@[{inside}]"
            if tcol in cols:
                pair = (tcol, vcol)
                break

    if pair is None:
        raise ValueError(f"Could not find Raven baro altitude columns in {path.name}")

    tcol, vcol = pair
    t = pd.to_numeric(df[tcol], errors="coerce")
    alt_ft = pd.to_numeric(df[vcol], errors="coerce")

    out = pd.DataFrame({"t_s": t, "alt_m": alt_ft * 0.3048}).dropna().sort_values("t_s").reset_index(drop=True)
    return out


def raven_apogee_features(
    raven_alt: pd.DataFrame,
    alt_rise_m: float,
    slope_mps: float,
) -> Dict[str, float]:
    t = raven_alt["t_s"].to_numpy(float)
    alt = raven_alt["alt_m"].to_numpy(float)
    if len(t) < 10:
        return {}

    liftoff = detect_liftoff(t, alt, alt_rise_m=alt_rise_m, slope_mps=slope_mps)
    post = raven_alt[raven_alt["t_s"] >= liftoff].copy()
    if len(post) < 5:
        return {}

    idx = int(post["alt_m"].idxmax())
    apogee_t = float(raven_alt.loc[idx, "t_s"])
    apogee_alt = float(raven_alt.loc[idx, "alt_m"])
    return {
        "raven_liftoff_t_s": liftoff,
        "raven_apogee_time_rel_s": apogee_t - liftoff,
        "raven_apogee_alt_m": apogee_alt,
    }


# -----------------------------
# Featherweight GPS parsing
# -----------------------------
def read_table_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def parse_featherweight_gps(path: Path, gps_alt_unit: str = "auto") -> pd.DataFrame:
    """
    Parses Featherweight GPS CSV/XLSX with columns like:
      UTCTIME, UNIXTIME, ALT, LAT, LON, #SATS, FIX, HORZV, VERTV, HEAD, ...
    or ground log:
      DATE, TIME, LAT, LON, ALT, FIX, ...

    Returns: columns t_s, lat_deg, lon_deg, alt_m (if ALT exists) plus optional quality columns.
    """
    df = read_table_any(path)
    # normalize column casing but keep originals
    colmap = {c.upper(): c for c in df.columns}

    def get(col: str) -> Optional[str]:
        return colmap.get(col.upper())

    # time -> t_s relative to first sample
    if get("UNIXTIME"):
        u = pd.to_numeric(df[get("UNIXTIME")], errors="coerce")
        u0 = u.dropna().iloc[0]
        t_s = u - u0
    elif get("DATE") and get("TIME"):
        dt = pd.to_datetime(df[get("DATE")].astype(str) + " " + df[get("TIME")].astype(str),
                            errors="coerce", utc=False)
        dt0 = dt.dropna().iloc[0]
        t_s = (dt - dt0).dt.total_seconds()
    elif get("UTCTIME"):
        dt = pd.to_datetime(df[get("UTCTIME")], errors="coerce", utc=True)
        dt0 = dt.dropna().iloc[0]
        t_s = (dt - dt0).dt.total_seconds()
    else:
        raise ValueError(f"Could not find a usable time column in {path.name}")

    # lat/lon
    lat_col = get("LAT") or get("LATITUDE")
    lon_col = get("LON") or get("LONGITUDE") or get("LNG")
    if not lat_col or not lon_col:
        raise ValueError(f"Could not find LAT/LON in {path.name}")

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    out = pd.DataFrame({"t_s": t_s, "lat_deg": lat, "lon_deg": lon})

    # altitude optional
    if get("ALT"):
        alt_raw = pd.to_numeric(df[get("ALT")], errors="coerce")
        out["alt_raw"] = alt_raw

        unit = gps_alt_unit.lower()
        if unit == "auto":
            med = float(np.nanmedian(alt_raw.to_numpy()))
            # Heuristic tuned for your samples (ALT ~ 2800 likely feet)
            if med > 100000:
                unit = "mm"
            elif med > 1500:
                unit = "ft"
            else:
                unit = "m"

        if unit == "ft":
            out["alt_m"] = out["alt_raw"] * 0.3048
        elif unit == "mm":
            out["alt_m"] = out["alt_raw"] / 1000.0
        else:
            out["alt_m"] = out["alt_raw"]

    # optional quality/kinematics
    if get("#SATS"):
        out["sats"] = pd.to_numeric(df[get("#SATS")], errors="coerce")
    elif get("SATS"):
        out["sats"] = pd.to_numeric(df[get("SATS")], errors="coerce")
    if get("FIX"):
        out["fix"] = pd.to_numeric(df[get("FIX")], errors="coerce")
    if get("HORZV"):
        out["horzv"] = pd.to_numeric(df[get("HORZV")], errors="coerce")
    if get("VERTV"):
        out["vertv"] = pd.to_numeric(df[get("VERTV")], errors="coerce")
    if get("HEAD"):
        out["head"] = pd.to_numeric(df[get("HEAD")], errors="coerce")

    # remove invalid rows and sort (your samples are sometimes out-of-order)
    out = out.dropna(subset=["t_s", "lat_deg", "lon_deg"]).sort_values("t_s").reset_index(drop=True)
    return out


def gps_pad_and_liftoff(gps: pd.DataFrame, alt_rise_m: float, slope_mps: float) -> Tuple[float, float, float]:
    """
    Returns: (gps_liftoff_t_s, pad_lat, pad_lon)
    """
    if "alt_m" in gps.columns and gps["alt_m"].notna().sum() >= 10:
        t = gps["t_s"].to_numpy(float)
        alt = gps["alt_m"].to_numpy(float)
        liftoff = detect_liftoff(t, alt, alt_rise_m=alt_rise_m, slope_mps=slope_mps)
        pre = gps[gps["t_s"] <= liftoff]
        if len(pre) < 5:
            pre = gps.head(10)
    else:
        liftoff = float(gps["t_s"].iloc[0])
        pre = gps.head(10)

    pad_lat = float(np.nanmedian(pre["lat_deg"].to_numpy()))
    pad_lon = float(np.nanmedian(pre["lon_deg"].to_numpy()))
    return liftoff, pad_lat, pad_lon


def gps_apogee_by_peak_alt(
    gps: pd.DataFrame,
    alt_rise_m: float,
    slope_mps: float,
    min_apogee_gain_m: float,
    min_descent_after_apogee_m: float,
) -> Dict[str, float]:
    """
    If GPS altitude exists, pick apogee index as max smoothed GPS altitude AFTER liftoff.
    """
    if "alt_m" not in gps.columns or gps["alt_m"].notna().sum() < 10:
        return {}

    t = gps["t_s"].to_numpy(float)
    alt = gps["alt_m"].to_numpy(float)
    liftoff = detect_liftoff(t, alt, alt_rise_m=alt_rise_m, slope_mps=slope_mps)

    post = gps[gps["t_s"] >= liftoff].copy()
    if len(post) < 10:
        return {}

    alt_s = smooth(post["alt_m"].to_numpy(float), window=9)
    post = post.iloc[: len(alt_s)].copy()
    post["alt_sm_m"] = alt_s

    pre_liftoff = gps[gps["t_s"] <= liftoff]["alt_m"].to_numpy(float)
    baseline = float(np.nanmedian(pre_liftoff))
    if not np.isfinite(baseline):
        baseline = float(np.nanmedian(post["alt_m"].to_numpy(float)))

    peak_pos = int(np.nanargmax(alt_s))
    peak_smoothed_alt = float(alt_s[peak_pos])
    if peak_smoothed_alt - baseline < min_apogee_gain_m:
        return {}
    if peak_pos < 2 or peak_pos >= len(alt_s) - 2:
        return {}

    descent_drop = peak_smoothed_alt - float(np.nanmin(alt_s[peak_pos + 1 :]))
    if descent_drop < min_descent_after_apogee_m:
        return {}

    idx = int(post["alt_sm_m"].idxmax())
    apogee_t = float(gps.loc[idx, "t_s"])
    return {
        "gps_liftoff_t_s": liftoff,
        "gps_apogee_time_rel_s": apogee_t - liftoff,
        "gps_apogee_alt_m": float(gps.loc[idx, "alt_m"]),
        "gps_apogee_lat_deg": float(gps.loc[idx, "lat_deg"]),
        "gps_apogee_lon_deg": float(gps.loc[idx, "lon_deg"]),
    }


# -----------------------------
# Trimmed TXT GPS parsing (optional)
# -----------------------------
_NUM_RX = re.compile(r"-?\d+(?:\.\d+)?")

def parse_trimmed_txt_gps(path: Path) -> pd.DataFrame:
    """
    Attempts to parse GPS lines in Trimmed.TXT.
    Many Trimmed files have NO GPS lines at all; in that case this will raise.

    Supports lines containing 'GPS' and tries both labeled and unlabeled numeric patterns.
    Interprets:
      time_us, lat_1e7, lon_1e7, alt_mm
    Returns: columns t_s, lat_deg, lon_deg, alt_m
    """
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "GPS" not in line:
            continue

        parts = [p.strip() for p in line.split(",")]

        def find_after(label: str) -> Optional[float]:
            for i, p in enumerate(parts[:-1]):
                if p.strip().lower() == label:
                    try:
                        return float(parts[i + 1])
                    except Exception:
                        return None
            return None

        time_us = find_after("t")
        lat_1e7 = find_after("lat")
        lon_1e7 = find_after("lon")
        alt_mm = find_after("alt")

        nums = _NUM_RX.findall(line)
        nums = [float(x) for x in nums] if nums else []

        if time_us is None and len(nums) >= 1:
            time_us = nums[0]

        if (lat_1e7 is None or lon_1e7 is None or alt_mm is None) and len(nums) >= 4:
            # common order: time_us, lat_1e7, lon_1e7, alt_mm
            cand = nums[:4]
            best = None
            for lat_i, lon_i, alt_i in [
                (cand[1], cand[2], cand[3]),
                (cand[2], cand[1], cand[3]),
            ]:
                lat = lat_i / 1e7
                lon = lon_i / 1e7
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    best = (lat_i, lon_i, alt_i)
                    break
            if best:
                lat_1e7, lon_1e7, alt_mm = best

        if time_us is None or lat_1e7 is None or lon_1e7 is None or alt_mm is None:
            continue

        rows.append({
            "t_s": time_us / 1e6,
            "lat_deg": lat_1e7 / 1e7,
            "lon_deg": lon_1e7 / 1e7,
            "alt_m": alt_mm / 1000.0,
        })

    if not rows:
        raise ValueError(f"No GPS records found in {path.name}")
    return pd.DataFrame(rows).dropna().sort_values("t_s").reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--out-dir", type=str, default="./out")
    ap.add_argument("--gps-alt-unit", type=str, default="auto", choices=["auto", "ft", "m", "mm"])
    ap.add_argument("--alt-rise-m", type=float, default=5.0)
    ap.add_argument("--slope-mps", type=float, default=2.0)
    ap.add_argument("--min-gps-apogee-gain-m", type=float, default=30.0)
    ap.add_argument("--min-gps-descent-after-apogee-m", type=float, default=10.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = out_dir / "cleaned_tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build_log.txt"
    if log_path.exists():
        log_path.unlink()

    alt_dir = data_dir / "altimeter"
    gps_dir = data_dir / "gps"
    meta_dir = data_dir / "metadata"

    log(f"Data dir: {data_dir.resolve()}", log_path)
    log(f"Out dir:  {out_dir.resolve()}", log_path)

    # Load manifest
    manifest_path = meta_dir / "manifest.csv"
    if not manifest_path.exists():
        log("ERROR: metadata/manifest.csv not found. Create it to pair files explicitly.", log_path)
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    required_cols = {"flight_id", "rocket_id", "motor_name", "altimeter_file", "gps_file"}
    if not required_cols.issubset(set(manifest.columns)):
        raise ValueError(f"manifest.csv must contain columns: {sorted(required_cols)}")

    # Load optional metadata
    motors_map = load_motors_manual(meta_dir)
    rockets_map = load_rockets_manual(meta_dir)

    # Default rocket constants if you don't provide rockets_manual.csv
    if "Madcow Adventurer" not in rockets_map:
        rockets_map["Madcow Adventurer"] = RocketMeta(
            rocket_id="Madcow Adventurer",
            dry_mass_kg=2.040,
            diameter_m=0.0575,
            length_m=1.69,
            cg_from_nose_m=36.25 * 0.0254,  # from your measurement text (no motor)
        )

    # Optional per-flight metadata
    flights_meta = pd.DataFrame()
    flights_meta_path = meta_dir / "flights_metadata.csv"
    if flights_meta_path.exists():
        flights_meta = pd.read_csv(flights_meta_path)

    rows = []
    failures = 0

    for _, r in manifest.iterrows():
        flight_id = str(r["flight_id"])
        rocket_id = str(r.get("rocket_id", "Madcow Adventurer") or "Madcow Adventurer")
        motor_raw = str(r.get("motor_name", "") or "")
        motor_name = normalize_motor_name(motor_raw)

        alt_file = resolve_input_path(str(r["altimeter_file"]), manifest_path.parent)
        gps_file_str = str(r.get("gps_file", "") or "").strip()
        gps_file = resolve_input_path(gps_file_str, manifest_path.parent) if gps_file_str else None

        log(f"---- {flight_id}", log_path)
        log(f"rocket_id={rocket_id} motor={motor_name}", log_path)
        log(f"altimeter={alt_file}", log_path)
        log(f"gps={gps_file if gps_file else '(none)'}", log_path)

        try:
            if not alt_file.exists():
                raise FileNotFoundError(f"Altimeter file not found: {alt_file}")

            # GPS: prefer external Featherweight file if provided
            gps_df = None
            gps_source = ""
            if gps_file and gps_file.exists():
                gps_df = parse_featherweight_gps(gps_file, gps_alt_unit=args.gps_alt_unit)
                gps_source = "external"
                (tracks_dir / f"{flight_id}_gps.csv").write_text(gps_df.to_csv(index=False), encoding="utf-8")

            # GPS: attempt embedded TXT GPS only if altimeter is TXT and external not given
            txtgps_df = None
            if (gps_df is None) and alt_file.suffix.lower() == ".txt":
                try:
                    txtgps_df = parse_trimmed_txt_gps(alt_file)
                    gps_df = txtgps_df
                    gps_source = "txt_embedded"
                    (tracks_dir / f"{flight_id}_txtgps.csv").write_text(txtgps_df.to_csv(index=False), encoding="utf-8")
                except Exception as e:
                    raise ValueError(
                        f"No usable GPS provided and no GPS lines parsed from TXT. "
                        f"Provide an on-board Featherweight GPS file in manifest. Details: {e}"
                    )

            if gps_df is None or len(gps_df) < 5:
                raise ValueError("No usable GPS data (need LAT/LON over time).")

            # Pad & GPS apogee
            gps_liftoff_t, pad_lat, pad_lon = gps_pad_and_liftoff(gps_df, args.alt_rise_m, args.slope_mps)
            gps_ap = gps_apogee_by_peak_alt(
                gps_df,
                args.alt_rise_m,
                args.slope_mps,
                args.min_gps_apogee_gain_m,
                args.min_gps_descent_after_apogee_m,
            )

            # Raven apogee (only if CSV)
            raven_alt_df = None
            raven_ap = {}
            if alt_file.suffix.lower() == ".csv":
                raven_alt_df = read_raven_baro_agl_or_asl(alt_file)
                raven_ap = raven_apogee_features(raven_alt_df, args.alt_rise_m, args.slope_mps)
                (tracks_dir / f"{flight_id}_raven_alt.csv").write_text(raven_alt_df.to_csv(index=False), encoding="utf-8")

            # Altimeter-timed apogee latitude (if Raven exists)
            alt_timed_lat = float("nan")
            alt_timed_lon = float("nan")
            alt_timed_time_rel = float("nan")
            alt_timed_alt_m = float("nan")
            if raven_ap and ("raven_apogee_time_rel_s" in raven_ap):
                gps_rel = gps_df.copy()
                gps_rel["t_rel_s"] = gps_rel["t_s"] - gps_liftoff_t

                alt_timed_time_rel = float(raven_ap["raven_apogee_time_rel_s"])
                alt_timed_alt_m = float(raven_ap.get("raven_apogee_alt_m", float("nan")))
                alt_timed_lat = interp_1d(gps_rel, "t_rel_s", "lat_deg", alt_timed_time_rel)
                alt_timed_lon = interp_1d(gps_rel, "t_rel_s", "lon_deg", alt_timed_time_rel)

            # Choose final target
            target_method = ""
            target_lat = float("nan")
            target_lon = float("nan")
            target_apogee_alt_m = float("nan")
            target_apogee_time_rel_s = float("nan")

            if gps_ap:
                # Prefer GPS-peak altitude method if GPS altitude exists
                target_method = "gps_peak_alt"
                target_lat = float(gps_ap["gps_apogee_lat_deg"])
                target_lon = float(gps_ap["gps_apogee_lon_deg"])
                target_apogee_alt_m = float(gps_ap["gps_apogee_alt_m"])
                target_apogee_time_rel_s = float(gps_ap["gps_apogee_time_rel_s"])
            elif not math.isnan(alt_timed_lat):
                target_method = "altimeter_timed"
                target_lat = alt_timed_lat
                target_lon = alt_timed_lon
                target_apogee_alt_m = alt_timed_alt_m
                target_apogee_time_rel_s = alt_timed_time_rel
            else:
                raise ValueError("Could not compute apogee target latitude (no GPS apogee and no altimeter-timed).")

            dlat = target_lat - pad_lat
            dnorth = delta_north_m(dlat)

            # Rocket constants
            rm = rockets_map.get(rocket_id, rockets_map["Madcow Adventurer"])
            ref_area_m2 = float(math.pi * (rm.diameter_m / 2.0) ** 2) if np.isfinite(rm.diameter_m) else float("nan")

            row = {
                "flight_id": flight_id,
                "rocket_id": rocket_id,
                "motor_name": motor_name,

                "pad_lat_deg": pad_lat,
                "pad_lon_deg": pad_lon,

                "target_apogee_lat_deg": target_lat,
                "target_apogee_lon_deg": target_lon,
                "target_apogee_alt_m": target_apogee_alt_m,
                "target_apogee_time_rel_s": target_apogee_time_rel_s,
                "target_method": target_method,

                "delta_lat_deg": dlat,
                "delta_north_m": dnorth,

                "gps_source": gps_source,
                "gps_samples": int(len(gps_df)),

                # rocket meta
                "dry_mass_kg": rm.dry_mass_kg,
                "diameter_m": rm.diameter_m,
                "length_m": rm.length_m,
                "ref_area_m2": ref_area_m2,
                "cg_from_nose_m": rm.cg_from_nose_m,
                "cp_from_nose_m": rm.cp_from_nose_m,
                "stability_cal": rm.stability_cal,
            }

            # attach Raven/GPS auxiliary summaries
            for k, v in gps_ap.items():
                row[k] = v
            for k, v in raven_ap.items():
                row[k] = v

            row["altimeter_timed_apogee_lat_deg"] = alt_timed_lat
            row["altimeter_timed_apogee_lon_deg"] = alt_timed_lon
            row["altimeter_timed_apogee_time_rel_s"] = alt_timed_time_rel
            row["altimeter_timed_apogee_alt_m"] = alt_timed_alt_m

            # motor stats if available
            ms = motors_map.get(motor_name)
            if ms:
                row.update({
                    "motor_manufacturer": ms.manufacturer,
                    "avg_thrust_N": ms.avg_thrust_N,
                    "initial_thrust_N": ms.initial_thrust_N,
                    "max_thrust_N": ms.max_thrust_N,
                    "total_impulse_Ns": ms.total_impulse_Ns,
                    "burn_time_s": ms.burn_time_s,
                })

            # optional per-flight metadata merge
            if not flights_meta.empty and "flight_id" in flights_meta.columns:
                mrow = flights_meta[flights_meta["flight_id"].astype(str) == flight_id]
                if len(mrow) == 1:
                    extra = mrow.iloc[0].to_dict()
                    for k, v in extra.items():
                        if k == "flight_id":
                            continue
                        if isinstance(v, (int, float, str)) and (pd.notna(v) if not isinstance(v, str) else True):
                            # avoid overwriting core columns
                            if k not in row:
                                row[k] = v

            rows.append(row)

        except Exception as e:
            failures += 1
            log(f"FAILED {flight_id}: {e}", log_path)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        log("ERROR: No flights processed successfully. See build_log.txt.", log_path)
        sys.exit(1)

    preferred = [
        "flight_id", "rocket_id", "motor_name", "motor_manufacturer",
        "pad_lat_deg", "pad_lon_deg",
        "target_apogee_lat_deg", "target_apogee_lon_deg", "target_apogee_alt_m", "target_apogee_time_rel_s",
        "target_method",
        "delta_lat_deg", "delta_north_m",
        "initial_thrust_N", "max_thrust_N", "avg_thrust_N", "burn_time_s", "total_impulse_Ns",
        "dry_mass_kg", "diameter_m", "length_m", "ref_area_m2", "cg_from_nose_m", "cp_from_nose_m", "stability_cal",
        "gps_source", "gps_samples",
        "gps_apogee_time_rel_s", "gps_apogee_alt_m", "gps_apogee_lat_deg", "gps_apogee_lon_deg",
        "raven_apogee_time_rel_s", "raven_apogee_alt_m",
        "altimeter_timed_apogee_lat_deg", "altimeter_timed_apogee_time_rel_s",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    out_path = out_dir / "flights_dataset.csv"
    out.to_csv(out_path, index=False)
    log(f"Wrote {out_path} rows={len(out)} failures={failures}", log_path)
    log("Done.", log_path)


if __name__ == "__main__":
    main()
