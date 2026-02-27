#!/usr/bin/env python3
"""
build_apogee_lat_dataset_dofpro.py

Build a one-row-per-flight dataset for the task:
    TARGET = latitude at apogee

It is tailored to the file formats you pasted:
- Raven .csv (multiple "Time@..." channels in one row)
- Featherweight GPS .csv (UTCTIME, UNIXTIME, ALT, LAT, LON, ...)
- "Trimmed.TXT" logs that can include "GPS," lines with lat/lon in 1E7 deg and altitude in mm

What it produces
----------------
out/
  flights_dataset.csv        # main supervised dataset (features + target)
  cleaned_tracks/
    <flight_id>_gps_clean.csv
    <flight_id>_alt_clean.csv   (if Raven altitude channel available)
    <flight_id>_txt_gps_clean.csv (if Trimmed.TXT GPS parsed)
  build_log.txt

Folder layout (recommended)
--------------------------
data/
  altimeter/   # Raven .csv and/or Trimmed.TXT
  gps/         # Featherweight GPS .csv (on-board preferred)
  motors/      # optional .eng motor files (RASP format) from ThrustCurve
  metadata/
    manifest.csv  # optional explicit pairing (highly recommended)

Manifest (optional but recommended)
----------------------------------
data/metadata/manifest.csv with columns:
  flight_id,rocket_id,motor_name,altimeter_file,gps_file
Example:
  20210417_Adventurer_J510W_1,Madcow Adventurer,J510W,20210417_Adventurer_J510W_1_Raven.csv,20210417_Adventurer_J510W_1_GPS.csv

Why two apogee definitions?
--------------------------
This script computes two candidate targets:
1) GPS-peak apogee: latitude at maximum GPS altitude after liftoff (robust, no clock sync)
2) Altimeter-timed apogee: Raven baro altitude peak time -> interpolate GPS latitude at that time (when Raven exists)

By default, the final target column is:
  target_apogee_lat_deg = gps_apogee_lat_deg  (if GPS altitude exists)
otherwise it falls back to altimeter-timed.

Dependencies:
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
# Basic helpers
# -----------------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def smooth(y: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    w = window if window % 2 == 1 else window + 1
    k = np.ones(w) / w
    ypad = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def detect_liftoff(t: np.ndarray, alt_m: np.ndarray,
                   alt_rise_m: float = 5.0, slope_mps: float = 2.0,
                   consec: int = 3) -> float:
    """
    Liftoff heuristic: first time alt exceeds baseline + alt_rise_m and slope > slope_mps
    for 'consec' consecutive samples.
    """
    if len(t) < 10:
        return float(t[0])

    # baseline from first ~5s
    t0 = t[0]
    base_mask = t <= (t0 + 5.0)
    if base_mask.sum() < 5:
        base_mask = np.arange(len(t)) < max(5, int(0.1 * len(t)))
    baseline = float(np.nanmedian(alt_m[base_mask]))

    alt_s = smooth(np.nan_to_num(alt_m, nan=baseline), window=9)
    dt = np.diff(t)
    da = np.diff(alt_s)
    slope = np.zeros_like(t)
    slope[1:] = np.where(dt > 0, da / dt, 0.0)

    cond = (alt_s >= baseline + alt_rise_m) & (slope >= slope_mps)

    run = 0
    for i, ok in enumerate(cond):
        run = run + 1 if ok else 0
        if run >= consec:
            return float(t[i - consec + 1])

    # fallback: first rise
    idx = np.where(alt_s >= baseline + alt_rise_m)[0]
    return float(t[idx[0]]) if len(idx) else float(t[0])


def earth_north_m(delta_lat_deg: float) -> float:
    R = 6371000.0
    return R * math.radians(delta_lat_deg)


# -----------------------------
# Raven (.csv) parsing: altitude channel
# -----------------------------
def read_raven_altitude(path: Path) -> pd.DataFrame:
    """
    Extract a single altitude time series from Raven CSV.

    Your header shows:
      Time@[Altitude (Baro-Ft-AGL)], [Altitude (Baro-Ft-AGL)]
      Time@[Altitude (Baro-Ft-ASL)], [Altitude (Baro-Ft-ASL)]

    We prefer Baro-Ft-AGL. If not present, use Baro-Ft-ASL.
    Returns df with columns: t_s, alt_m
    """
    df = pd.read_csv(path)
    df = norm_cols(df)

    cols = list(df.columns)

    def find_pair(label: str) -> Optional[Tuple[str, str, str]]:
        # time column often looks like "Time@[Altitude (Baro-Ft-AGL)]"
        # value column often looks like "[Altitude (Baro-Ft-AGL)]"
        time_rx = re.compile(rf"^Time@\[\s*Altitude\s*\({re.escape(label)}\)\s*\]$", re.IGNORECASE)
        val_rx = re.compile(rf"^\[\s*Altitude\s*\({re.escape(label)}\)\s*\]$", re.IGNORECASE)
        tcol = next((c for c in cols if time_rx.search(c)), None)
        vcol = next((c for c in cols if val_rx.search(c)), None)
        if tcol and vcol:
            return tcol, vcol, "ft"
        return None

    pair = find_pair("Baro-Ft-AGL")
    if pair is None:
        pair = find_pair("Baro-Ft-ASL")
    if pair is None:
        # fallback: try any column containing "Altitude (Baro"
        # and look for its corresponding Time@[...]
        alt_candidates = [c for c in cols if "Altitude (Baro" in c and c.strip().startswith("[")]
        for vcol in alt_candidates:
            inside = vcol.strip()[1:-1]  # remove brackets
            tcol = f"Time@[{inside}]"
            if tcol in cols:
                pair = (tcol, vcol, "ft")
                break

    if pair is None:
        raise ValueError(f"Could not find Raven baro altitude columns in {path.name}")

    tcol, vcol, unit = pair
    t = pd.to_numeric(df[tcol], errors="coerce")
    alt = pd.to_numeric(df[vcol], errors="coerce")

    out = pd.DataFrame({"t_s": t, "alt_raw": alt}).dropna().sort_values("t_s")

    # Raven label says Ft; convert to meters
    if unit == "ft":
        out["alt_m"] = out["alt_raw"] * 0.3048
    else:
        out["alt_m"] = out["alt_raw"]

    out = out[["t_s", "alt_m"]].reset_index(drop=True)
    return out


def raven_summary_features(df_alt: pd.DataFrame) -> Dict[str, float]:
    """
    Simple summary features from Raven altitude channel.
    """
    t = df_alt["t_s"].to_numpy(float)
    alt = df_alt["alt_m"].to_numpy(float)
    if len(t) < 5:
        return {}

    liftoff_t = detect_liftoff(t, alt)
    post = df_alt[df_alt["t_s"] >= liftoff_t]
    if len(post) < 5:
        return {}

    idx = int(post["alt_m"].idxmax())
    apogee_t = float(df_alt.loc[idx, "t_s"])
    apogee_alt = float(df_alt.loc[idx, "alt_m"])
    apogee_rel = apogee_t - liftoff_t

    return {
        "raven_liftoff_t_s": liftoff_t,
        "raven_apogee_time_rel_s": apogee_rel,
        "raven_apogee_alt_m": apogee_alt,
    }


# -----------------------------
# Featherweight GPS (.csv / .xlsx) parsing
# -----------------------------
def parse_featherweight_gps(path: Path, gps_alt_unit: str = "auto") -> pd.DataFrame:
    """
    Parse Featherweight GPS logs like:
      UTCTIME,UNIXTIME,ALT,LAT,LON,#SATS,FIX,HORZV,VERTV,HEAD,...

    Returns df with columns:
      t_s, lat_deg, lon_deg, alt_m (if available), horzv, vertv, head, fix, sats

    Time:
      - If UNIXTIME exists => seconds from first UNIXTIME
      - Else if DATE+TIME exists => seconds from first timestamp
      - Else if UTCTIME exists => seconds from first parsed UTCTIME
    """
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    df = norm_cols(df)

    # Standardize column lookup (case-insensitive)
    cols_upper = {c.upper(): c for c in df.columns}

    def col(name: str) -> Optional[str]:
        return cols_upper.get(name.upper())

    t_s = None

    if col("UNIXTIME") is not None:
        u = pd.to_numeric(df[col("UNIXTIME")], errors="coerce")
        u0 = u.dropna().iloc[0]
        t_s = u - u0
    elif col("DATE") is not None and col("TIME") is not None:
        dt = pd.to_datetime(df[col("DATE")].astype(str) + " " + df[col("TIME")].astype(str), errors="coerce", utc=False)
        dt0 = dt.dropna().iloc[0]
        t_s = (dt - dt0).dt.total_seconds()
    elif col("UTCTIME") is not None:
        dt = pd.to_datetime(df[col("UTCTIME")], errors="coerce", utc=True)
        dt0 = dt.dropna().iloc[0]
        t_s = (dt - dt0).dt.total_seconds()
    else:
        raise ValueError(f"Could not find a time column in {path.name}")

    lat = pd.to_numeric(df[col("LAT")] if col("LAT") else df[col("LATITUDE")], errors="coerce")
    lon = pd.to_numeric(df[col("LON")] if col("LON") else df[col("LONGITUDE")], errors="coerce")

    out = pd.DataFrame({"t_s": t_s, "lat_deg": lat, "lon_deg": lon})

    # Altitude (optional)
    if col("ALT") is not None:
        alt_raw = pd.to_numeric(df[col("ALT")], errors="coerce")
        out["alt_raw"] = alt_raw

        unit = gps_alt_unit.lower()
        if unit == "auto":
            med = float(np.nanmedian(alt_raw.to_numpy()))
            # Heuristic:
            # - huge values => mm
            # - mid values (typical pad ~2000-5000) usually feet for this dataset
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

    # Optional kinematics/quality
    if col("HORZV") is not None:
        out["horzv"] = pd.to_numeric(df[col("HORZV")], errors="coerce")
    if col("VERTV") is not None:
        out["vertv"] = pd.to_numeric(df[col("VERTV")], errors="coerce")
    if col("HEAD") is not None:
        out["head"] = pd.to_numeric(df[col("HEAD")], errors="coerce")
    if col("FIX") is not None:
        out["fix"] = pd.to_numeric(df[col("FIX")], errors="coerce")
    if col("#SATS") is not None:
        out["sats"] = pd.to_numeric(df[col("#SATS")], errors="coerce")
    elif col("SATS") is not None:
        out["sats"] = pd.to_numeric(df[col("SATS")], errors="coerce")

    out = out.dropna(subset=["t_s", "lat_deg", "lon_deg"]).sort_values("t_s").reset_index(drop=True)
    return out


def gps_liftoff_and_pad(gps: pd.DataFrame,
                        alt_rise_m: float = 5.0, slope_mps: float = 2.0) -> Tuple[float, float, float]:
    """
    Returns (liftoff_t_s, pad_lat, pad_lon)
    If GPS altitude not present, liftoff_t_s = first time.
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


def gps_apogee_by_peak_altitude(gps: pd.DataFrame,
                               alt_rise_m: float = 5.0, slope_mps: float = 2.0) -> Dict[str, float]:
    """
    If GPS altitude exists: find apogee by peak GPS altitude after liftoff.
    Returns apogee lat/lon, apogee time rel, and apogee altitude.
    """
    if "alt_m" not in gps.columns or gps["alt_m"].notna().sum() < 10:
        return {}

    t = gps["t_s"].to_numpy(float)
    alt = gps["alt_m"].to_numpy(float)

    liftoff = detect_liftoff(t, alt, alt_rise_m=alt_rise_m, slope_mps=slope_mps)
    post = gps[gps["t_s"] >= liftoff].copy()
    if len(post) < 5:
        return {}

    # Smooth altitude before finding max (reduces GPS noise spikes)
    alt_s = smooth(post["alt_m"].to_numpy(float), window=9)
    post = post.iloc[: len(alt_s)].copy()
    post["alt_sm_m"] = alt_s

    idx = int(post["alt_sm_m"].idxmax())
    apogee_t = float(gps.loc[idx, "t_s"])
    apogee_rel = apogee_t - liftoff
    return {
        "gps_liftoff_t_s": liftoff,
        "gps_apogee_time_rel_s": apogee_rel,
        "gps_apogee_alt_m": float(gps.loc[idx, "alt_m"]),
        "gps_apogee_lat_deg": float(gps.loc[idx, "lat_deg"]),
        "gps_apogee_lon_deg": float(gps.loc[idx, "lon_deg"]),
    }


def interp(df: pd.DataFrame, x: str, y: str, xq: float) -> float:
    d = df[[x, y]].dropna().sort_values(x)
    xs = d[x].to_numpy(float)
    ys = d[y].to_numpy(float)
    if len(xs) < 2:
        return float("nan")
    return float(np.interp(xq, xs, ys))


# -----------------------------
# Trimmed .TXT parsing (GPS lines)
# -----------------------------
_NUM_RX = re.compile(r"-?\d+(?:\.\d+)?")

def parse_trimmed_txt_gps(path: Path) -> pd.DataFrame:
    """
    Parse GPS lines inside Trimmed.TXT logs.

    The header hints:
      "GPS data in us, 1E7 deg, mm, #, ISO Date Time"

    Formats vary, so we parse numbers on lines that start with 'GPS'.
    We try to interpret the first few numeric fields as:
      time_us, lat_1e7, lon_1e7, alt_mm, sats (optional)

    Returns df columns:
      t_s, lat_deg, lon_deg, alt_m
    """
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("GPS"):
            continue

        # extract all numbers in the line
        nums = _NUM_RX.findall(line)
        if len(nums) < 4:
            continue

        # Heuristic mapping:
        # 0: time_us
        # 1: lat_1e7
        # 2: lon_1e7
        # 3: alt_mm
        try:
            time_us = float(nums[0])
            lat_1e7 = float(nums[1])
            lon_1e7 = float(nums[2])
            alt_mm = float(nums[3])
        except Exception:
            continue

        rows.append({
            "t_s": time_us / 1e6,
            "lat_deg": lat_1e7 / 1e7,
            "lon_deg": lon_1e7 / 1e7,
            "alt_m": alt_mm / 1000.0,
        })

    if not rows:
        raise ValueError(f"No GPS lines parsed from {path.name}")

    df = pd.DataFrame(rows).dropna().sort_values("t_s").reset_index(drop=True)
    return df


# -----------------------------
# Optional: motor .eng parsing (RASP)
# -----------------------------
@dataclass
class MotorStats:
    motor_name: str
    burn_time_s: float
    total_impulse_Ns: float
    avg_thrust_N: float
    max_thrust_N: float
    initial_thrust_N: float


def parse_rasp_eng(path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Parse FIRST motor block of a RASP .eng file.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith(";")]
    if not lines:
        raise ValueError(f"Empty .eng: {path.name}")

    header = lines[0].split()
    motor_name = header[0]

    t_list, f_list = [], []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 7 and re.match(r"^[A-Za-z]\w*$", parts[0]):
            break
        if len(parts) < 2:
            continue
        try:
            t_list.append(float(parts[0]))
            f_list.append(float(parts[1]))
        except ValueError:
            continue

    if len(t_list) < 3:
        raise ValueError(f"Could not parse thrust points from {path.name}")

    t = np.array(t_list, float)
    f = np.array(f_list, float)
    return motor_name, t, f


def compute_motor_stats(motor_name: str, t: np.ndarray, thrust: np.ndarray) -> MotorStats:
    idx = np.argsort(t)
    t = t[idx]
    thrust = thrust[idx]

    burn_time = float(t[-1])
    total_impulse = float(np.trapz(thrust, t))
    avg_thrust = total_impulse / burn_time if burn_time > 0 else float("nan")
    max_thrust = float(np.nanmax(thrust))

    # ThrustCurve definition: initial thrust = avg thrust over first 0.5s
    t_end = min(0.5, burn_time)
    if t_end <= 0:
        init = float("nan")
    else:
        # ensure we have endpoints
        grid_t = np.unique(np.concatenate([t[t <= t_end], np.array([0.0, t_end])]))
        grid_f = np.interp(grid_t, t, thrust)
        init = float(np.trapz(grid_f, grid_t) / t_end)

    return MotorStats(
        motor_name=motor_name,
        burn_time_s=burn_time,
        total_impulse_Ns=total_impulse,
        avg_thrust_N=avg_thrust,
        max_thrust_N=max_thrust,
        initial_thrust_N=init,
    )


# -----------------------------
# Pairing logic
# -----------------------------
MOTOR_TOKEN_RX = re.compile(r"(?:^|[_\-\s])(J\d{3,4}[A-Z]*)(?:[_\-\s]|$)", re.IGNORECASE)
DATE_TOKEN_RX = re.compile(r"(20\d{6})")

def infer_tokens(stem: str) -> Dict[str, Optional[str]]:
    motor = None
    date = None
    m = MOTOR_TOKEN_RX.search(stem)
    if m:
        motor = m.group(1).upper()
    d = DATE_TOKEN_RX.search(stem)
    if d:
        date = d.group(1)
    return {"motor": motor, "date": date}


def auto_manifest(alt_files: List[Path], gps_files: List[Path]) -> pd.DataFrame:
    """
    Try to pair Raven/Trimmed with GPS by date+motor tokens in filenames.
    """
    gps_index = []
    for g in gps_files:
        tok = infer_tokens(g.stem)
        gps_index.append({"gps_file": str(g), "date": tok["date"], "motor": tok["motor"], "name": g.name})
    gps_df = pd.DataFrame(gps_index)

    rows = []
    for a in alt_files:
        tok = infer_tokens(a.stem)
        date = tok["date"] or "unk"
        motor = tok["motor"] or ""
        # For Trimmed.TXT, gps might be inside; allow gps_file empty if not found
        candidates = gps_df.copy()
        if tok["date"]:
            candidates = candidates[candidates["date"] == tok["date"]]
        if tok["motor"]:
            candidates = candidates[candidates["motor"] == tok["motor"]]

        gps_file = ""
        if len(candidates) >= 1:
            # pick best overlap
            def score(name: str) -> int:
                s1 = set(re.split(r"[_\-\s\.]+", a.name.lower()))
                s2 = set(re.split(r"[_\-\s\.]+", name.lower()))
                return len(s1.intersection(s2))
            candidates = candidates.copy()
            candidates["score"] = candidates["name"].apply(score)
            gps_file = str(candidates.sort_values("score", ascending=False).iloc[0]["gps_file"])

        flight_id = f"{date}_{motor}_{a.stem}"
        rows.append({
            "flight_id": flight_id,
            "rocket_id": "Madcow Adventurer",
            "motor_name": motor,
            "altimeter_file": str(a),
            "gps_file": gps_file
        })

    return pd.DataFrame(rows)


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
    motors_dir = data_dir / "motors"
    meta_dir = data_dir / "metadata"

    log(f"Data dir: {data_dir.resolve()}", log_path)

    # Default rocket constants (from your pasted measurements)
    rocket_defaults = {
        "Madcow Adventurer": {
            "dry_mass_kg": 2.040,                 # 2040 g
            "diameter_m": 0.0575,                 # 57.5 mm
            "cg_from_nose_m": 36.25 * 0.0254,     # inches -> m
        }
    }

    # Load motor stats from .eng if provided
    motor_stats = {}
    if motors_dir.exists():
        for eng in motors_dir.glob("*.eng"):
            try:
                mname, t, f = parse_rasp_eng(eng)
                st = compute_motor_stats(mname, t, f)
                motor_stats[mname.upper()] = st
            except Exception as e:
                log(f"WARNING motor parse failed {eng.name}: {e}", log_path)

    # Collect files
    alt_files = sorted([p for p in alt_dir.glob("*") if p.suffix.lower() in [".csv", ".txt"]])
    gps_files = sorted([p for p in gps_dir.glob("*") if p.suffix.lower() in [".csv", ".xlsx", ".xls"]])

    if not alt_files:
        log("ERROR: no files in data/altimeter/", log_path)
        sys.exit(1)

    # Manifest
    manifest_path = meta_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        need = {"flight_id", "rocket_id", "motor_name", "altimeter_file", "gps_file"}
        if not need.issubset(set(manifest.columns)):
            raise ValueError(f"manifest.csv must have columns: {sorted(need)}")
        log(f"Loaded manifest with {len(manifest)} flights.", log_path)
    else:
        manifest = auto_manifest(alt_files, gps_files)
        log(f"Auto-manifest built with {len(manifest)} flights. (You can create metadata/manifest.csv to override.)", log_path)
        manifest.to_csv(out_dir / "manifest_TEMPLATE.csv", index=False)

    rows = []
    failures = 0

    for _, m in manifest.iterrows():
        flight_id = str(m["flight_id"])
        rocket_id = str(m.get("rocket_id", "Madcow Adventurer"))
        motor_name = str(m.get("motor_name", "") or "").upper()

        alt_path = Path(str(m["altimeter_file"]))
        gps_file_str = str(m.get("gps_file", "") or "")
        gps_path = Path(gps_file_str) if gps_file_str else None

        log(f"---- Flight {flight_id}", log_path)
        log(f"altimeter_file={alt_path.name}", log_path)
        log(f"gps_file={gps_path.name if gps_path else '(none)'}", log_path)

        try:
            # GPS source A: external Featherweight file
            gps_df = None
            if gps_path and gps_path.exists():
                gps_df = parse_featherweight_gps(gps_path, gps_alt_unit=args.gps_alt_unit)
                gps_df.to_csv(tracks_dir / f"{flight_id}_gps_clean.csv", index=False)

            # GPS source B: Trimmed.TXT embedded GPS
            txt_gps_df = None
            if alt_path.suffix.lower() == ".txt":
                try:
                    txt_gps_df = parse_trimmed_txt_gps(alt_path)
                    txt_gps_df.to_csv(tracks_dir / f"{flight_id}_txt_gps_clean.csv", index=False)
                except Exception as e:
                    log(f"INFO: no embedded TXT GPS parsed: {e}", log_path)

            # Choose the "best" GPS track (more rows, and must have lat/lon)
            gps_candidates = []
            if gps_df is not None and len(gps_df) >= 5:
                gps_candidates.append(("external", gps_df))
            if txt_gps_df is not None and len(txt_gps_df) >= 5:
                gps_candidates.append(("embedded", txt_gps_df))

            if not gps_candidates:
                raise ValueError("No usable GPS track found (need lat/lon over time).")

            gps_source, gps = max(gps_candidates, key=lambda x: len(x[1]))
            log(f"Using GPS source: {gps_source} ({len(gps)} samples)", log_path)

            # Pad location and GPS apogee (peak altitude if available)
            gps_liftoff_t, pad_lat, pad_lon = gps_liftoff_and_pad(gps, args.alt_rise_m, args.slope_mps)
            gps_ap = gps_apogee_by_peak_altitude(gps, args.alt_rise_m, args.slope_mps)

            # Altimeter (Raven) altitude series (optional; used for altimeter-timed apogee)
            raven_df = None
            raven_ap = {}
            if alt_path.suffix.lower() == ".csv":
                raven_df = read_raven_altitude(alt_path)
                raven_df.to_csv(tracks_dir / f"{flight_id}_alt_clean.csv", index=False)
                raven_ap = raven_summary_features(raven_df)

            # Compute "altimeter-timed" apogee latitude if possible
            alt_timed_apogee_lat = float("nan")
            alt_timed_apogee_lon = float("nan")
            if raven_ap and ("raven_apogee_time_rel_s" in raven_ap):
                # Create GPS time relative to its liftoff
                gps_rel = gps.copy()
                gps_rel["t_rel_s"] = gps_rel["t_s"] - gps_liftoff_t
                t_rel = float(raven_ap["raven_apogee_time_rel_s"])
                alt_timed_apogee_lat = interp(gps_rel, "t_rel_s", "lat_deg", t_rel)
                alt_timed_apogee_lon = interp(gps_rel, "t_rel_s", "lon_deg", t_rel)

            # Decide final target:
            # Prefer GPS-peak apogee latitude when GPS altitude exists; else use altimeter-timed.
            target_lat = float("nan")
            target_method = ""
            if gps_ap.get("gps_apogee_lat_deg") is not None and not math.isnan(float(gps_ap.get("gps_apogee_lat_deg", float("nan")))):
                target_lat = float(gps_ap["gps_apogee_lat_deg"])
                target_method = "gps_peak_alt"
            elif not math.isnan(alt_timed_apogee_lat):
                target_lat = float(alt_timed_apogee_lat)
                target_method = "altimeter_timed"
            else:
                raise ValueError("Could not compute apogee latitude target (no GPS altitude peak and no altimeter-timed interpolation).")

            # Rocket constants
            rc = rocket_defaults.get(rocket_id, rocket_defaults["Madcow Adventurer"])
            dry_mass_kg = float(rc.get("dry_mass_kg", np.nan))
            diameter_m = float(rc.get("diameter_m", np.nan))
            ref_area_m2 = float(math.pi * (diameter_m / 2.0) ** 2) if np.isfinite(diameter_m) else np.nan
            cg_from_nose_m = float(rc.get("cg_from_nose_m", np.nan))

            # Motor stats from .eng (optional)
            ms = motor_stats.get(motor_name, None)

            # Derived deltas
            delta_lat = target_lat - pad_lat
            delta_north = earth_north_m(delta_lat)

            row = {
                "flight_id": flight_id,
                "rocket_id": rocket_id,
                "motor_name": motor_name,

                "pad_lat_deg": pad_lat,
                "pad_lon_deg": pad_lon,

                # Final target + metadata
                "target_apogee_lat_deg": target_lat,
                "target_method": target_method,
                "delta_lat_deg": delta_lat,
                "delta_north_m": delta_north,

                # GPS summaries
                "gps_source": gps_source,
                "gps_samples": int(len(gps)),
                "gps_liftoff_t_s": float(gps_liftoff_t),
            }

            # Add GPS apogee-by-peak-altitude fields (if present)
            for k, v in gps_ap.items():
                row[k] = v

            # Add Raven/altimeter summaries (if present)
            for k, v in raven_ap.items():
                row[k] = v
            row["altimeter_timed_apogee_lat_deg"] = alt_timed_apogee_lat
            row["altimeter_timed_apogee_lon_deg"] = alt_timed_apogee_lon

            # Rocket constants
            row.update({
                "dry_mass_kg": dry_mass_kg,
                "diameter_m": diameter_m,
                "ref_area_m2": ref_area_m2,
                "cg_from_nose_m": cg_from_nose_m,
            })

            # Motor constants (if provided)
            if ms is not None:
                row.update({
                    "initial_thrust_N": ms.initial_thrust_N,
                    "max_thrust_N": ms.max_thrust_N,
                    "avg_thrust_N": ms.avg_thrust_N,
                    "burn_time_s": ms.burn_time_s,
                    "total_impulse_Ns": ms.total_impulse_Ns,
                })

            rows.append(row)

        except Exception as e:
            failures += 1
            log(f"FAILED {flight_id}: {e}", log_path)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        log("ERROR: No flights processed successfully. See build_log.txt and consider using manifest.csv.", log_path)
        sys.exit(1)

    # Arrange columns
    preferred = [
        "flight_id", "rocket_id", "motor_name",
        "pad_lat_deg", "pad_lon_deg",
        "target_apogee_lat_deg", "target_method",
        "delta_lat_deg", "delta_north_m",
        "gps_source", "gps_samples",
        "gps_apogee_time_rel_s", "gps_apogee_alt_m", "gps_apogee_lat_deg", "gps_apogee_lon_deg",
        "raven_apogee_time_rel_s", "raven_apogee_alt_m",
        "altimeter_timed_apogee_lat_deg",
        "initial_thrust_N", "max_thrust_N", "avg_thrust_N", "burn_time_s", "total_impulse_Ns",
        "dry_mass_kg", "diameter_m", "ref_area_m2", "cg_from_nose_m",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    out_path = out_dir / "flights_dataset.csv"
    out.to_csv(out_path, index=False)
    log(f"Wrote {out_path} rows={len(out)} failures={failures}", log_path)
    log("Done.", log_path)


if __name__ == "__main__":
    main()