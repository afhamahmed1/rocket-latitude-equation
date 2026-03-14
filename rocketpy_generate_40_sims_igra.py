#!/usr/bin/env python3
"""
rocketpy_generate_40_sims_igra.py

Generates 40 RocketPy simulations using:
- Rocket parameters extracted from ONE OpenRocket export CSV (mass, inertia, ref area, pad lat/lon)
- OpenRocket-derived drag curve files (Mach,Cd) for power_off_drag and power_on_drag
- Multi-level wind profiles parsed from IGRA2 (radiosonde) text
- Motor definitions from motors_manual.csv (+ optional thrust curve CSV files)

Outputs:
- out/sim_manifest.csv  (inputs per run)
- out/sim_results.csv   (lat_apogee, lon_apogee, apogee altitude/time, drift)

Docs references:
- Rocket constructor expects mass w/o motor, inertia, drag curves as Mach,Cd CSV paths. :contentReference[oaicite:4]{index=4}
- Custom atmosphere wind_u and wind_v can be set as altitude-value lists. :contentReference[oaicite:5]{index=5}
- Flight uses inclination (0=horizontal, 90=vertical) and heading (deg from North). :contentReference[oaicite:6]{index=6}
- Thrust CSV must be 2 cols: time(s), thrust(N). :contentReference[oaicite:7]{index=7}
- GenericMotor init signature (needs burn_time, chamber geometry, masses, nozzle radius). :contentReference[oaicite:8]{index=8}
"""

from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


# -------------------------
# IGRA2 parsing
# -------------------------
HEADER_RX = re.compile(
    r"^#(?P<station>\S+)\s+(?P<yyyy>\d{4})\s+(?P<mm>\d{2})\s+(?P<dd>\d{2})\s+(?P<hh>\d{2}).*?(?P<lat>-?\d+)\s+(?P<lon>-?\d+)\s*$"
)
# token like "18905A" (GPH with quality flag)
GPH_TOKEN_RX = re.compile(r"^-?\d+A$")


@dataclass
class WindProfile:
    profile_id: str
    station: str
    datetime_utc: str
    # altitude ASL (m) and wind components (m/s)
    z_m: np.ndarray
    wind_u: np.ndarray  # east +
    wind_v: np.ndarray  # north +
    # for convenience
    surface_wdir_from_deg: float
    surface_wspd_mps: float


def wind_uv_from_wdir_wspd(wdir_from_deg: float, wspd_mps: float) -> Tuple[float, float]:
    """
    IGRA2:
      WDIR = wind direction (deg from North). This is meteorological: direction the wind is COMING FROM.
      WSPD = m/s (in our script, already converted from tenths).

    Convert to direction wind is GOING TO by adding 180°, then:
      u (east)  = V * sin(theta_to)
      v (north) = V * cos(theta_to)
    """
    theta_to = math.radians((wdir_from_deg + 180.0) % 360.0)
    u = wspd_mps * math.sin(theta_to)
    v = wspd_mps * math.cos(theta_to)
    return u, v


def resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p

    candidates = [base_dir / p, PROJECT_ROOT / p, p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_dir / p


def parse_igra2_profiles(path: Path, max_profiles: int = 20) -> List[WindProfile]:
    """
    Parses an IGRA2-like text file containing multiple soundings separated by header lines starting with '#'.

    We extract only rows where:
      - WDIR and WSPD are present (not -9999)
      - GPH exists as a standalone token like '18905A' or '98A'
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    profiles: List[WindProfile] = []
    cur_station = None
    cur_dt = None
    cur_lat = None
    cur_lon = None
    z_list: List[float] = []
    u_list: List[float] = []
    v_list: List[float] = []
    surf_wdir = float("nan")
    surf_wspd = float("nan")
    lowest_z = float("inf")

    def flush():
        nonlocal z_list, u_list, v_list, cur_station, cur_dt, cur_lat, cur_lon, surf_wdir, surf_wspd, lowest_z
        if cur_station and cur_dt and len(z_list) >= 3:
            # sort by altitude, remove duplicates
            arr = np.array(list(zip(z_list, u_list, v_list)), dtype=float)
            arr = arr[np.isfinite(arr).all(axis=1)]
            arr = arr[arr[:, 0].argsort()]
            # de-dup by altitude
            _, idx = np.unique(arr[:, 0], return_index=True)
            arr = arr[np.sort(idx)]
            if len(arr) >= 3:
                profiles.append(
                    WindProfile(
                        profile_id=f"{cur_station}_{cur_dt.replace(':','').replace('-','')}",
                        station=cur_station,
                        datetime_utc=cur_dt,
                        z_m=arr[:, 0],
                        wind_u=arr[:, 1],
                        wind_v=arr[:, 2],
                        surface_wdir_from_deg=float(surf_wdir) if np.isfinite(surf_wdir) else float("nan"),
                        surface_wspd_mps=float(surf_wspd) if np.isfinite(surf_wspd) else float("nan"),
                    )
                )
        z_list, u_list, v_list = [], [], []
        surf_wdir, surf_wspd = float("nan"), float("nan")
        lowest_z = float("inf")

    for line in text:
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            # new header => flush old profile
            flush()
            m = HEADER_RX.match(line)
            if m:
                cur_station = m.group("station")
                cur_dt = f"{m.group('yyyy')}-{m.group('mm')}-{m.group('dd')}T{m.group('hh')}:00:00Z"
                # lat/lon are usually in 1e4 degrees in IGRA; user snippet looks like that
                # Example: 454170 -> 45.4170
                cur_lat = int(m.group("lat")) / 10000.0
                cur_lon = int(m.group("lon")) / 10000.0
            else:
                # header not matching; still start a new block
                cur_station = line[1:].split()[0]
                cur_dt = "unknown"
                cur_lat = None
                cur_lon = None
            if len(profiles) >= max_profiles:
                break
            continue

        # data record
        toks = line.split()
        if len(toks) < 6:
            continue

        # WDIR & WSPD are typically near the end in your sample:
        # "... 110 120 62" -> WDIR=110, WSPD=120 (tenths)
        try:
            wdir = int(toks[-3])
            wspd_tenths = int(toks[-2])
        except Exception:
            continue
        if wdir == -9999 or wspd_tenths == -9999:
            continue
        wspd = wspd_tenths / 10.0

        # Find GPH token like "98A" or "18905A"
        gph_m = None
        for tok in toks:
            if GPH_TOKEN_RX.match(tok):
                val = int(tok[:-1])
                # plausible geopotential heights (m)
                if 0 <= val <= 50000:
                    gph_m = float(val)
                    break
        if gph_m is None:
            continue

        u, v = wind_uv_from_wdir_wspd(wdir, wspd)
        if gph_m < lowest_z:
            lowest_z = gph_m
            surf_wdir = float(wdir)
            surf_wspd = float(wspd)
        z_list.append(gph_m)
        u_list.append(u)
        v_list.append(v)

    flush()
    return profiles


# -------------------------
# OpenRocket rocket template extraction
# -------------------------
def read_openrocket_template(path: Path) -> Dict[str, float]:
    """
    Reads ONE OpenRocket export data CSV (time series) and extracts:
      - pad lat/lon
      - rocket mass without motor (kg) = mass0 - motor_mass0
      - inertia guess from 'Longitudinal moment of inertia' and 'Rotational moment of inertia'
      - reference area -> radius
    """
    df = pd.read_csv(path, comment="#", engine="python")
    cols = list(df.columns)

    def find(must: List[str]) -> str:
        low = {c: c.lower() for c in cols}
        for c in cols:
            ok = True
            for token in must:
                if token not in low[c]:
                    ok = False
                    break
            if ok:
                return c
        raise KeyError(f"Missing column tokens={must}")

    time_col = find(["time"])
    lat_col = find(["latitude"])
    lon_col = find(["longitude"])
    mass_col = find(["mass (g)"])
    mmass_col = find(["motor mass (g)"])
    area_col = find(["reference area"])
    I_long_col = find(["longitudinal moment of inertia"])
    I_rot_col = find(["rotational moment of inertia"])

    # first row
    lat0 = float(pd.to_numeric(df[lat_col], errors="coerce").iloc[0])
    lon0 = float(pd.to_numeric(df[lon_col], errors="coerce").iloc[0])
    mass0_g = float(pd.to_numeric(df[mass_col], errors="coerce").iloc[0])
    mmass0_g = float(pd.to_numeric(df[mmass_col], errors="coerce").iloc[0])
    area_cm2 = float(pd.to_numeric(df[area_col], errors="coerce").iloc[0])
    I_long = float(pd.to_numeric(df[I_long_col], errors="coerce").iloc[0])
    I_rot = float(pd.to_numeric(df[I_rot_col], errors="coerce").iloc[0])

    mass_wo_motor_kg = (mass0_g - mmass0_g) / 1000.0
    area_m2 = area_cm2 / 1e4
    radius_m = math.sqrt(area_m2 / math.pi)

    # Heuristic mapping:
    # - OpenRocket "longitudinal" looks like inertia about perpendicular axes (big)
    # - OpenRocket "rotational" looks like inertia about symmetry axis (small)
    I11 = I_long
    I22 = I_long
    I33 = I_rot

    return {
        "pad_lat_deg": lat0,
        "pad_lon_deg": lon0,
        "mass_wo_motor_kg": mass_wo_motor_kg,
        "radius_m": radius_m,
        "I11": I11,
        "I22": I22,
        "I33": I33,
        "ref_area_m2": area_m2,
    }


# -------------------------
# Motor loading / fallback thrust curve generation
# -------------------------
@dataclass
class MotorDef:
    motor_id: str
    thrust_curve_file: Optional[Path]
    diameter_m: float
    length_m: float
    total_mass_kg: float
    prop_mass_kg: float
    burn_time_s: float


def load_motors_manual(path: Path) -> List[MotorDef]:
    df = pd.read_csv(path)
    motors: List[MotorDef] = []
    for _, r in df.iterrows():
        thrust_curve_file = None
        if pd.notna(r.get("thrust_curve_file")) and str(r["thrust_curve_file"]).strip():
            thrust_curve_file = resolve_input_path(str(r["thrust_curve_file"]).strip(), path.parent)
        motors.append(
            MotorDef(
                motor_id=str(r["motor_id"]),
                thrust_curve_file=thrust_curve_file,
                diameter_m=float(r["diameter_m"]),
                length_m=float(r["length_m"]),
                total_mass_kg=float(r["total_mass_kg"]),
                prop_mass_kg=float(r["prop_mass_kg"]),
                burn_time_s=float(r["burn_time_s"]),
            )
        )
    return motors


def write_trapezoid_thrust_curve(out_path: Path, burn_time: float, total_impulse: float,
                                 peak_thrust: float, initial_thrust: float) -> None:
    """
    Generates a simple trapezoidal thrust curve (time, thrust) and scales it to match total impulse.
    This is a fallback if you don't have a real curve file.

    Curve shape:
      - ramp to peak at 10% burn
      - hold near peak to 70% burn
      - decay to 0 at burn end
    """
    t1 = 0.10 * burn_time
    t2 = 0.70 * burn_time
    t3 = burn_time

    # initial -> peak -> peak -> 0
    t = np.array([0.0, t1, t2, t3], dtype=float)
    f = np.array([initial_thrust, peak_thrust, peak_thrust, 0.0], dtype=float)

    imp = float(np.trapz(f, t))
    if imp > 0:
        scale = total_impulse / imp
        f *= scale

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": t, "F": f}).to_csv(out_path, index=False, header=False)


# -------------------------
# RocketPy simulation
# -------------------------
def simulate(
    pad_lat: float,
    pad_lon: float,
    elevation_m: float,
    drag_off: Path,
    drag_on: Path,
    rocket_mass_kg: float,
    radius_m: float,
    inertia: Tuple[float, float, float],
    motor: MotorDef,
    wind_profile: WindProfile,
    rail_length_m: float,
    inclination_deg: float,
    heading_deg: float,
    seed: int,
    allow_generated_thrust_curves: bool,
    generated_curves_dir: Path,
) -> Dict[str, float]:
    from rocketpy import Environment, Rocket
    from rocketpy.motors import GenericMotor
    from rocketpy import Flight

    # Build environment with multi-level winds from IGRA profile
    env = Environment(latitude=pad_lat, longitude=pad_lon, elevation=elevation_m)
    wind_u = list(zip(wind_profile.z_m.tolist(), wind_profile.wind_u.tolist()))
    wind_v = list(zip(wind_profile.z_m.tolist(), wind_profile.wind_v.tolist()))

    # Use ISA pressure/temperature by leaving them as None; define wind profiles explicitly. :contentReference[oaicite:9]{index=9}
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=None,
        temperature=None,
        wind_u=wind_u,
        wind_v=wind_v,
    )

    # Rocket (mass without motor), inertia about COM w/o motor, drag curves. :contentReference[oaicite:10]{index=10}
    rocket = Rocket(
        radius=radius_m,
        mass=rocket_mass_kg,
        inertia=inertia,
        power_off_drag=str(drag_off),
        power_on_drag=str(drag_on),
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    # Motor: prefer real thrust curve file; otherwise generate fallback trapezoid
    thrust_source = None
    if motor.thrust_curve_file and motor.thrust_curve_file.exists():
        thrust_source = str(motor.thrust_curve_file)
    else:
        if not allow_generated_thrust_curves:
            raise FileNotFoundError(
                f"Missing thrust curve for motor {motor.motor_id}. "
                "Provide a valid curve file or re-run with --allow-generated-thrust-curves."
            )
        # fallback curve: estimate peak and initial thrust ~ avg thrust
        # total impulse estimate: avg_thrust * burn_time
        avg_thrust = 500.0  # generic fallback if you didn't provide a curve
        total_impulse = avg_thrust * motor.burn_time_s
        peak_thrust = 1.5 * avg_thrust
        initial_thrust = 1.2 * avg_thrust
        fallback_curve = generated_curves_dir / f"{motor.motor_id}.csv"
        write_trapezoid_thrust_curve(fallback_curve, motor.burn_time_s, total_impulse, peak_thrust, initial_thrust)
        thrust_source = str(fallback_curve)

    # GenericMotor required params (rough geometry + masses). :contentReference[oaicite:11]{index=11}
    chamber_radius = motor.diameter_m / 2.0
    chamber_height = motor.length_m
    chamber_position = 0.0
    nozzle_radius = 0.85 * chamber_radius  # common rough assumption used in RocketPy docs
    dry_mass = max(0.0, motor.total_mass_kg - motor.prop_mass_kg)

    gen_motor = GenericMotor(
        thrust_source=thrust_source,
        burn_time=motor.burn_time_s,
        chamber_radius=chamber_radius,
        chamber_height=chamber_height,
        chamber_position=chamber_position,
        propellant_initial_mass=motor.prop_mass_kg,
        nozzle_radius=nozzle_radius,
        dry_mass=dry_mass,
        center_of_dry_mass_position=0.0,
        dry_inertia=(0.0, 0.0, 0.0),
        nozzle_position=0.0,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    # Place motor nozzle exit at rocket tail.
    # With center_of_mass_without_motor=0, we approximate tail at -0.85 m for a ~1.7 m rocket.
    # If you want exact, replace tail_offset with your measured COM-from-tail distance.
    tail_offset = -0.85
    rocket.add_motor(gen_motor, position=tail_offset)

    # Run flight. inclination: degrees from horizontal; heading: degrees from North. :contentReference[oaicite:12]{index=12}
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=rail_length_m,
        inclination=inclination_deg,
        heading=heading_deg,
    )

    # Outputs
    t_ap = float(flight.apogee_time)
    lat_ap = float(flight.latitude(t_ap))  # Flight.latitude is a function of time. :contentReference[oaicite:13]{index=13}
    lon_ap = float(flight.longitude(t_ap))
    apogee_m = float(flight.apogee)

    # drift in local frame
    apogee_x = float(flight.apogee_x)  # east
    apogee_y = float(flight.apogee_y)  # north

    return {
        "apogee_time_s": t_ap,
        "apogee_alt_m": apogee_m,
        "apogee_lat_deg": lat_ap,
        "apogee_lon_deg": lon_ap,
        "apogee_east_m": apogee_x,
        "apogee_north_m": apogee_y,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openrocket_template", type=str, required=True, help="One OpenRocket export CSV to extract rocket constants")
    ap.add_argument("--drag_off", type=str, required=True, help="powerOffDragCurve.csv (Mach,Cd)")
    ap.add_argument("--drag_on", type=str, required=True, help="powerOnDragCurve.csv (Mach,Cd) - can be same as off")
    ap.add_argument("--igra_file", type=str, required=True, help="IGRA2 text file with multiple soundings")
    ap.add_argument("--motors_csv", type=str, required=True, help="motors_manual.csv")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--n_sims", type=int, default=40)
    ap.add_argument("--n_wind_profiles", type=int, default=10)
    ap.add_argument("--n_motors", type=int, default=4)
    ap.add_argument("--elevation_m", type=float, default=0.0)
    ap.add_argument("--rail_length_m", type=float, default=1.0)
    ap.add_argument("--inclination_deg", type=float, default=90.0)
    ap.add_argument("--heading_jitter_deg", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-generated-thrust-curves", action="store_true", help="Use a synthetic trapezoidal thrust curve when a motor curve file is missing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_curves_dir = out_dir / "generated_thrust_curves"

    openrocket_template = resolve_input_path(args.openrocket_template, PROJECT_ROOT)
    drag_off = resolve_input_path(args.drag_off, PROJECT_ROOT)
    drag_on = resolve_input_path(args.drag_on, PROJECT_ROOT)
    igra_file = resolve_input_path(args.igra_file, PROJECT_ROOT)
    motors_csv = resolve_input_path(args.motors_csv, PROJECT_ROOT)

    rocket_const = read_openrocket_template(openrocket_template)
    pad_lat = rocket_const["pad_lat_deg"]
    pad_lon = rocket_const["pad_lon_deg"]
    rocket_mass = rocket_const["mass_wo_motor_kg"]
    radius = rocket_const["radius_m"]
    inertia = (rocket_const["I11"], rocket_const["I22"], rocket_const["I33"])

    motors = load_motors_manual(motors_csv)
    if not motors:
        raise SystemExit("No motors loaded from motors_manual.csv")

    if not args.allow_generated_thrust_curves:
        missing_curves = [
            m.motor_id for m in motors if m.thrust_curve_file is None or not m.thrust_curve_file.exists()
        ]
        if missing_curves:
            missing_str = ", ".join(missing_curves)
            raise SystemExit(
                "Missing thrust curve files for: "
                f"{missing_str}. Fix the metadata paths or re-run with --allow-generated-thrust-curves."
            )

    # Parse IGRA soundings (multi-level wind)
    profiles = parse_igra2_profiles(igra_file, max_profiles=max(args.n_wind_profiles, 20))
    if len(profiles) < 2:
        raise SystemExit("Not enough IGRA profiles parsed. Ensure your file contains multiple '#STATION YYYY MM DD HH ...' blocks and GPH tokens like '18905A'.")

    # Select subset for the run
    profiles = profiles[: args.n_wind_profiles]
    motors = motors[: args.n_motors]

    # Build 40-run design: motors x wind profiles, repeated or truncated to n_sims
    design: List[Tuple[MotorDef, WindProfile]] = []
    for wp in profiles:
        for m in motors:
            design.append((m, wp))
    rng = random.Random(args.seed)
    rng.shuffle(design)

    if len(design) < args.n_sims:
        # repeat with replacement
        while len(design) < args.n_sims:
            design.append(rng.choice(design))
    design = design[: args.n_sims]

    manifest_rows = []
    results_rows = []

    for i, (mdef, wp) in enumerate(design, start=1):
        sim_id = f"SIM_{i:03d}_{mdef.motor_id}_{wp.profile_id}"
        # Choose heading "into wind" using surface wind direction, plus jitter
        base_heading = wp.surface_wdir_from_deg if np.isfinite(wp.surface_wdir_from_deg) else 0.0
        heading = (base_heading + rng.uniform(-args.heading_jitter_deg, args.heading_jitter_deg)) % 360.0

        manifest_rows.append({
            "sim_id": sim_id,
            "motor_id": mdef.motor_id,
            "wind_profile_id": wp.profile_id,
            "wind_station": wp.station,
            "wind_datetime_utc": wp.datetime_utc,
            "heading_deg": heading,
            "inclination_deg": args.inclination_deg,
            "rail_length_m": args.rail_length_m,
            "pad_lat_deg": pad_lat,
            "pad_lon_deg": pad_lon,
        })

        try:
            out = simulate(
                pad_lat=pad_lat,
                pad_lon=pad_lon,
                elevation_m=args.elevation_m,
                drag_off=drag_off,
                drag_on=drag_on,
                rocket_mass_kg=rocket_mass,
                radius_m=radius,
                inertia=inertia,
                motor=mdef,
                wind_profile=wp,
                rail_length_m=args.rail_length_m,
                inclination_deg=args.inclination_deg,
                heading_deg=heading,
                seed=args.seed + i,
                allow_generated_thrust_curves=args.allow_generated_thrust_curves,
                generated_curves_dir=generated_curves_dir,
            )
            out_row = {"sim_id": sim_id, "motor_id": mdef.motor_id, "wind_profile_id": wp.profile_id}
            out_row.update(out)
            results_rows.append(out_row)
            print(f"[OK] {sim_id} apogee={out['apogee_alt_m']:.1f} m lat={out['apogee_lat_deg']:.6f}")
        except Exception as e:
            print(f"[FAIL] {sim_id}: {e}")
            results_rows.append({
                "sim_id": sim_id, "motor_id": mdef.motor_id, "wind_profile_id": wp.profile_id,
                "error": str(e)
            })

    pd.DataFrame(manifest_rows).to_csv(out_dir / "sim_manifest.csv", index=False)
    pd.DataFrame(results_rows).to_csv(out_dir / "sim_results.csv", index=False)
    print(f"Done. Wrote:\n  {out_dir/'sim_manifest.csv'}\n  {out_dir/'sim_results.csv'}")


if __name__ == "__main__":
    main()
