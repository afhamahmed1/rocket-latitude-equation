# Rocket Altitude Equation

This repo has two separate workflows:

1. Build a tabular dataset from real rocket flight logs.
2. Generate synthetic flights with RocketPy using IGRA wind soundings.

## Setup

Install dependencies with your preferred environment manager. For example:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Real-flight Dataset Builder

Entry point: `data-formater.py`

This script reads:

- `data/metadata/manifest.csv`
- Raven altimeter files from `data/altimeter/`
- Featherweight GPS files from `data/gps/`
- optional metadata from `data/metadata/motors_manual.csv`
- optional metadata from `data/metadata/rockets_manual.csv`
- optional metadata from `data/metadata/flights_metadata.csv`

It writes:

- `out/flights_dataset.csv`
- cleaned intermediate tracks under `out/cleaned_tracks/`
- `out/build_log.txt`

Run it from the repo root:

```bash
python data-formater.py --data-dir ./data --out-dir ./out
```

Useful options:

- `--gps-alt-unit auto|ft|m|mm`
- `--alt-rise-m 5.0`
- `--slope-mps 2.0`
- `--min-gps-apogee-gain-m 30.0`
- `--min-gps-descent-after-apogee-m 10.0`

Notes:

- Manifest and metadata paths can be relative to the metadata file location or the repo root.
- The script now rejects GPS apogee targets that do not show a meaningful climb and descent.
- Altimeter-timed interpolation only succeeds when GPS actually covers the Raven apogee time.

## RocketPy Simulation Builder

Entry point: `rocketpy_generate_40_sims_igra.py`

This script needs:

- an OpenRocket export CSV with rocket mass, motor mass, latitude, longitude, reference area, and inertia columns
- drag curves such as `data/drag_curves/powerOffDragCurve.csv` and `data/drag_curves/powerOnDragCurve.csv`
- an IGRA text file such as `data/igra/igra_sample.txt`
- motor metadata in `data/motors/motors_manual.csv`

Run it from the repo root:

```bash
python rocketpy_generate_40_sims_igra.py ^
  --openrocket_template data/openrocket_template.csv ^
  --drag_off data/drag_curves/powerOffDragCurve.csv ^
  --drag_on data/drag_curves/powerOnDragCurve.csv ^
  --igra_file data/igra/igra_sample.txt ^
  --motors_csv data/motors/motors_manual.csv ^
  --out_dir out
```

Useful options:

- `--n_sims 40`
- `--n_wind_profiles 10`
- `--n_motors 4`
- `--heading_jitter_deg 10.0`
- `--allow-generated-thrust-curves`

Notes:

- Thrust-curve file paths are validated before simulation. By default, missing files stop the run.
- `--allow-generated-thrust-curves` enables a synthetic trapezoidal fallback for experimentation only.
- The checked-in `data/openrocket_template.csv` is currently empty. Replace it with a real OpenRocket export before running simulations.

## Data Files

- `data/metadata/manifest.csv` pairs each flight with its altimeter and GPS files.
- `data/motors/motors_manual.csv` provides motor geometry and thrust-curve paths for simulation.
- `sample_data.txt` contains snippets of supported Raven, GPS, and trimmed TXT formats.

## Current Limitations

- Simulation accuracy still depends on rough geometry assumptions such as motor placement and inertia mapping.
- There is not yet an automated test suite in this repo.
