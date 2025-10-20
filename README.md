## Global Wind Oscillation (GWO) – Daily Dataset (1974–2024)

This document describes how the daily GWO dataset in this folder was produced, including data inputs, formulas, processing steps, quality control, and how to reproduce results. It also explains how to interpret `GWO1`/`GWO2`, amplitude, and phase in `gwo_new.csv`.

### Scope and outputs
- **Coverage**: 1974–2024, daily cadence (UTC)
- **Outputs**: `date,GWO1,GWO2,amp,phase`
  - `GWO1`: standardized AAM anomaly (unitless)
  - `GWO2`: standardized AAM tendency (unitless)
  - `amp`: √(GWO1²+GWO2²)
  - `phase`: integer 1–8, 45° sectors in GWO phase space
- **Metadata**: Two header lines above the CSV table giving the absolute standard deviations used for standardization.
  - `# AAMstd,<value>` and `# TENDstd,<value>`

### Data inputs
- ERA5 pressure-level `u_component_of_wind` at 1–1000 hPa, global, 00/06/12/18 UTC
- Base period for climatology: 1991–2020

Scripts used (found in this folder):
- `download_era5_u_yearly.py` (optional helper) – retrieves ERA5 pressure-level u-wind (used for data from 2023-2024 only)
- `compute_aam_simple.py` or `compute_aam_from_cloud.py` – computes total relative AAM per timestamp; `compute_aam_from_cloud.py` computes AAM directly from the ARCO ERA5 datasets found on GCP (data is available through 2022)
- `merge_aam.py` – merges AAM CSV segments to `aam_master_1974_2024.csv`
- `compute_gwo_daily.py` – computes daily `GWO1/GWO2`, `amp`, and `phase`

### Methodology

1) Compute total relative AAM time series
- Discrete integral of zonal wind weighted by layer thickness and latitude:

$$
M_R = \frac{a^3}{g} \iiint \cos^2(\phi)\, u \, dp \, d\phi \, d\lambda
$$
- Key implementation details:
  - Pressure layer thickness `dp` via half-levels with top=1 hPa, bottom=1000 hPa
  - Zonal mean over longitude, weighted by $\cos^2(\phi)$, integrated over latitude with grid spacing
  - Units: kg m² s⁻¹

```python
# From compute_aam_simple.py
def compute_aam_timeseries(u: xr.DataArray) -> xr.DataArray:
    # ... coords and dp ...
    udp = u * xr.DataArray(dp_pa, dims=["level"])      # (time, level, lat, lon)
    level_udp = udp.sum(dim="level")                   # (time, lat, lon)
    zonal_udp = level_udp.mean(dim="longitude")        # (time, lat)
    cos_phi_sq = np.cos(lat * DEG_TO_RAD) ** 2
    cos_phi_sq_da = xr.DataArray(cos_phi_sq, dims=["latitude"], coords={"latitude": u["latitude"]})
    factor = 2.0 * PI * (EARTH_RADIUS_M ** 3) / GRAVITY_MS2 * dphi
    aam_by_lat = zonal_udp * cos_phi_sq_da * factor
    return aam_by_lat.sum(dim="latitude")  # (time,)
```

2) Build AAM dataset and climatology
- Merge individual AAM CSVs (columns normalized to `date(YYYY-MM-DD-HH),aam`) into `aam_master_1974_2024.csv` with duplicates dropped (keep first)
- From the 1991–2020 subset, compute daily climatology by month–day (after averaging to daily means)

```python
# From merge_aam.py
def normalize_and_load(csv_path: str) -> pd.DataFrame:
    # rename (timestamp→date, aam_total_kg_m2_s→aam), coerce datetime → %Y-%m-%d-%H
    return df[["date", "aam"]]
```

3) Daily AAM anomalies and smoothing
- Resample AAM to daily means (mean of 00/06/12/18 UTC values)
- Subtract daily (MM–DD) climatology to get anomaly
- Apply centered rolling mean (default window=5 days)

```python
# From compute_gwo_daily.py
daily = series.resample("1D").mean()
anom = d_df["aam"] - d_df["climo"]
smoothed = anom.rolling(window=5, center=True, min_periods=1).mean()
```

4) Daily tendency of the anomaly (robust edges)
- Centered 5-point stencil with fallbacks to 3-point and forward/backward 1-step at edges

```python
# From compute_gwo_daily.py
def five_point_tendency_with_edges(series: pd.Series, dt_seconds: float) -> pd.Series:
    s_m2, s_m1, s_p1, s_p2 = s.shift(2), s.shift(1), s.shift(-1), s.shift(-2)
    five = (s_m2 - 8*s_m1 + 8*s_p1 - s_p2) / (12*dt_seconds)
    three = (s_p1 - s_m1) / (2*dt_seconds)
    fwd = (s_p1 - s) / dt_seconds
    bwd = (s - s_m1) / dt_seconds
    # choose available stencil per index
```

5) Standardization constants from climatology
- On the climatology base (after smoothing and same tendency operator):
  - `mr_std_kg_m2_s` = std of smoothed daily anomaly (absolute units)
  - `tend_std_kg_m2_s2` = std of its daily tendency (absolute units)
- These two values are written as metadata above the output CSV (AAMstd, TENDstd)

6) GWO coordinates, amplitude, and phase
- Standardize to unitless components:
  - `GWO1 = (smoothed anomaly) / mr_std_kg_m2_s`
  - `GWO2 = (daily tendency) / tend_std_kg_m2_s2`
- Amplitude: `amp = sqrt(GWO1**2 + GWO2**2)`
- Phase: angle from `atan2(GWO1, GWO2)` in degrees, binned into 8×45° sectors (1–8)

```python
# From compute_gwo_daily.py (orientation: x=GWO2, y=GWO1)
angle_deg = (np.degrees(np.arctan2(gwo1, gwo2)) + 360.0) % 360.0
amp = np.sqrt(gwo1**2 + gwo2**2)
phase = np.floor(angle_deg / 45.0).astype("Int64") + 1
phase = phase.where(phase != 9, other=8)
```

### Phase calibration (origin and numbering)

- Orientation in this project: x = `GWO2` (horizontal), y = `GWO1` (vertical). Angles use `atan2(GWO1, GWO2)` and increase counter‑clockwise.
- Default numbering (from `compute_gwo_daily.py`): Phase 1 spans 0–45° (along +`GWO2`), then increases CCW.
- Calibrated numbering (preferred here): rotate angles by 180° so Phase 1 begins in the left→bottom‑left sector (180–225°), then increases CCW. Use `aam_scripts/calibrate_gwo_phase_yx.py` to apply this without changing `GWO1/GWO2/amp`.

Phase mapping (default → calibrated):
- 1 → 5
- 2 → 6
- 3 → 7
- 4 → 8
- 5 → 1
- 6 → 2
- 7 → 3
- 8 → 4

Apply calibration (example):

```bash
python aam_scripts/calibrate_gwo_phase_yx.py \
  --in-csv aam/gwo_new.csv \
  --out-csv aam/gwo_new.csv
```

This overwrites `gwo_new.csv` in place.

### Interpretation of phases and amplitude
- Phases are octants of the (`GWO2`,`GWO1`) plane:
- 1: falling through neutral (`GWO2 < 0`, `GWO1 ≈ 0`)
- 2: low and falling (`GWO1 < 0`, `GWO2 < 0`)
- 3: low, steady/bottoming (`GWO1 < 0`, `GWO2 ≈ 0`)
- 4: low and rising (`GWO1 < 0`, `GWO2 > 0`)
- 5: rising through neutral (`GWO2 > 0`, `GWO1 ≈ 0`)
- 6: high and rising (`GWO1 > 0`, `GWO2 > 0`)
- 7: high, steady/peaking (`GWO1 > 0`, `GWO2 ≈ 0`)
- 8: high and falling (`GWO1 > 0`, `GWO2 < 0`)
- Amplitude ≈ event strength; a common working threshold is `amp ≥ ~1` for “active” days.

### Differences vs legacy GWO file (`data/GWO_legacy.csv`)
- Legacy `Phase` is continuous (e.g., 6.5) and includes `Stage`
- New dataset provides integer `phase` (1–8) and exposes `GWO1/GWO2` directly

### Quality control (when computing AAM from cloud)
- Normalize lat/lon orientation to a baseline; flag mismatches
- Flag u-wind out-of-range (|u| > 200 m/s) and NaN fraction >1%
- Skip days missing requested synoptic hours (00/06/12/18)

### Reproducibility – example commands
These examples illustrate the sequence; adjust input paths/globs as needed.

```bash
# 1) (Optional) Download ERA5 u-wind pressure levels
python aam/download_era5_u_yearly.py --start-year 1991 --end-year 2020 --output-dir

# 2) Compute AAM time series from NetCDF files
python aam/compute_aam_simple.py --input "E:/era5/uwind/era5_u_pl_*.nc" --output-csv aam_1991_2020.csv

# 3) Merge AAM CSV segments (example set)
python aam/merge_aam.py

# 4) Compute daily GWO (uses merged AAM and the climatology CSV)
python aam/compute_gwo_daily.py --aam-csv aam/aam_master_1974_2024.csv \
  --climo-csv aam/aam_climo.csv --out-csv aam/gwo_new.csv --window-days 5
```

Notes:
- The header keys in the output CSV appear as `AAMstd/TENDstd` and are equivalent to `mr_std_kg_m2_s/tend_std_kg_m2_s2`
- The climatology base period is 1991–2020. Changing this period will change standardization and thus the resulting `GWO1/GWO2` magnitudes and `amp`.

### File manifest
- `compute_aam_simple.py` – AAM integral from NetCDF
- `compute_aam_from_cloud.py` – AAM integral from cloud Zarr with QC
- `download_era5_u_yearly.py` – ERA5 u-wind downloader
- `merge_aam.py` – merge/normalize AAM CSVs
- `compute_gwo_daily.py` – build daily GWO
- `aam_climo.csv` – AAM time series used to derive the daily climatology (base 1991–2020)
- `aam_master_1974_2024.csv` – merged AAM time series
- `gwo_new.csv` – final daily GWO dataset

### Forecast notes: ECMWF direct + 13→37 mapping calibration

This project uses the 13‑pressure‑level open data for forecasts and maps AAM to the historical 37‑level scale for consistency.

13→37 mapping – how it was created and calibrated
- We derived a monthly linear map using overlapping AAM time series at 13L and 37L:
  - For each month m, fit A37 ≈ a(m) + b(m)·A13 via least squares over the overlap period
  - Script: `aam_scripts/calibrate_mapping_13to37.py`
  - Outputs: `aam_data/mapping_13to37.csv` (columns: month,a,b) and `aam_data/mapping_metrics.csv` (fit diagnostics)

Example (template) to reproduce calibration
```bash
python aam_scripts/calibrate_mapping_13to37.py \
  --aam13 "path/to/aam_13level_*.csv" \
  --aam37 "path/to/aam_37level_*.csv" \
  --out-mapping aam_data/mapping_13to37.csv \
  --out-report  aam_data/mapping_metrics.csv
```

ECMWF download (AIFS ENS u‑wind, 13 PL)
```bash
# Prefer ECMWF first; AWS as fallback
python aam_scripts/download_aifs_ens_u_13pl.py \
  --date 2025-10-16 \
  --skip-ef --skip-cf \
  --source-order ecmwf,aws \
  --out-dir "forecast_aifs_ens"
```

Compute ensemble AAM at 13L and map to 37L
```bash
# For very large GRIBs on Windows, ecCodes streaming is robust
setx ECCODES_GRIB_64BIT_GLOBALS 1

python aam_scripts/compute_aam_open_data_ens.py \
  --engine eccodes \
  --input "forecast_aifs_ens/*.grib2" \
  --use-open-data-13 \
  --hours 00:00 06:00 12:00 18:00 \
  --map-monthly-csv "aam_data/mapping_13to37.csv" \
  --output-csv "aam_data/aam_ens_6h_mapped.csv"
```

Compute daily ensemble GWO
```bash
python aam_scripts/compute_gwo_daily_ens.py \
  --aam-ens-csv "aam_data/aam_ens_6h_mapped.csv" \
  --climo-csv   "aam_data/aam_climo.csv" \
  --out-csv     "aam_data/gwo_daily_ens.csv" \
  --window-days 5
```

Note:
- `--window-days` controls a centered smoothing window (in days) applied to daily AAM anomalies before computing tendency and GWO. A value of 5 matches the historical pipeline and reduces day-to-day phase flip‑flop while retaining synoptic signals.

Phase numbering calibration
- Applied automatically in `compute_gwo_daily_ens.py` (180° rotation so Phase 1 starts bottom‑left, CCW increase). This calibration is required for consistency with our interpretation.
