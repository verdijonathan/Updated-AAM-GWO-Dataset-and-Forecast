import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# Physical constants
GRAVITY_MS2: float = 9.81
EARTH_RADIUS_M: float = 6_371_220.0
PI: float = math.pi
DEG_TO_RAD: float = PI / 180.0


# Open Data pressure levels (hPa)
OPEN_DATA_13_LEVELS_HPA: List[float] = [
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50,
]


@dataclass
class QCIssue:
    date_str: str
    kind: str
    detail: str


def compute_dp_pa(levels_input: np.ndarray) -> np.ndarray:
    """Compute layer thickness dp (Pa) using dataset bounds (top=min(level), bottom=max(level)).

    For each level i (levels in hPa ascending by pressure):
      upper_bound = top=min(p) for i=0 else 0.5*(p[i-1] + p[i])
      lower_bound = bottom=max(p) for i=n-1 else 0.5*(p[i] + p[i+1])
      dp_hpa = lower_bound - upper_bound (positive)

    Accepts levels in hPa or Pa (auto-detected) and returns dp in Pa.
    """
    p = np.asarray(levels_input, dtype=float)
    if p.ndim != 1:
        raise ValueError("levels must be 1D")
    # Auto-detect units (Pa vs hPa)
    if p.max() > 2000.0:
        p = p / 100.0  # convert Pa → hPa
    # Ensure ascending by pressure
    reversed_order = False
    if p[0] > p[-1]:
        p = p[::-1]
        reversed_order = True
    n = p.size
    if n < 2:
        return np.zeros_like(p)
    mid = 0.5 * (p[:-1] + p[1:])
    upper = np.empty(n, dtype=float)
    lower = np.empty(n, dtype=float)
    upper[0] = float(p[0])          # dataset top (e.g., 50 hPa)
    upper[1:] = mid
    lower[:-1] = mid
    lower[-1] = float(p[-1])        # dataset bottom (e.g., 1000 hPa)
    dp_hpa = lower - upper
    if reversed_order:
        dp_hpa = dp_hpa[::-1]
    return dp_hpa * 100.0  # hPa → Pa


def find_u_variable(ds: xr.Dataset) -> xr.DataArray:
    """Return the u-wind DataArray from a pressure-level dataset.

    Tries common names and normalizes dimensions to (time, level, latitude, longitude).
    Requires an existing time dimension.
    """
    for cand in ("u", "u_component_of_wind"):
        if cand in ds:
            u = ds[cand]
            break
    else:
        raise KeyError("Could not find 'u' variable in dataset (looked for 'u' or 'u_component_of_wind')")

    # Normalize dimension names
    dim_map: Dict[str, str] = {
        "isobaricInhPa": "level",
        "pressure_level": "level",
        "latitude": "latitude",
        "lat": "latitude",
        "longitude": "longitude",
        "lon": "longitude",
        "time": "time",
        "valid_time": "time",
    }
    rename = {d: dim_map[d] for d in u.dims if d in dim_map and dim_map[d] != d}
    if rename:
        u = u.rename(rename)

    # Ensure required dims exist
    if "level" not in u.dims:
        raise ValueError("u-wind missing required dimension: level")
    if "latitude" not in u.dims or "longitude" not in u.dims:
        raise ValueError("u-wind must have latitude and longitude dimensions")
    if "time" not in u.dims:
        raise ValueError("u-wind missing required dimension: time")

    # Order dims
    u = u.transpose("time", "level", "latitude", "longitude")
    return u


def compute_aam_timeseries(u: xr.DataArray) -> xr.DataArray:
    """Compute total relative AAM per timestamp (kg m^2 s^-1) using dataset-bounded dp."""
    # Coordinates and spacing
    lat = u["latitude"].values.astype(float)
    if lat.size < 2:
        raise ValueError("Latitude coordinate must have at least 2 points")
    dlat_deg = abs(float(lat[1] - lat[0]))
    dphi = dlat_deg * DEG_TO_RAD

    # Pressure levels and dp (Pa) — dataset bounds (e.g., 50..1000 hPa)
    levels = u["level"].values
    dp_pa = compute_dp_pa(levels)

    # Compute UDP and collapse dimensions
    udp = u * xr.DataArray(dp_pa, dims=["level"])  # (time, level, lat, lon)
    level_udp = udp.sum(dim="level")              # (time, lat, lon)
    zonal_udp = level_udp.mean(dim="longitude")   # (time, lat)

    # cos^2(phi) factor per latitude
    cos_phi_sq = np.cos(lat * DEG_TO_RAD) ** 2
    cos_phi_sq_da = xr.DataArray(cos_phi_sq, dims=["latitude"], coords={"latitude": u["latitude"]})

    # Integrate over lon (2π via factor) and lat (dphi)
    factor = 2.0 * PI * (EARTH_RADIUS_M ** 3) / GRAVITY_MS2 * dphi
    aam_by_lat = zonal_udp * cos_phi_sq_da * factor
    aam_total = aam_by_lat.sum(dim="latitude")
    aam_total.name = "aam_total"
    aam_total.attrs["units"] = "kg m^2 s^-1"
    return aam_total


def normalize_latlon_to_baseline(u: xr.DataArray, baseline_lat: Optional[np.ndarray], baseline_lon: Optional[np.ndarray]) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    lat = u["latitude"].values
    lon = u["longitude"].values
    if baseline_lat is None or baseline_lon is None:
        return u, lat.copy(), lon.copy()
    # Handle reversed latitude
    if lat.shape == baseline_lat.shape and np.allclose(lat[::-1], baseline_lat, atol=0, rtol=0):
        u = u.isel(latitude=slice(None, None, -1))
        lat = u["latitude"].values
    # Handle reversed longitude
    if lon.shape == baseline_lon.shape and np.allclose(lon[::-1], baseline_lon, atol=0, rtol=0):
        u = u.isel(longitude=slice(None, None, -1))
        lon = u["longitude"].values
    return u, lat, lon


def select_synoptic_hours(u: xr.DataArray, hours: List[int]) -> xr.DataArray:
    hour_indexer = np.isin(u["time"].dt.hour, np.asarray(hours, dtype=int))
    selected = u.sel(time=hour_indexer)
    return selected.sortby("time")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute AAM (13 pressure levels) from cloud Zarr stores with QC and export to CSV")
    p.add_argument("--zarr-template", required=True, help="Zarr URL template or consolidated store (e.g., gs://...zarr-v3)")
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--times", nargs="+", default=["00:00", "06:00", "12:00", "18:00"], help="UTC times to include per day")
    p.add_argument("--output-csv", default="aam_13pl.csv", help="Output CSV path (date,aam)")
    p.add_argument("--qc-report", default="aam_13pl_qc.txt", help="Path to write QC findings")
    args = p.parse_args()

    # Parse hours
    hours: List[int] = []
    for t in args.times:
        hh = int(t.split(":")[0])
        hours.append(hh)
    hours = sorted(set(hours))

    records: List[Tuple[str, float]] = []
    issues: List[QCIssue] = []
    baseline_lat: Optional[np.ndarray] = None
    baseline_lon: Optional[np.ndarray] = None

    date_range = pd.date_range(f"{args.start_year}-01-01", f"{args.end_year}-12-31", freq="D")
    total_days = int(len(date_range))
    for idx, dt in enumerate(date_range, start=1):
        y, m, d = int(dt.year), int(dt.month), int(dt.day)
        date_str = dt.strftime("%Y-%m-%d")
        # Open per-day to keep memory bounded and avoid cache growth
        try:
            if "{year" in args.zarr_template or "{month" in args.zarr_template or "{day" in args.zarr_template:
                url = args.zarr_template.format(year=y, month=m, day=d)
            else:
                url = args.zarr_template
            ds = xr.open_zarr(url, consolidated=True, chunks={"time": 24}, storage_options={"token": "anon"})
        except Exception as exc:  # noqa: BLE001
            issues.append(QCIssue(date_str, "open_failed", f"{type(exc).__name__}: {exc}"))
            print(f"{date_str} [{idx}/{total_days}] ERROR: open_failed", flush=True)
            continue
        try:
            u = find_u_variable(ds)
        except Exception as exc:  # noqa: BLE001
            issues.append(QCIssue(date_str, "u_missing", f"{type(exc).__name__}: {exc}"))
            ds.close()
            print(f"{date_str} [{idx}/{total_days}] ERROR: u_missing", flush=True)
            continue

        # Restrict to the specific day, then select requested synoptic hours and 13 pressure levels
        try:
            day_start = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T00:00:00")
            day_end = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T23:59:59")
            u_day = u.sel(time=slice(day_start, day_end))
            u_sel = select_synoptic_hours(u_day, hours)
            # Select 13 open-data levels
            u_sel = u_sel.sel(level=OPEN_DATA_13_LEVELS_HPA, drop=True)
        except Exception as exc:  # noqa: BLE001
            issues.append(QCIssue(date_str, "time_or_level_select_failed", f"{type(exc).__name__}: {exc}"))
            ds.close()
            print(f"{date_str} [{idx}/{total_days}] ERROR: select_failed", flush=True)
            continue

        # Check we have any requested times
        got_hours = sorted(set(int(h) for h in u_sel["time"].dt.hour.values.tolist()))
        if not got_hours:
            issues.append(QCIssue(date_str, "empty_selection", "no requested hours present"))
            ds.close()
            continue

        # QC and normalize grids
        u_qc, lat_vals, lon_vals = normalize_latlon_to_baseline(u_sel, baseline_lat, baseline_lon)
        if baseline_lat is None:
            baseline_lat = lat_vals.copy()
        if baseline_lon is None:
            baseline_lon = lon_vals.copy()

        # Basic sanity checks
        if u_qc.size == 0:
            issues.append(QCIssue(date_str, "empty_selection", "no data after selection"))
            ds.close()
            continue
        max_abs = float(np.nanmax(np.abs(u_qc.values)))
        if not np.isfinite(max_abs) or max_abs > 200.0:
            issues.append(QCIssue(date_str, "u_out_of_range", f"abs(u) max {max_abs:.3f} m/s"))

        # Compute AAM for these timestamps (dp uses dataset 50–1000 bounds automatically)
        try:
            aam_series = compute_aam_timeseries(u_qc)
        except Exception as exc:  # noqa: BLE001
            issues.append(QCIssue(date_str, "aam_failed", f"{type(exc).__name__}: {exc}"))
            ds.close()
            print(f"{date_str} [{idx}/{total_days}] ERROR: aam_failed", flush=True)
            continue

        # Append records (YYYY-MM-DD-HH)
        times = pd.to_datetime(aam_series["time"].values)
        vals = aam_series.values.astype(float)
        for tt, vv in zip(times, vals):
            ts_str = pd.Timestamp(tt).strftime("%Y-%m-%d-%H")
            records.append((ts_str, float(vv)))

        print(f"{date_str} [{idx}/{total_days}] done ({len(times)} timestamps)", flush=True)
        ds.close()

    # Write output CSV
    out_df = pd.DataFrame(records, columns=["date", "aam"]).sort_values("date")
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved 13-level AAM time series to {args.output_csv} with {len(out_df)} rows")


if __name__ == "__main__":
    main()


