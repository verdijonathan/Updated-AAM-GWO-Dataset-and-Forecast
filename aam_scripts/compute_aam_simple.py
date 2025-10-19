import argparse
import glob
import math
from typing import List, Dict

import numpy as np
import xarray as xr


# Physical constants
GRAVITY_MS2: float = 9.81
EARTH_RADIUS_M: float = 6_371_220.0
PI: float = math.pi
DEG_TO_RAD: float = PI / 180.0


def compute_dp_pa(levels_input: np.ndarray) -> np.ndarray:
    """Compute layer thickness dp (Pa) assuming top/bottom bounds at 1 and 1000 hPa.

    For each level i (levels in hPa ascending by pressure):
      upper_bound = 1 hPa for i=0 else 0.5*(p[i-1] + p[i])
      lower_bound = 1000 hPa for i=n-1 else 0.5*(p[i] + p[i+1])
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
    upper[0] = 1.0
    upper[1:] = mid
    lower[:-1] = mid
    lower[-1] = 1000.0
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
    dim_map = {
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
    """Compute total relative atmospheric angular momentum (AAM) per timestamp.

    Discrete form of
        M_R = (a^3 / g) * ∫∫∫ cos^2(phi) * u * dp dphi dlambda

    where u is zonal wind (m/s), dp is layer thickness (Pa), phi is latitude (rad),
    a is Earth radius (m), and g is gravitational acceleration (m/s^2).

    Returns an xr.DataArray with dims (time,) and units kg m^2 s^-1.
    """
    # Coordinates and spacing
    lat = u["latitude"].values.astype(float)  # degrees
    if lat.size < 2:
        raise ValueError("Latitude coordinate must have at least 2 points")
    dlat_deg = abs(float(lat[1] - lat[0]))
    dphi = dlat_deg * DEG_TO_RAD

    # Pressure levels and dp (Pa)
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


def load_dataset(inputs: List[str], chunks: Dict[str, int]) -> xr.Dataset:
    """Open one or more files as a single dataset using chunked reads to limit memory."""
    if len(inputs) == 1:
        return xr.open_dataset(inputs[0], chunks=chunks)
    return xr.open_mfdataset(inputs, combine="by_coords", chunks=chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Atmospheric Relative Angular Momentum (AAM) from pressure-level u-wind"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path or glob pattern (e.g., era5_u_pl_*.nc)",
    )
    parser.add_argument(
        "--output-csv",
        default="aam_total.csv",
        help="CSV path for the output AAM time series",
    )
    parser.add_argument(
        "--chunks-time",
        type=int,
        default=1,
        help="Chunk size along time dimension (use 1 to stream over time)",
    )
    parser.add_argument(
        "--chunks-latlon",
        type=int,
        nargs=2,
        metavar=("NLAT", "NLON"),
        default=[180, 180],
        help="Chunk sizes for latitude and longitude",
    )
    args = parser.parse_args()

    # Resolve files from pattern or single path
    files = sorted(glob.glob(args.input))
    if not files:
        # Treat as literal path if glob did not expand
        files = [args.input]

    chunks: Dict[str, int] = {
        "time": int(args.chunks_time),
        "latitude": int(args.chunks_latlon[0]),
        "longitude": int(args.chunks_latlon[1]),
    }

    ds = load_dataset(files, chunks)
    try:
        u = find_u_variable(ds)
        aam_total = compute_aam_timeseries(u)
        df = aam_total.to_series().reset_index()
        # Normalize column names
        if "time" in df.columns:
            df.rename(columns={"time": "timestamp", "aam_total": "aam_total_kg_m2_s"}, inplace=True)
        else:
            df.rename(columns={"aam_total": "aam_total_kg_m2_s"}, inplace=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved AAM time series to {args.output_csv}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()


