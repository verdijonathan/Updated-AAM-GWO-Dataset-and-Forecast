import argparse
import glob
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
try:
    from eccodes import (
        codes_grib_new_from_file,
        codes_get,
        codes_get_array,
        codes_release,
    )
except Exception:  # pragma: no cover - optional dependency
    codes_grib_new_from_file = None  # type: ignore[assignment]


# Physical constants
GRAVITY_MS2: float = 9.81
EARTH_RADIUS_M: float = 6_371_220.0
PI: float = math.pi
DEG_TO_RAD: float = PI / 180.0


OPEN_DATA_13_LEVELS_HPA: List[float] = [
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
]


def compute_dp_pa(levels_input: np.ndarray, *, top_hpa: Optional[float], bottom_hpa: Optional[float]) -> np.ndarray:
    """Compute layer thickness dp (Pa) using half-levels and configurable top/bottom bounds.

    levels_input can be in Pa or hPa (auto-detected). Returned units are Pa.
    Assumes levels are distinct and represent mid-pressure of each layer.
    """
    p = np.asarray(levels_input, dtype=float)
    if p.ndim != 1:
        raise ValueError("levels must be 1D")
    if p.max() > 2000.0:
        p = p / 100.0  # convert Pa → hPa
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
    # Use dataset bounds if provided; otherwise default to 1–1000 hPa
    upper[0] = float(top_hpa if top_hpa is not None else 1.0)
    upper[1:] = mid
    lower[:-1] = mid
    lower[-1] = float(bottom_hpa if bottom_hpa is not None else 1000.0)
    dp_hpa = lower - upper
    if reversed_order:
        dp_hpa = dp_hpa[::-1]
    return dp_hpa * 100.0


def normalize_dims_and_find_u(ds: xr.Dataset) -> xr.DataArray:
    """Find u-wind and normalize dims to (time, level, latitude, longitude[, number])."""
    name_candidates = ("u", "u_component_of_wind")
    for cand in name_candidates:
        if cand in ds:
            u = ds[cand]
            break
    else:
        raise KeyError("Could not find u-wind variable (tried 'u' and 'u_component_of_wind')")

    dim_map: Dict[str, str] = {
        "isobaricInhPa": "level",
        "pressure_level": "level",
        "latitude": "latitude",
        "lat": "latitude",
        "longitude": "longitude",
        "lon": "longitude",
        "valid_time": "time",
        "time": "time",
        "step": "step",
        "number": "number",
    }
    rename = {d: dim_map[d] for d in u.dims if d in dim_map and dim_map[d] != d}
    if rename:
        u = u.rename(rename)

    # If step dimension is present (forecast files), attach a 'valid_time' coordinate if available,
    # but keep 'step' as the dimension to avoid heavy reindexing.
    if "step" in u.dims:
        if "valid_time" in ds.coords:
            vt = ds["valid_time"]
            if "time" in vt.dims and vt.sizes.get("time", 1) == 1 and "step" in vt.dims:
                u = u.assign_coords(valid_time=vt.squeeze("time", drop=True))
            elif vt.ndim == 1 and "step" in vt.dims:
                u = u.assign_coords(valid_time=vt)

    # Ensure required dims: require lat/lon/level and either 'time' or 'step'
    dims_set = set(u.dims)
    required_basic = {"level", "latitude", "longitude"}
    if not required_basic.issubset(dims_set) or ("time" not in dims_set and "step" not in dims_set):
        need = required_basic.union({"time|step"})
        raise ValueError(f"u-wind missing dims; need at least {need}, got {dims_set}")

    # Keep original dimension order to avoid triggering eager loads; only ensure latitude/longitude names are normalized
    # (u already renamed earlier). We won't transpose here.
    return u


def compute_aam(u: xr.DataArray, dp_pa: np.ndarray) -> xr.DataArray:
    """Compute total relative AAM with an optional 'number' ensemble dimension.

    Returns DataArray with dims (time,) or (time, number); units kg m^2 s^-1.
    Notes: This function expects u with dims (time[, number], level, latitude, longitude)
    and may materialize large arrays. For very large inputs, prefer compute_aam_streaming.
    """
    lat = u["latitude"].values.astype(float)
    if lat.size < 2:
        raise ValueError("Latitude coordinate must have at least 2 points")
    dlat_deg = abs(float(lat[1] - lat[0]))
    dphi = dlat_deg * DEG_TO_RAD

    # Broadcast dp across level
    udp = u.astype("float32") * xr.DataArray(dp_pa.astype(np.float32), dims=["level"])  # (..., level, lat, lon)
    level_udp = udp.sum(dim="level")
    zonal_udp = level_udp.mean(dim="longitude")

    cos_phi_sq = np.cos(lat * DEG_TO_RAD).astype(np.float32) ** 2
    cos_phi_sq_da = xr.DataArray(cos_phi_sq, dims=["latitude"], coords={"latitude": u["latitude"]})

    factor = np.float32(2.0 * PI * (EARTH_RADIUS_M ** 3) / GRAVITY_MS2 * dphi)
    aam_by_lat = zonal_udp * cos_phi_sq_da * factor
    aam_total = aam_by_lat.sum(dim="latitude")
    aam_total.name = "aam_total"
    aam_total.attrs["units"] = "kg m^2 s^-1"
    return aam_total


def compute_aam_streaming(u: xr.DataArray, dp_pa: np.ndarray, allowed_hours: List[int]) -> xr.DataArray:
    """Memory-safe AAM compute: process one time slice at a time and stack results.

    Returns DataArray with dims (time,) or (time, number).
    """
    lat = u["latitude"].values.astype(np.float32)
    if lat.size < 2:
        raise ValueError("Latitude coordinate must have at least 2 points")
    dlat_deg = abs(float(lat[1] - lat[0]))
    dphi = dlat_deg * DEG_TO_RAD

    cos_phi_sq = (np.cos(lat * DEG_TO_RAD).astype(np.float32)) ** 2
    cos_phi_sq_da = xr.DataArray(cos_phi_sq, dims=["latitude"], coords={"latitude": u["latitude"]})
    dp_vec = dp_pa.astype(np.float32)  # (level,)
    factor = np.float32(2.0 * PI * (EARTH_RADIUS_M ** 3) / GRAVITY_MS2 * dphi)

    # Determine iteration dimension ('time' after swap, otherwise 'step')
    iter_dim = "time" if "time" in u.dims else ("step" if "step" in u.dims else None)
    if iter_dim is None:
        raise ValueError("Expected 'time' or 'step' dimension in forecast array")
    use_vt = False
    time_coord = None
    if ("valid_time" in u.coords) and ((iter_dim in u["valid_time"].dims) or u["valid_time"].ndim == 1):
        vt = u["valid_time"]
        vtv = vt.values
        time_coord = vtv if getattr(vtv, "ndim", 1) == 1 else np.squeeze(vtv)
        use_vt = True
    elif "time" in u.coords:
        t = u["time"].values
        time_coord = t if (getattr(t, "ndim", 1) == 1) else np.squeeze(t)
        use_vt = False
    else:
        time_coord = np.arange(u.sizes.get(iter_dim, 0))

    time_len = int(u.sizes.get(iter_dim, 0))
    has_number = "number" in u.dims
    num_len = int(u.sizes.get("number", 0)) if has_number else 0

    # Prepare output buffers
    # We will append only selected time indices to avoid preallocating huge arrays
    out_times: List[object] = []
    out_vals: List[np.ndarray] = []

    # Pure-numpy latitude weights
    cos_phi_sq_np = cos_phi_sq  # (lat,)

    for i in range(time_len):
        # Filter by hour if requested
        if allowed_hours:
            if use_vt:
                # numpy datetime64 array
                hh = int(np.datetime64(time_coord[i]).astype('datetime64[h]').astype(int) % 24)
            else:
                # best-effort: if time_coord provides datetime64
                try:
                    hh = int(np.datetime64(time_coord[i]).astype('datetime64[h]').astype(int) % 24)
                except Exception:
                    hh = None
            if hh is not None and (hh not in allowed_hours):
                continue
        if has_number:
            for j in range(num_len):
                ui = u.isel({iter_dim: i, "number": j})  # (level, lat, lon)
                arr = ui.values.astype(np.float32)
                # Multiply by dp across level: (level, lat, lon)
                udp = arr * dp_vec[:, None, None]
                # Sum over level -> (lat, lon)
                level_sum = udp.sum(axis=0, dtype=np.float64)
                # Zonal mean over lon -> (lat,)
                zonal = level_sum.mean(axis=1, dtype=np.float64)
                # Latitude weighting and integral
                aam_lat = zonal * cos_phi_sq_np * float(factor)
                total = aam_lat.sum(dtype=np.float64)
                out_vals.append(np.array([total], dtype=np.float64))
            out_times.append(time_coord[i])
        else:
            ui = u.isel({iter_dim: i})  # (level, lat, lon)
            arr = ui.values.astype(np.float32)
            udp = arr * dp_vec[:, None, None]
            level_sum = udp.sum(axis=0, dtype=np.float64)
            zonal = level_sum.mean(axis=1, dtype=np.float64)
            aam_lat = zonal * cos_phi_sq_np * float(factor)
            total = aam_lat.sum(dtype=np.float64)
            out_vals.append(np.array([total], dtype=np.float64))
            out_times.append(time_coord[i])

    # Build DataArray with appropriate coords
    if not out_times:
        return xr.DataArray([], dims=["time"], name="aam_total")
    times_da = xr.DataArray(out_times, dims=["time"])
    if has_number:
        # Reshape collected values into (time, number). We appended sequentially by (time, number).
        num_len = int(u.sizes.get("number", 0))
        vals = np.array([v[0] for v in out_vals], dtype=np.float64)
        # Determine unique times preserving order
        times_np = np.array(out_times)
        _, idx_first = np.unique(times_np, return_index=True)
        order = np.sort(idx_first)
        t_unique = times_np[order]
        out_arr = np.full((t_unique.size, num_len), np.nan, dtype=np.float64)
        # Fill row by row in the same sequence
        row = -1
        col = 0
        for k, val in enumerate(vals):
            if col == 0:
                row += 1
            out_arr[row, col] = val
            col = (col + 1) % num_len
        aam = xr.DataArray(out_arr, dims=["time", "number"], coords={"time": t_unique, "number": u["number"]}, name="aam_total")
    else:
        vals = np.stack(out_vals)[:, 0]
        aam = xr.DataArray(vals, dims=["time"], coords={"time": times_da.values}, name="aam_total")
    aam.attrs["units"] = "kg m^2 s^-1"
    return aam


def select_synoptic_hours(u: xr.DataArray, hours: List[int]) -> xr.DataArray:
    # Prefer filtering by a coordinate that aligns with a dimension
    hours_arr = np.asarray(hours, dtype=int)
    if "time" in u.dims:
        coord = u["time"]
        mask = np.isin(coord.dt.hour, hours_arr)
        out = u.sel(time=mask)
        return out.sortby("time")
    # If 'step' is the dimension and 'valid_time' is a 1D coord over 'step', filter against it
    if ("step" in u.dims) and ("valid_time" in u.coords) and (u["valid_time"].ndim == 1) and ("step" in u["valid_time"].dims):
        vt = u["valid_time"]
        mask_da = vt.dt.hour.isin(hours_arr)
        step_idx = np.nonzero(mask_da.values)[0]
        out = u.isel(step=step_idx)
        # keep valid_time subset aligned
        out = out.assign_coords(valid_time=vt.isel(step=step_idx))
        return out
    # Cannot filter by hour safely; return as-is
    return u


def load_mapping_csv(path: str) -> Dict[int, Tuple[float, float]]:
    """Load monthly mapping coefficients from CSV with columns: month,a,b."""
    df = pd.read_csv(path)
    need = {"month", "a", "b"}
    if not need.issubset(df.columns):
        raise ValueError("mapping CSV must have columns: month,a,b")
    mapping: Dict[int, Tuple[float, float]] = {}
    for _, r in df.iterrows():
        m = int(r["month"])  # 1..12
        mapping[m] = (float(r["a"]), float(r["b"]))
    return mapping


def apply_mapping(aam: pd.DataFrame, a_const: Optional[float], b_const: Optional[float], monthly: Optional[Dict[int, Tuple[float, float]]]) -> pd.DataFrame:
    if monthly is None and (a_const is None or b_const is None):
        return aam
    ts = pd.to_datetime(aam["timestamp"])  # type: ignore[index]
    months = ts.dt.month.values
    if monthly is not None:
        a_vals = np.array([monthly.get(int(m), (0.0, 1.0))[0] for m in months], dtype=float)
        b_vals = np.array([monthly.get(int(m), (0.0, 1.0))[1] for m in months], dtype=float)
    else:
        a_vals = np.full_like(months, float(a_const), dtype=float)
        b_vals = np.full_like(months, float(b_const), dtype=float)
    mapped = a_vals + b_vals * aam["aam_total_kg_m2_s"].to_numpy()
    out = aam.copy()
    out["aam_total_kg_m2_s"] = mapped
    return out


def open_inputs(paths: List[str], engine: Optional[str]) -> xr.Dataset:
    # Try to open as a multi-file dataset if multiple files are provided
    backend_kwargs = None
    if engine == "cfgrib":
        backend_kwargs = {
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "u"},
        }
    open_kwargs = {}
    # Avoid CF decoding of timedeltas that triggers warnings/errors; cfgrib already decodes appropriately
    open_kwargs["decode_timedelta"] = False
    if len(paths) == 1:
        return xr.open_dataset(paths[0], engine=engine, backend_kwargs=backend_kwargs, **open_kwargs)
    return xr.open_mfdataset(paths, combine="by_coords", engine=engine, backend_kwargs=backend_kwargs, **open_kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute 6-hourly ensemble AAM from ECMWF Open Data pressure levels (13 PL) with optional mapping to 37-level scale")
    p.add_argument("--input", required=True, help="Input GRIB/NetCDF path or glob (ensemble with 'number' dimension if available)")
    p.add_argument(
        "--engine",
        choices=["cfgrib", "netcdf4", "h5netcdf", "eccodes", None],
        default=None,
        help="Engine: cfgrib/netcdf/h5netcdf via xarray, or eccodes for direct GRIB streaming",
    )
    p.add_argument("--output-csv", default="aam_ens_6h.csv", help="Output CSV with columns: timestamp,member,aam_total_kg_m2_s")
    p.add_argument("--use-open-data-13", action="store_true", help="Restrict to the 13 Open Data PL levels (50–1000 hPa)")
    p.add_argument("--levels-hpa", type=float, nargs="+", help="Explicit list of pressure levels (hPa) to select")
    p.add_argument("--dp-bounds", choices=["dataset", "1-1000"], default="dataset", help="Vertical bounds for dp. 'dataset' uses min/max available levels; '1-1000' uses fixed")
    p.add_argument("--hours", nargs="+", default=["00:00", "06:00", "12:00", "18:00"], help="UTC times to include per day (HH:MM)")
    p.add_argument("--map-monthly-csv", help="Optional CSV with columns month,a,b to map 13L AAM to 37L")
    p.add_argument("--map-const-a", type=float, help="Constant intercept a for mapping (if monthly not provided)")
    p.add_argument("--map-const-b", type=float, help="Constant slope b for mapping (if monthly not provided)")
    args = p.parse_args()

    files = sorted(glob.glob(args.input)) or [args.input]
    # Eccodes-only streaming path: skip xarray/cfgrib entirely
    if args.engine == "eccodes":
        # Select levels
        levels_to_use: Optional[List[float]] = None
        if args.use_open_data_13 and not args.levels_hpa:
            levels_to_use = OPEN_DATA_13_LEVELS_HPA
        elif args.levels_hpa:
            levels_to_use = [float(x) for x in args.levels_hpa]

        hours_int = sorted({int(hh.split(":")[0]) for hh in args.hours})
        if codes_grib_new_from_file is None:
            raise RuntimeError("eccodes is not available; install 'eccodes' to use engine=eccodes")

        # Compute dp using dataset bounds (for provided levels)
        if levels_to_use is None:
            raise RuntimeError("When using engine=eccodes, please specify levels via --use-open-data-13 or --levels-hpa")
        lev_vals = np.array(levels_to_use, dtype=float)
        top_hpa = float(np.min(lev_vals))
        bottom_hpa = float(np.max(lev_vals))
        dp_pa = compute_dp_pa(lev_vals, top_hpa=top_hpa, bottom_hpa=bottom_hpa)
        dp_map = {float(lv): float(dp) for lv, dp in zip(lev_vals, dp_pa)}

        # Iterate GRIB messages and accumulate AAM per (valid_time, number)
        from datetime import datetime, timedelta
        EARTH_FACTOR_CACHE: Optional[Tuple[np.ndarray, float]] = None
        accum: Dict[Tuple[datetime, int], float] = {}
        times_set: set = set()
        numbers_set: set = set()

        for path in files:
            with open(path, "rb") as f:
                while True:
                    gid = codes_grib_new_from_file(f)
                    if gid is None:
                        break
                    try:
                        short_name = codes_get(gid, "shortName")
                        if short_name != "u":
                            continue
                        level_type = codes_get(gid, "typeOfLevel")
                        if level_type != "isobaricInhPa":
                            continue
                        level = float(codes_get(gid, "level"))
                        if (levels_to_use is not None) and (level not in levels_to_use):
                            continue
                        step = int(codes_get(gid, "step"))
                        data_date = int(codes_get(gid, "dataDate"))  # YYYYMMDD
                        data_time = int(codes_get(gid, "dataTime"))  # HHMM
                        hh_init = (data_time // 100) % 24
                        init_dt = datetime.strptime(f"{data_date:08d}{hh_init:02d}", "%Y%m%d%H")
                        valid_dt = init_dt + timedelta(hours=step)
                        if hours_int and (valid_dt.hour not in hours_int):
                            continue
                        # Determine ensemble member number robustly
                        try:
                            num = int(codes_get(gid, "number"))
                        except Exception:
                            try:
                                num = int(codes_get(gid, "perturbationNumber"))
                            except Exception:
                                # Default 0 (control or unknown)
                                num = 0

                        # Build cos^2(lat) and factor once from first field
                        if EARTH_FACTOR_CACHE is None:
                            Ni = int(codes_get(gid, "Ni"))
                            Nj = int(codes_get(gid, "Nj"))
                            lats = codes_get_array(gid, "latitudes").astype(np.float32).reshape(Nj, Ni)
                            lat1d = lats[:, 0]
                            dlat_deg = float(np.abs(np.diff(lat1d)).mean())
                            dphi = dlat_deg * DEG_TO_RAD
                            cos_phi_sq = (np.cos(lat1d * DEG_TO_RAD).astype(np.float32)) ** 2
                            factor = np.float32(2.0 * PI * (EARTH_RADIUS_M ** 3) / GRAVITY_MS2 * dphi)
                            EARTH_FACTOR_CACHE = (cos_phi_sq, float(factor))
                        cos_phi_sq, factor_f = EARTH_FACTOR_CACHE

                        # Read data values and compute contribution for this level
                        Ni = int(codes_get(gid, "Ni"))
                        Nj = int(codes_get(gid, "Nj"))
                        vals = codes_get_array(gid, "values").astype(np.float32).reshape(Nj, Ni)
                        dp = dp_map.get(level)
                        if dp is None:
                            continue
                        level_sum = (vals * dp).sum(axis=0, dtype=np.float64)  # sum over lon first? we'll compute zonal mean below
                        # Actually compute zonal mean over lon then integrate over lat
                        zonal = (vals * dp).mean(axis=1, dtype=np.float64)  # (lat,)
                        aam_lat = zonal * cos_phi_sq.astype(np.float64) * float(factor_f)
                        contrib = float(aam_lat.sum(dtype=np.float64))
                        key = (valid_dt, num)
                        accum[key] = accum.get(key, 0.0) + contrib
                        times_set.add(valid_dt)
                        numbers_set.add(num)
                    finally:
                        codes_release(gid)

        # Build DataArray (time, number)
        times_sorted = sorted(times_set)
        nums_sorted = sorted(numbers_set)
        out_arr = np.full((len(times_sorted), len(nums_sorted)), np.nan, dtype=np.float64)
        num_index = {n: i for i, n in enumerate(nums_sorted)}
        time_index = {t: i for i, t in enumerate(times_sorted)}
        for (t, n), v in accum.items():
            out_arr[time_index[t], num_index[n]] = v
        aam_da = xr.DataArray(out_arr, dims=["time", "number"], coords={"time": times_sorted, "number": nums_sorted}, name="aam_total")
        aam_da.attrs["units"] = "kg m^2 s^-1"

        # Continue pipeline below to CSV
        df = aam_da.to_series().reset_index()
        if "time" in df.columns:
            df.rename(columns={"time": "timestamp", "aam_total": "aam_total_kg_m2_s"}, inplace=True)
        else:
            df.rename(columns={"aam_total": "aam_total_kg_m2_s"}, inplace=True)
        if "number" in df.columns:
            df.rename(columns={"number": "member"}, inplace=True)
        else:
            df["member"] = 0

        # Apply optional mapping to 37-level scale
        monthly: Optional[Dict[int, Tuple[float, float]]] = None
        if args.map_monthly_csv:
            monthly = load_mapping_csv(args.map_monthly_csv)
        df = apply_mapping(df, args.map_const_a, args.map_const_b, monthly)

        out = df[["timestamp", "member", "aam_total_kg_m2_s"]].sort_values(["timestamp", "member"]).reset_index(drop=True)
        out.to_csv(args.output_csv, index=False)
        print(f"Saved ensemble AAM series to {args.output_csv} with {len(out)} rows")
        return

    ds = open_inputs(files, engine=args.engine)
    try:
        u = normalize_dims_and_find_u(ds)

        # Select requested levels
        levels_to_use: Optional[List[float]] = None
        if args.use_open_data_13 and not args.levels_hpa:
            levels_to_use = OPEN_DATA_13_LEVELS_HPA
        elif args.levels_hpa:
            levels_to_use = [float(x) for x in args.levels_hpa]
        if levels_to_use is not None:
            u = u.sel(level=levels_to_use, drop=True)

        # Compute dp using dataset bounds by default
        lev_vals = u["level"].values.astype(float)
        if args.dp_bounds == "dataset":
            top_hpa = float(np.min(lev_vals))
            bottom_hpa = float(np.max(lev_vals))
        else:
            top_hpa = 1.0
            bottom_hpa = 1000.0
        dp_pa = compute_dp_pa(lev_vals, top_hpa=top_hpa, bottom_hpa=bottom_hpa)

        # Select synoptic hours
        hours_int = sorted({int(hh.split(":")[0]) for hh in args.hours})
        u_sel = select_synoptic_hours(u, hours_int)

        # Compute AAM with streaming unconditionally to avoid large allocations.
        hours_int = sorted({int(hh.split(":")[0]) for hh in args.hours}) if isinstance(args.hours[0], str) else [int(h) for h in args.hours]
        aam_da = compute_aam_streaming(u_sel, dp_pa, hours_int)

        # Build DataFrame with timestamp and optional member
        df = aam_da.to_series().reset_index()
        if "time" in df.columns:
            df.rename(columns={"time": "timestamp", "aam_total": "aam_total_kg_m2_s"}, inplace=True)
        else:
            df.rename(columns={"aam_total": "aam_total_kg_m2_s"}, inplace=True)
        if "number" in df.columns:
            df.rename(columns={"number": "member"}, inplace=True)
        else:
            df["member"] = 0

        # Apply optional mapping to 37-level scale
        monthly: Optional[Dict[int, Tuple[float, float]]] = None
        if args.map_monthly_csv:
            monthly = load_mapping_csv(args.map_monthly_csv)
        df = apply_mapping(df, args.map_const_a, args.map_const_b, monthly)

        # Reorder columns and save
        out = df[["timestamp", "member", "aam_total_kg_m2_s"]].sort_values(["timestamp", "member"]).reset_index(drop=True)
        out.to_csv(args.output_csv, index=False)
        print(f"Saved ensemble AAM series to {args.output_csv} with {len(out)} rows")
    finally:
        ds.close()


if __name__ == "__main__":
    main()


