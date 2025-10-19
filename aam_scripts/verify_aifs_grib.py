import argparse
import glob
import os

import numpy as np
import xarray as xr


def main() -> None:
    p = argparse.ArgumentParser(description="Verify AIFS ENS GRIB: members, levels, steps, and global grid coverage")
    p.add_argument("--file", help="Path to GRIB2 file. If omitted, auto-pick the only .grib2 in forecast_aifs_ens")
    args = p.parse_args()

    if args.file:
        path = args.file
    else:
        default_dir = r"D:\nixon et al\forecast_aifs_ens"
        candidates = sorted(glob.glob(os.path.join(default_dir, "*.grib2")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(default_dir, "*.grib")))
        if not candidates:
            raise SystemExit(f"No .grib/.grib2 files found in {default_dir}")
        if len(candidates) > 1:
            print("Multiple GRIB files found; using the first:", candidates[0])
        path = candidates[0]

    print("Verifying:", path)

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "u", "typeOfLevel": "isobaricInhPa"}},
    )
    try:
        u = ds["u"]
        dims = {d: int(u.sizes[d]) for d in u.dims}
        print("Dims:", u.dims, dims)
        print("Coords:", list(u.coords))

        # Times
        time = u.coords.get("time", None)
        valid_time = ds.coords.get("valid_time", None)
        if time is not None:
            t0 = np.unique(np.array(time))
            print("Init time count:", t0.size, "first:", t0[0] if t0.size else None)
        if valid_time is not None:
            vt = np.array(valid_time)
            print("Valid time count:", vt.size, "first:", vt[0] if vt.size else None, "last:", vt[-1] if vt.size else None)
            if vt.size > 2:
                diffs_h = np.unique(np.diff(vt).astype("timedelta64[h]").astype(int))
                print("Unique valid_time step sizes (h):", diffs_h[:10])

        # Members
        if "number" in u.dims:
            nums = np.array(u["number"])  # type: ignore[index]
            print("Members range:", int(nums.min()), "..", int(nums.max()), "count:", nums.size)
        else:
            print("No 'number' dimension found (this may be CF-only or EF)")

        # Levels
        lev = np.array(u["isobaricInhPa"])  # type: ignore[index]
        print("Levels (hPa):", lev.tolist())
        print("Level count:", lev.size)

        # Grid coverage
        lat = np.array(u["latitude"])  # type: ignore[index]
        lon = np.array(u["longitude"])  # type: ignore[index]
        print("Lat min/max/count:", float(lat.min()), float(lat.max()), lat.size)
        print("Lon min/max/count:", float(lon.min()), float(lon.max()), lon.size)
        dlats = np.unique(np.round(np.diff(lat), 6))
        dlons = np.unique(np.round(np.diff(lon), 6))
        print("Unique dlat:", dlats[:5], "Unique dlon:", dlons[:5])
        lat_ok = (abs(lat.min() + 90) < 1e-3) and (abs(lat.max() - 90) < 1e-3)
        # For longitudes, many grids use cell centers so max < 180 by one step.
        # Treat as global if (max - min + dlon) ≈ 360 within tolerance.
        mean_dlon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 0.0
        lon_span = float(lon.max() - lon.min())
        lon_ok = abs((lon_span + mean_dlon) - 360.0) < 0.5
        print("Covers full globe? lat:", lat_ok, "lon:", lon_ok)

        # Sample NaN check – index by integer positions (isel), not coordinate values
        idx = {"step": 0, "isobaricInhPa": 0}
        if "number" in u.dims:
            idx["number"] = 0
        sample = u.isel(**idx).values
        nan_frac = float(np.isnan(sample).mean())
        print("NaN fraction (one slice):", nan_frac)
    finally:
        ds.close()


if __name__ == "__main__":
    main()


