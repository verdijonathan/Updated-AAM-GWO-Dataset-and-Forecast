import glob, os
import xarray as xr
import numpy as np

files = glob.glob(r"D:\nixon et al\forecast_ens\*.grib2")
if not files:
    print("No files found"); raise SystemExit

members = set()
maxh = 0
per_member = {}

for f in files:
    try:
        ds = xr.open_dataset(
            f,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "u"}},
        )
        n = int(np.atleast_1d(ds["number"].values)[0]) if "number" in ds else 0
        members.add(n)
        per_member[n] = per_member.get(n, 0) + 1
        if "step" in ds:
            h = int(ds["step"].values.max() / np.timedelta64(1, "h"))
            maxh = max(maxh, h)
        ds.close()
    except Exception as e:
        print("Skip", os.path.basename(f), "->", e)

print("members:", sorted(members))
print("member_count:", len(members))
print("files_per_member (sample):", dict(list(per_member.items())[:5]))
print("max_step_hours:", maxh)
