import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


def parse_header_constants(path: str) -> Tuple[float, float]:
    """Read mr_std and tend_std from the two-line header used in gwo_daily_1974_2024.csv style.

    Lines look like:
      # mr_std_kg_m2_s,<value>
      # tend_std_kg_m2_s2,<value>
    Fallback to keys '# AAMstd' and '# TENDstd' if needed.
    """
    mr_std = None
    tend_std = None
    with open(path, "r", encoding="utf-8") as f:
        for _ in range(4):
            line = f.readline()
            if not line or not line.startswith("#"):
                break
            token = line.strip().lstrip("#").split(",")
            if len(token) == 2:
                k, v = token[0].strip(), token[1].strip()
                if k in ("mr_std_kg_m2_s", "AAMstd"):
                    mr_std = float(v)
                if k in ("tend_std_kg_m2_s2", "TENDstd"):
                    tend_std = float(v)
    if mr_std is None or tend_std is None:
        raise ValueError("Could not parse mr_std/tend_std from header")
    return float(mr_std), float(tend_std)


def five_point_tendency(series: pd.Series, dt_seconds: float) -> pd.Series:
    s = series
    s_m2, s_m1, s_p1, s_p2 = s.shift(2), s.shift(1), s.shift(-1), s.shift(-2)
    five = (s_m2 - 8.0 * s_m1 + 8.0 * s_p1 - s_p2) / (12.0 * dt_seconds)
    three = (s_p1 - s_m1) / (2.0 * dt_seconds)
    fwd = (s_p1 - s) / dt_seconds
    bwd = (s - s_m1) / dt_seconds
    out = five.where(s_m2.notna() & s_m1.notna() & s_p1.notna() & s_p2.notna())
    out = out.where(out.notna(), three.where(s_m1.notna() & s_p1.notna()))
    out = out.where(out.notna(), fwd.where(s_p1.notna()))
    out = out.where(out.notna(), bwd.where(s_m1.notna()))
    return out


def compute_stds_from_climo_series(climo_csv: str, window_days: int) -> Tuple[float, float]:
    # Accept either date,aam or timestamp,aam_total_kg_m2_s
    df = pd.read_csv(climo_csv)
    cols = set(df.columns)
    if {"date", "aam"}.issubset(cols):
        ts = pd.to_datetime(df["date"], errors="coerce")
        s = pd.Series(df["aam"].astype(float).values, index=ts)
    elif {"timestamp", "aam_total_kg_m2_s"}.issubset(cols):
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        s = pd.Series(df["aam_total_kg_m2_s"].astype(float).values, index=ts)
    else:
        raise ValueError("Climo CSV must have date,aam or timestamp,aam_total_kg_m2_s")

    s = s.sort_index()
    daily = s.resample("1D").mean()
    ddf = daily.to_frame(name="aam").reset_index()
    time_col = ddf.columns[0]
    ddf["mm"] = pd.to_datetime(ddf[time_col]).dt.strftime("%m")
    ddf["dd"] = pd.to_datetime(ddf[time_col]).dt.strftime("%d")
    clim = ddf.groupby(["mm", "dd"], as_index=False)["aam"].mean().rename(columns={"aam": "climo"})
    ddf = ddf.merge(clim, on=["mm", "dd"], how="left")
    anom = ddf["aam"] - ddf["climo"]
    sm = anom.rolling(window=window_days, center=True, min_periods=1).mean()
    tend = five_point_tendency(pd.Series(sm.values, index=pd.to_datetime(ddf[time_col])), 86400.0)
    return float(sm.std(skipna=True)), float(tend.std(skipna=True))


def compute_daily_gwo_from_ens(aam_csv: str, climo_csv: str, out_csv: str, window_days: int = 5) -> None:
    # Read standardization constants; if missing in header, compute from climo series
    try:
        mr_std_abs, tend_std_abs = parse_header_constants(climo_csv)
    except Exception:
        mr_std_abs, tend_std_abs = compute_stds_from_climo_series(climo_csv, window_days)

    df = pd.read_csv(aam_csv, parse_dates=["timestamp"])  # timestamp, member, aam_total_kg_m2_s
    if not {"timestamp", "member", "aam_total_kg_m2_s"}.issubset(df.columns):
        raise ValueError("AAM ensemble CSV must have columns: timestamp,member,aam_total_kg_m2_s")

    # Daily mean per member
    df = df.sort_values(["member", "timestamp"])  # ensure order
    df["date"] = df["timestamp"].dt.floor("D")
    daily = df.groupby(["member", "date"], as_index=False)["aam_total_kg_m2_s"].mean()

    # Build daily climatology from climo_csv (like compute_gwo_daily)
    climo = pd.read_csv(climo_csv, comment="#")
    # Support both climo formats
    if {"date", "aam"}.issubset(climo.columns):
        ts = pd.to_datetime(climo["date"], errors="coerce")
        cser = pd.Series(climo["aam"].astype(float).values, index=ts).sort_index()
    elif {"timestamp", "aam_total_kg_m2_s"}.issubset(climo.columns):
        ts = pd.to_datetime(climo["timestamp"], errors="coerce")
        cser = pd.Series(climo["aam_total_kg_m2_s"].astype(float).values, index=ts).sort_index()
    else:
        # Fallback to first two columns
        ts = pd.to_datetime(climo[climo.columns[0]], errors="coerce")
        cser = pd.Series(climo[climo.columns[1]].astype(float).values, index=ts).sort_index()
    c_daily = cser.resample("1D").mean()
    c_df = c_daily.to_frame(name="aam").reset_index()
    c_time_col = c_df.columns[0]
    c_df["mm"] = pd.to_datetime(c_df[c_time_col]).dt.strftime("%m")
    c_df["dd"] = pd.to_datetime(c_df[c_time_col]).dt.strftime("%d")
    climo_daily = c_df.groupby(["mm", "dd"], as_index=False)["aam"].mean().rename(columns={"aam": "climo"})

    # Merge and get anomalies per member
    d_df = daily.rename(columns={"aam_total_kg_m2_s": "aam"}).copy()
    d_df["mm"] = d_df["date"].dt.strftime("%m")
    d_df["dd"] = d_df["date"].dt.strftime("%d")
    d_df = d_df.merge(climo_daily, on=["mm", "dd"], how="left")
    d_df["anom"] = d_df["aam"] - d_df["climo"]

    # Smooth per member
    def smooth_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date")
        g["anom_sm"] = g["anom"].rolling(window=window_days, center=True, min_periods=1).mean()
        return g
    # Pandas compatibility: avoid include_groups (older versions)
    d_df = d_df.groupby("member", as_index=False, group_keys=False).apply(smooth_group)

    # Tendency per member
    tend_list = []
    for m, g in d_df.groupby("member"):
        s = pd.Series(g["anom_sm"].values, index=g["date"])  # daily cadence
        tend = five_point_tendency(s, 86400.0)
        gg = g.copy()
        gg["tend"] = tend.values
        gg["member"] = m
        tend_list.append(gg)
    d_df = pd.concat(tend_list, ignore_index=True)

    # Standardize per member
    d_df["GWO1"] = d_df["anom_sm"] / mr_std_abs
    d_df["GWO2"] = d_df["tend"] / tend_std_abs
    angle_deg = (np.degrees(np.arctan2(d_df["GWO1"], d_df["GWO2"])) + 360.0) % 360.0
    # Phase calibration: rotate by 180° so Phase 1 starts in bottom-left, increasing CCW
    angle_shift = (angle_deg - 180.0 + 360.0) % 360.0
    d_df["amp"] = np.sqrt(d_df["GWO1"] ** 2 + d_df["GWO2"] ** 2)
    phase = np.floor(angle_shift / 45.0).astype("Int64") + 1
    d_df["phase"] = phase.where(phase != 9, other=8)

    # Aggregate across members: median and probabilities
    agg = d_df.groupby("date").apply(
        lambda g: pd.Series({
            "GWO1_med": g["GWO1"].median(),
            "GWO2_med": g["GWO2"].median(),
            "amp_med": g["amp"].median(),
            **{f"P_phase_{k}": float((g["phase"] == k).mean()) for k in range(1, 9)},
            "P_amp_ge_1": float((g["amp"] >= 1.0).mean()),
            "P_amp_ge_2": float((g["amp"] >= 2.0).mean()),
        })
    ).reset_index()

    # Add single integer phase label (most probable phase 1–8) and drop P_phase_* columns
    prob_cols = [f"P_phase_{k}" for k in range(1, 9)]
    if any(col in agg.columns for col in prob_cols):
        present = [c for c in prob_cols if c in agg.columns]
        if present:
            agg["phase"] = (
                agg[present].idxmax(axis=1).str.replace("P_phase_", "", regex=False).astype(int)
            )
            agg = agg.drop(columns=present)

    agg.to_csv(out_csv, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute daily ensemble GWO from an ensemble AAM CSV (6-hourly), using existing climatology header constants")
    p.add_argument("--aam-ens-csv", required=True, help="Input CSV from compute_aam_open_data_ens.py")
    p.add_argument("--climo-csv", default="aam_climo.csv", help="Climatology CSV with header mr_std/tend_std")
    p.add_argument("--out-csv", default="gwo_daily_ens.csv", help="Output aggregated daily GWO CSV")
    p.add_argument("--window-days", type=int, default=5)
    args = p.parse_args()

    compute_daily_gwo_from_ens(args.aam_ens_csv, args.climo_csv, args.out_csv, window_days=args.window_days)


if __name__ == "__main__":
    main()


