import argparse
from typing import Tuple

import numpy as np
import pandas as pd


def parse_climo_daily_stats(aam_climo_csv: str, window_days: int = 5) -> Tuple[float, float]:
    """Compute absolute stds of daily MR anomaly and its daily tendency from climo.

    Steps:
      - Load climo (YYYY-MM-DD-HH, aam)
      - Average to daily means
      - Build daily climatology by mm-dd across years
      - Daily anomalies per day-year
      - Apply centered running mean (window_days)
      - Compute std of smoothed anomalies (kg m^2 s^-1)
      - Compute 5-point daily tendency and its std (kg m^2 s^-2)
    """
    df = pd.read_csv(aam_climo_csv)
    if not {"date", "aam"}.issubset(df.columns):
        raise ValueError("aam_climo.csv must have columns: date,aam")
    ts = pd.to_datetime(df["date"], format="%Y-%m-%d-%H", errors="coerce")
    series = pd.Series(df["aam"].astype(float).values, index=ts).sort_index()

    daily = series.resample("1D").mean()
    daily_df = daily.to_frame(name="aam").reset_index()
    daily_df["mm"] = daily_df["date"].dt.strftime("%m")
    daily_df["dd"] = daily_df["date"].dt.strftime("%d")
    # Daily climatology by month-day
    climo_daily = daily_df.groupby(["mm", "dd"], as_index=False)["aam"].mean().rename(columns={"aam": "climo"})
    daily_df = daily_df.merge(climo_daily, on=["mm", "dd"], how="left")
    anom = daily_df["aam"] - daily_df["climo"]
    smoothed = anom.rolling(window=window_days, center=True, min_periods=1).mean()

    # 5-point centered tendency with edge fallbacks
    def five_point_tendency(s: pd.Series, dt_seconds: float) -> pd.Series:
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

    tend = five_point_tendency(smoothed, 86400.0)
    return float(smoothed.std(skipna=True)), float(tend.std(skipna=True))


def daily_anom_from_series(aam_csv: str, climo_csv: str, window_days: int = 5) -> pd.Series:
    """Compute smoothed daily MR anomalies from an AAM CSV using climo daily means."""
    df = pd.read_csv(aam_csv, parse_dates=["timestamp"]).sort_values("timestamp")
    if not {"timestamp", "aam_total_kg_m2_s"}.issubset(df.columns):
        raise ValueError("AAM CSV must have columns: timestamp,aam_total_kg_m2_s")
    series = df.set_index("timestamp")["aam_total_kg_m2_s"]
    daily = series.resample("1D").mean()

    climo = pd.read_csv(climo_csv)
    ts = pd.to_datetime(climo["date"], format="%Y-%m-%d-%H", errors="coerce")
    cser = pd.Series(climo["aam"].astype(float).values, index=ts).sort_index()
    c_daily = cser.resample("1D").mean()
    c_df = c_daily.to_frame(name="aam").reset_index()
    c_df["mm"] = c_df["date"].dt.strftime("%m")
    c_df["dd"] = c_df["date"].dt.strftime("%d")
    climo_daily = c_df.groupby(["mm", "dd"], as_index=False)["aam"].mean().rename(columns={"aam": "climo"})

    d_df = daily.to_frame(name="aam").reset_index()
    d_df["mm"] = d_df["timestamp"].dt.strftime("%m")
    d_df["dd"] = d_df["timestamp"].dt.strftime("%d")
    d_df = d_df.merge(climo_daily, on=["mm", "dd"], how="left")
    anom = d_df["aam"] - d_df["climo"]
    smoothed = anom.rolling(window=window_days, center=True, min_periods=1).mean()
    smoothed.index = d_df["timestamp"]
    return smoothed


def five_point_tendency_with_edges(series: pd.Series, dt_seconds: float) -> pd.Series:
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


def compute_gwo_daily(aam_csv: str, climo_csv: str, out_csv: str, window_days: int = 5) -> None:
    # Stds from climo
    mr_std_abs, tend_std_abs = parse_climo_daily_stats(climo_csv, window_days=window_days)
    # Daily smoothed anomalies from actual series
    mr_anom = daily_anom_from_series(aam_csv, climo_csv, window_days=window_days)
    dmr_dt = five_point_tendency_with_edges(mr_anom, 86400.0)
    gwo1 = mr_anom / mr_std_abs
    gwo2 = dmr_dt / tend_std_abs
    # Phase-space orientation: x = standardized GWO2 (tendency), y = standardized GWO1 (anomaly)
    angle_deg = (np.degrees(np.arctan2(gwo1, gwo2)) + 360.0) % 360.0
    amp = np.sqrt(gwo1 ** 2 + gwo2 ** 2)
    phase = np.floor(angle_deg / 45.0).astype("Int64") + 1
    phase = phase.where(phase != 9, other=8)

    out = pd.DataFrame(
        {
            "date": mr_anom.index.normalize(),
            "GWO1": gwo1.values,
            "GWO2": gwo2.values,
            "amp": amp.values,
            "phase": phase.values,
        }
    ).dropna(subset=["GWO1", "GWO2"], how="any")
    out = out.drop_duplicates(subset=["date"]).sort_values("date")
    # Write a short note with climatological stds above the CSV header, then the table
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(f"# mr_std_kg_m2_s,{mr_std_abs}\n")
        f.write(f"# tend_std_kg_m2_s2,{tend_std_abs}\n")
    out.to_csv(out_csv, index=False, mode="a")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute daily GWO (phase 1-8) from AAM and climatology")
    p.add_argument("--aam-csv", default="aam_jan2016.csv", help="Actual AAM CSV with timestamp,aam_total_kg_m2_s")
    p.add_argument("--climo-csv", default="aam_climo.csv", help="Climatology CSV with date(YYYY-MM-DD-HH),aam")
    p.add_argument("--out-csv", default="gwo_daily.csv", help="Output daily GWO CSV")
    p.add_argument("--window-days", type=int, default=5, help="Centered smoothing window for MR anomalies")
    args = p.parse_args()
    compute_gwo_daily(args.aam_csv, args.climo_csv, args.out_csv, window_days=args.window_days)


if __name__ == "__main__":
    main()


