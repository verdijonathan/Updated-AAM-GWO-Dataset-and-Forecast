import argparse
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_aam_series(paths: List[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) or [p])
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])  # expects timestamp[,member],aam_total_kg_m2_s
        if "aam_total_kg_m2_s" not in df.columns:
            raise ValueError(f"Missing 'aam_total_kg_m2_s' in {f}")
        # If ensemble present, take member 0 (ERA5 is deterministic)
        if "member" in df.columns:
            df = df[df["member"] == 0].copy()
        dfs.append(df[["timestamp", "aam_total_kg_m2_s"]])
    if not dfs:
        raise ValueError("No input files found")
    out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return out


def fit_monthly_mapping(df13: pd.DataFrame, df37: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = (df13.rename(columns={"aam_total_kg_m2_s": "A13"})
                   .merge(df37.rename(columns={"aam_total_kg_m2_s": "A37"}), on="timestamp", how="inner"))
    merged = merged.dropna(subset=["A13", "A37"]).copy()
    merged["month"] = merged["timestamp"].dt.month

    rows = []
    metrics = []
    for m, g in merged.groupby("month"):
        if len(g) < 20:
            continue
        x = g["A13"].to_numpy()
        y = g["A37"].to_numpy()
        # Ordinary least squares (robust methods optional)
        b, a = np.polyfit(x, y, 1)  # y ≈ a + b*x
        rows.append({"month": int(m), "a": float(a), "b": float(b)})
        yhat = a + b * x
        resid = y - yhat
        r2 = 1.0 - float(np.var(resid)) / float(np.var(y)) if np.var(y) > 0 else 1.0
        bias = float(np.mean(resid))
        mae = float(np.mean(np.abs(resid)))
        metrics.append({"month": int(m), "n": int(len(g)), "r2": r2, "bias": bias, "mae": mae})

    coef_df = pd.DataFrame(rows).sort_values("month")
    met_df = pd.DataFrame(metrics).sort_values("month")
    return coef_df, met_df


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate monthly linear mapping AAM37 ≈ a(m) + b(m)·AAM13 from ERA5 time series")
    p.add_argument("--aam13", nargs="+", required=True, help="CSV paths/globs for AAM computed on 13 pressure levels")
    p.add_argument("--aam37", nargs="+", required=True, help="CSV paths/globs for AAM computed on 37 pressure levels")
    p.add_argument("--out-mapping", default="mapping_13to37.csv", help="Output CSV with columns month,a,b")
    p.add_argument("--out-report", default="mapping_metrics.csv", help="Output CSV with fit metrics per month")
    args = p.parse_args()

    df13 = load_aam_series(args.aam13)
    df37 = load_aam_series(args.aam37)

    coef_df, met_df = fit_monthly_mapping(df13, df37)
    coef_df.to_csv(args.out_mapping, index=False)
    met_df.to_csv(args.out_report, index=False)
    print(f"Saved mapping to {args.out_mapping} and metrics to {args.out_report}")


if __name__ == "__main__":
    main()


