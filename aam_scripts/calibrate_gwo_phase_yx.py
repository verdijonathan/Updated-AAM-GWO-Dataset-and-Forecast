import argparse
from typing import List

import numpy as np
import pandas as pd


def read_header_comments(path: str) -> List[str]:
    comments: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
            else:
                break
    return comments


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate GWO phase with x=GWO2, y=GWO1; phase 1 starts bottom-left and increases CCW")
    p.add_argument("--in-csv", required=True, help="Input CSV (date,GWO1,GWO2,amp,phase)")
    p.add_argument("--out-csv", required=True, help="Output CSV path")
    args = p.parse_args()

    comments = read_header_comments(args.in_csv)
    df = pd.read_csv(args.in_csv, comment="#", parse_dates=["date"])  # expects columns present
    if not {"date", "GWO1", "GWO2"}.issubset(df.columns):
        raise ValueError("Input must contain date,GWO1,GWO2")

    # x = GWO2, y = GWO1
    ang = (np.degrees(np.arctan2(df["GWO1"].astype(float).values, df["GWO2"].astype(float).values)) + 360.0) % 360.0
    # Phase 1 spans 180–225° (bottom-left), increase CCW: shift by -180°
    ang_shift = (ang - 180.0 + 360.0) % 360.0
    phase = np.floor(ang_shift / 45.0).astype(np.int64) + 1
    phase = np.where(phase == 9, 8, phase)
    df["phase"] = phase.astype(int)

    with open(args.out_csv, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(c + "\n")
    df.to_csv(args.out_csv, index=False, mode="a")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()


