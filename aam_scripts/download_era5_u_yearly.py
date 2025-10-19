import os
import time
import argparse
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi


PRESSURE_LEVELS_ALL: List[str] = [
    "1", "2", "3", "5", "7",
    "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200",
    "225", "250", "300", "350", "400",
    "450", "500", "550", "600", "650",
    "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950",
    "975", "1000",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def retrieve_with_retry(client: cdsapi.Client, request: dict, target: str, max_attempts: int = 8) -> None:
    """Retrieve with exponential backoff retries (useful for CDS queue/timeouts)."""
    attempt = 0
    wait_seconds = 30
    while True:
        try:
            client.retrieve("reanalysis-era5-pressure-levels", request, target)
            return
        except Exception as exc:  # broad except is intentional around network/HTTP/CDS errors
            attempt += 1
            if attempt >= max_attempts:
                raise
            print(
                f"Download failed ({type(exc).__name__}: {exc}); retrying in {wait_seconds}s "
                f"(attempt {attempt}/{max_attempts})..."
            )
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2, 1800)  # cap at 30 minutes


def build_request(
    year: int,
    months: List[int],
    times_utc: List[str],
    grid: Optional[Tuple[float, float]] = None,
    area: Optional[List[float]] = None,
    pressure_levels: Optional[List[str]] = None,
) -> dict:
    """Build a CDS API request for a list of months within a year.

    CDS accepts lists for month/day; invalid days are ignored per month.
    """
    month_list = [f"{m:02d}" for m in months]
    days = [f"{d:02d}" for d in range(1, 32)]
    req = {
        "product_type": "reanalysis",
        "variable": ["u_component_of_wind"],
        "pressure_level": pressure_levels or PRESSURE_LEVELS_ALL,
        "year": f"{year:04d}",
        "month": month_list,
        "day": days,
        "time": times_utc,  # ["00:00","06:00","12:00","18:00"]
        "format": "netcdf",
    }
    if grid is not None:
        req["grid"] = [float(grid[0]), float(grid[1])]
    if area is not None:
        # Expect [N, W, S, E]
        req["area"] = area
    return req


def download_month_chunk(
    year: int,
    months: List[int],
    times_utc: List[str],
    output_dir: str,
    grid: Optional[Tuple[float, float]],
    area: Optional[List[float]],
    max_attempts: int,
    pressure_levels: Optional[List[str]] = None,
) -> Tuple[int, str, bool, Optional[str]]:
    """Download a chunk of months within a year to one NetCDF. Returns (year, label, success, error)."""
    if not months:
        return year, "", True, None
    label = f"M{months[0]:02d}-{months[-1]:02d}"
    target_file = os.path.join(output_dir, f"era5_u_pl_{year:04d}_{label}.nc")
    if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
        print(f"Exists, skip: {target_file}")
        return year, label, True, None
    req = build_request(
        year=year,
        months=months,
        times_utc=times_utc,
        grid=grid,
        area=area,
        pressure_levels=pressure_levels,
    )
    print(f"Requesting {year:04d} {label} -> {target_file}")
    try:
        client = cdsapi.Client()
        retrieve_with_retry(client, req, target_file, max_attempts=max_attempts)
        print(f"Saved: {target_file}")
        return year, label, True, None
    except Exception as exc:  # noqa: BLE001
        return year, label, False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ERA5 u-wind on pressure levels (chunked by months per file, sequential)"
    )
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument(
        "--times",
        nargs="+",
        default=["00:00", "06:00", "12:00", "18:00"],
        help="UTC times to retrieve (e.g., 00:00 06:00 12:00 18:00)",
    )
    parser.add_argument(
        "--output-dir",
        default=r"E:\era5\uwind",
        help="Output directory for downloads",
    )
    parser.add_argument(
        "--grid",
        type=float,
        nargs=2,
        metavar=("DX", "DY"),
        help="Optional output grid spacing in degrees (e.g., --grid 1 1)",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        metavar=("N", "W", "S", "E"),
        help="Optional area subset [N W S E]",
    )
    parser.add_argument("--max-attempts", type=int, default=8, help="Max retry attempts per file")
    parser.add_argument("--months-per-file", type=int, default=6, help="Number of months per output file (e.g., 6 for half-year)")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4)), help="Parallel downloads across chunks/years")
    parser.add_argument("--start-month", type=int, default=1, help="Start month for the first year (e.g., 5 to start in May of the first year)")
    args = parser.parse_args()

    start_year = args.start_year
    end_year = args.end_year
    times_utc = list(args.times)
    output_dir = args.output_dir
    grid = tuple(args.grid) if args.grid is not None else None
    area = list(args.area) if args.area is not None else None
    max_attempts = max(1, int(args.max_attempts))

    ensure_dir(output_dir)

    months_per_file = max(1, int(args.months_per_file))
    workers = max(1, int(args.workers))
    start_month_first_year = max(1, min(12, int(args.start_month)))

    # Build all chunk tasks first (respecting start-month for first year)
    tasks: List[Tuple[int, List[int]]] = []
    for y in range(start_year, end_year + 1):
        first_month = start_month_first_year if y == start_year else 1
        m = first_month
        while m <= 12:
            months = list(range(m, min(m + months_per_file - 1, 12) + 1))
            if months:
                tasks.append((y, months))
            m += months_per_file

    print(f"Planned {len(tasks)} requests across years {start_year}-{end_year} (months per file={months_per_file}, workers={workers}).")

    failures: List[Tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_month_chunk,
                y,
                months,
                times_utc,
                output_dir,
                grid,
                area,
                max_attempts,
                PRESSURE_LEVELS_ALL,
            )
            for (y, months) in tasks
        ]

        for fut in as_completed(futures):
            y, label, success, err = fut.result()
            if not success:
                print(f"FAILED {y:04d} {label}: {err}")
                failures.append((y, err or "unknown"))

    if failures:
        print(f"Completed with {len(failures)} failures. You can re-run; existing files are skipped.")
    else:
        print("All requested years downloaded successfully.")


if __name__ == "__main__":
    main()


