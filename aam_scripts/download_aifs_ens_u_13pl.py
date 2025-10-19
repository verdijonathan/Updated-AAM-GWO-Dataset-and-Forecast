import argparse
import os
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

from ecmwf.opendata import Client


PRESSURE_LEVELS_13 = "1000/925/850/700/600/500/400/300/250/200/150/100/50"


def parse_member_list(members: Sequence[str]) -> List[int]:
    if not members:
        return list(range(1, 51))
    out: List[int] = []
    for token in members:
        token = token.strip()
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            out.extend(list(range(min(start, end), max(start, end) + 1)))
        else:
            out.append(int(token))
    # de-dup and sort
    return sorted({m for m in out if 1 <= m <= 50})


def try_retrieve(request: dict, sources: Iterable[str], *, dataset: str) -> bool:
    for source in sources:
        try:
            # model/dataset is part of the request ('model': 'aifs'|'ifs')
            Client(source=source).retrieve(request)
            print(f"Saved {request['target']} from {dataset}/{source}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Skip {request.get('date')} {request.get('time')} {dataset}/{source} -> {exc}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download ECMWF AIFS ensemble u-wind at 13 PL levels (00Z daily, 6-hourly steps). "
            "Tries multiple Open Data sources and dataset identifiers ('aifs-ens', 'aifs')."
        )
    )
    parser.add_argument("--date", help="Forecast init date (YYYY-MM-DD), default=today UTC")
    parser.add_argument(
        "--members",
        nargs="*",
        default=[],
        help="Ensemble member list (e.g., 1-50 or 1 2 3). Default: 1-50",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("D:", "nixon et al", "forecast_aifs_ens"),
        help="Output directory for GRIB2 files",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=360,
        help="Maximum forecast step in hours (multiple of 6). Default: 360",
    )
    parser.add_argument(
        "--skip-ef",
        action="store_true",
        help="Skip aggregated ensemble (ef) product and go straight to cf/pf",
    )
    parser.add_argument(
        "--pf-split",
        action="store_true",
        help="Download perturbed members as separate files (one GRIB per member)",
    )
    parser.add_argument(
        "--skip-cf",
        action="store_true",
        help="Skip control forecast (cf) and proceed directly to pf",
    )
    parser.add_argument(
        "--source-order",
        default="aws,ecmwf,azure",
        help="Comma-separated sources to try in order (aws,ecmwf,azure)",
    )
    parser.add_argument(
        "--step-chunk-hours",
        type=int,
        default=0,
        help="If >0, split requests into chunks of this many hours (multiples of 6)",
    )
    args = parser.parse_args()

    init_date = (
        datetime.now(timezone.utc).date() if not args.date else datetime.strptime(args.date, "%Y-%m-%d").date()
    )
    init_time = "00"  # AIFS ENS daily 00Z
    steps = list(range(0, int(args.max_step) + 1, 6))
    numbers = parse_member_list(args.members)

    os.makedirs(args.out_dir, exist_ok=True)

    dataset_candidates = ("aifs-ens", "aifs")
    source_candidates = tuple([s.strip() for s in args.source_order.split(",") if s.strip()])

    def chunk_step_ranges(all_steps: List[int], chunk_hours: int) -> List[List[int]]:
        if chunk_hours is None or chunk_hours <= 0:
            return [all_steps]
        chunk_len = max(1, int(chunk_hours // 6))  # steps are 6-hour increments
        chunks: List[List[int]] = []
        for i in range(0, len(all_steps), chunk_len):
            chunks.append(all_steps[i : i + chunk_len])
        return chunks

    # Try aggregated ensemble file (EF) first: all members together (unless skipped)
    if not args.skip_ef:
        ef_ok = False
        for dataset in dataset_candidates:
            step_groups = chunk_step_ranges(steps, args.step_chunk_hours)
            group_success = 0
            for group in step_groups:
                start_h = group[0]
                end_h = group[-1]
                ef_request = {
                    "date": init_date.strftime("%Y-%m-%d"),
                    "time": init_time,
                    "stream": "enfo",
                    "type": "ef",
                    "step": group,
                    "param": "u",
                    "levtype": "pl",
                    "levelist": PRESSURE_LEVELS_13,
                    "model": dataset,  # ecmwf-opendata expects 'model' ('aifs'|'ifs')
                    "target": os.path.join(
                        args.out_dir,
                        f"aifs_ens_ef_u_{init_date.strftime('%Y%m%d')}_{init_time}_{start_h:03d}-{end_h:03d}.grib2",
                    ),
                }
                if try_retrieve(ef_request, source_candidates, dataset=dataset):
                    group_success += 1
                else:
                    print(f"EF chunk {start_h}-{end_h}h failed for {dataset}")
            if group_success == len(step_groups):
                ef_ok = True
                break

        if ef_ok:
            print("Downloaded AIFS ENS aggregated (ef) u-wind in chunks.")
            return

    # Control forecast (CF)
    cf_ok = False
    if args.skip_cf:
        print("Skipping control forecast (cf) as requested.")
    else:
        for dataset in dataset_candidates:
            step_groups = chunk_step_ranges(steps, args.step_chunk_hours)
            successes = 0
            for group in step_groups:
                start_h = group[0]
                end_h = group[-1]
                cf_request = {
                    "date": init_date.strftime("%Y-%m-%d"),
                    "time": init_time,
                    "stream": "enfo",
                    "type": "cf",
                    "step": group,
                    "param": "u",
                    "levtype": "pl",
                    "levelist": PRESSURE_LEVELS_13,
                    "model": dataset,
                    "target": os.path.join(
                        args.out_dir, f"aifs_ens_cf_u_{init_date.strftime('%Y%m%d')}_{init_time}_{start_h:03d}-{end_h:03d}.grib2"
                    ),
                }
                if try_retrieve(cf_request, source_candidates, dataset=dataset):
                    successes += 1
            if successes == len(step_groups):
                cf_ok = True
                break

        if not cf_ok:
            print("Could not find AIFS ENS control forecast for the requested run.")

    # Perturbed members (PF)
    pf_ok = False
    for dataset in dataset_candidates:
        step_groups = chunk_step_ranges(steps, args.step_chunk_hours)
        if args.pf_split:
            successes = 0
            for n in numbers:
                for group in step_groups:
                    start_h = group[0]
                    end_h = group[-1]
                    pf_request = {
                        "date": init_date.strftime("%Y-%m-%d"),
                        "time": init_time,
                        "stream": "enfo",
                        "type": "pf",
                        "number": n,
                        "step": group,
                        "param": "u",
                        "levtype": "pl",
                        "levelist": PRESSURE_LEVELS_13,
                        "model": dataset,
                        "target": os.path.join(
                            args.out_dir,
                            f"aifs_ens_pf{n:02d}_u_{init_date.strftime('%Y%m%d')}_{init_time}_{start_h:03d}-{end_h:03d}.grib2",
                        ),
                    }
                    if try_retrieve(pf_request, source_candidates, dataset=dataset):
                        successes += 1
            if successes > 0:
                pf_ok = True
                break
        else:
            successes = 0
            for group in step_groups:
                start_h = group[0]
                end_h = group[-1]
                pf_request = {
                    "date": init_date.strftime("%Y-%m-%d"),
                    "time": init_time,
                    "stream": "enfo",
                    "type": "pf",
                    "number": numbers,
                    "step": group,
                    "param": "u",
                    "levtype": "pl",
                    "levelist": PRESSURE_LEVELS_13,
                    "model": dataset,
                    "target": os.path.join(
                        args.out_dir,
                        f"aifs_ens_pf_u_{init_date.strftime('%Y%m%d')}_{init_time}_{start_h:03d}-{end_h:03d}.grib2",
                    ),
                }
                if try_retrieve(pf_request, source_candidates, dataset=dataset):
                    successes += 1
            if successes == len(step_groups):
                pf_ok = True
                break

    if not pf_ok:
        print("Could not find AIFS ENS perturbed members for the requested run.")

    if cf_ok or pf_ok:
        print("Finished AIFS ENS u-wind download.")
    else:
        print("No AIFS ENS data found across tried datasets/sources.")


if __name__ == "__main__":
    main()


