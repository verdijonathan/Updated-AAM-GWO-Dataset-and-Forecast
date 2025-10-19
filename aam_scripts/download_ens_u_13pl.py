import os
from datetime import datetime, timedelta, timezone
from typing import Iterable

from ecmwf.opendata import Client


LEVELS = "1000/925/850/700/600/500/400/300/250/200/150/100/50"
STEPS = list(range(0, 361, 6))  # 0..360 h by 6


def try_sources(request: dict, sources: Iterable[str]) -> bool:
    for src in sources:
        try:
            # Use single-dict call (request includes target & dataset), robust across client versions
            Client(source=src).retrieve(request)
            print(f"Saved {request['target']} from {src}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Skip {request.get('date')} {request.get('time')} {src} -> {exc}")
    return False


def main() -> None:
    out_dir = os.path.join("D:", "nixon et al", "forecast_ens")
    os.makedirs(out_dir, exist_ok=True)

    # Clear previous GRIBs before each run
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        try:
            if os.path.isfile(p) and name.lower().endswith(".grib2"):
                os.remove(p)
        except Exception:
            pass

    today = datetime.now(timezone.utc).date()
    # ENS 15-day is for 00/12 UTC; 06/18 stop at +144
    candidates = [
        (today, "12"),
        (today, "00"),
        (today - timedelta(days=1), "12"),
        (today - timedelta(days=1), "00"),
    ]

    sources = ("ecmwf", "cdn", "azure")

    for d, t in candidates:
        # Control forecast (CF)
        cf_req = {
            "dataset": "ifs",
            "date": d.strftime("%Y-%m-%d"),
            "time": t,
            "stream": "enfo",
            "type": "cf",
            "step": STEPS,
            "param": "u",
            "levtype": "pl",
            "levelist": LEVELS,
            "target": os.path.join(out_dir, f"ens_cf_u_{d.strftime('%Y%m%d')}_{t}.grib2"),
        }
        if not try_sources(cf_req, sources):
            continue

        # Perturbed members (1..50) in a single file
        pf_req = {
            "dataset": "ifs",
            "date": d.strftime("%Y-%m-%d"),
            "time": t,
            "stream": "enfo",
            "type": "pf",
            "number": list(range(1, 51)),
            "step": STEPS,
            "param": "u",
            "levtype": "pl",
            "levelist": LEVELS,
            "target": os.path.join(out_dir, f"ens_pf_u_{d.strftime('%Y%m%d')}_{t}.grib2"),
        }
        try_sources(pf_req, sources)
        return

    print("No ENS run found.")


if __name__ == "__main__":
    main()

from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client

LEVELS = "1000/925/850/700/600/500/400/300/250/200/150/100/50"
STEPS  = list(range(0, 361, 6))  # 0..360 h by 6

def fetch(req):
    Client(source=req["source"]).retrieve(req)  # request-as-dict: includes dataset & target

def make_req(source, d, t, kind, target, number=None):
    r = {
        "dataset": "ifs",
        "target":  target,
        "date":    d.strftime("%Y-%m-%d"),
        "time":    t,
        "stream":  "enfo",      # ENS
        "type":    kind,        # "cf" (control) or "pf"
        "step":    STEPS,
        "param":   "u",
        "levtype": "pl",
        "levelist": LEVELS,
        "source":  source,
    }
    if number is not None:
        r["number"] = number
    return r

def main():
    today = datetime.now(timezone.utc).date()
    candidates = [(today,"00"),(today,"12"),(today - timedelta(days=1),"12"),(today - timedelta(days=1),"00")]
    for d,t in candidates:
        for src in ("ecmwf","cdn","azure"):
            try:
                # Control
                fetch(make_req(src, d, t, "cf",
                      fr"D:\nixon et al\forecast\ens_cf_u_{d.strftime('%Y%m%d')}_{t}.grib2"))
                # A few perturbed members (adjust range 1..50 as desired)
                for n in (1,2,3,4,5):
                    fetch(make_req(src, d, t, "pf",
                          fr"D:\nixon et al\forecast\ens_pf{n:02d}_u_{d.strftime('%Y%m%d')}_{t}.grib2",
                          number=n))
                print("Downloaded ENS 0–360 h set")
                return
            except Exception as e:
                print("Skip", d, t, src, "->", e)
    print("No ENS run found.")

if __name__ == "__main__":
    main()
