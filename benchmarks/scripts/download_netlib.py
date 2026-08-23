"""Download small Netlib LP instances for benchmarking.

Usage:
    python benchmarks/scripts/download_netlib.py

Downloads MPS files to benchmarks/netlib/mps/
"""

import sys
import urllib.request
from pathlib import Path

# Target instances (small enough for Internal Simplex)
_BASE = "https://raw.githubusercontent.com/ozy4dm/lp-data-netlib/main/mps_files"
INSTANCES = {
    "afiro": f"{_BASE}/afiro.mps",
    "adlittle": f"{_BASE}/adlittle.mps",
    "blend": f"{_BASE}/blend.mps",
    "sc50a": f"{_BASE}/sc50a.mps",
    "sc50b": f"{_BASE}/sc50b.mps",
    "sc105": f"{_BASE}/sc105.mps",
    "kb2": f"{_BASE}/kb2.mps",
    "share2b": f"{_BASE}/share2b.mps",
}

MEDIUM_INSTANCES = {
    name: f"{_BASE}/{name}.mps"
    for name in ["bandm", "scagr25", "degen2", "scsd8", "fffff800",
                 "scfxm2", "ship08l", "25fv47", "maros", "czprob",
                 "sctap2", "ship12l", "pilot4", "scrs8", "shell"]
}

OUTPUT_DIR = Path(__file__).parent.parent / "netlib" / "mps"
MEDIUM_DIR = Path(__file__).parent.parent / "netlib" / "mps_medium"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium", action="store_true",
                    help="also download the medium instance set")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.medium:
        MEDIUM_DIR.mkdir(parents=True, exist_ok=True)
        for name, url in MEDIUM_INSTANCES.items():
            dest = MEDIUM_DIR / f"{name}.mps"
            if dest.exists():
                continue
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  medium {name}: downloaded")
            except Exception as e:
                print(f"  medium {name}: FAILED ({e})")

    success = 0
    failed = []

    for name, url in INSTANCES.items():
        dest = OUTPUT_DIR / f"{name}.mps"
        if dest.exists():
            print(f"  {name}: already exists, skipping")
            success += 1
            continue

        print(f"  Downloading {name}...", end=" ")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"OK ({dest.stat().st_size} bytes)")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)

    print(f"\nDownloaded: {success}/{len(INSTANCES)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print("\nManual download: visit https://github.com/jump-dev/Netlib.jl/tree/master/data")
        print(f"Save .mps files to: {OUTPUT_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
