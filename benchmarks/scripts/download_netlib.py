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

OUTPUT_DIR = Path(__file__).parent.parent / "netlib" / "mps"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
