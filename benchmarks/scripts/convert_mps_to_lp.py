"""Convert Netlib MPS files to LP format using HiGHS.

Usage:
    python benchmarks/scripts/convert_mps_to_lp.py

Input:  benchmarks/netlib/mps/*.mps
Output: benchmarks/netlib/lp/*.lp
"""

from pathlib import Path

MPS_DIR = Path(__file__).parent.parent / "netlib" / "mps"
LP_DIR = Path(__file__).parent.parent / "netlib" / "lp"


def main():
    import highspy

    LP_DIR.mkdir(parents=True, exist_ok=True)
    mps_files = sorted(MPS_DIR.glob("*.mps"))

    if not mps_files:
        print("No MPS files found. Run download_netlib.py first.")
        return

    converted = []
    skipped = []

    for mps_file in mps_files:
        name = mps_file.stem
        lp_file = LP_DIR / f"{name}.lp"

        print(f"  {name}: ", end="")
        try:
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            status = h.readModel(str(mps_file))
            if status != highspy.HighsStatus.kOk:
                print(f"SKIP (read error)")
                skipped.append(name)
                continue

            h.writeModel(str(lp_file))
            print(f"OK → {lp_file.name}")
            converted.append(name)
        except Exception as e:
            print(f"SKIP ({e})")
            skipped.append(name)

    print(f"\nConverted: {len(converted)}/{len(mps_files)}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")

    # Verify CLARA parser can read each LP file
    print("\nCLARA parser verification:")
    try:
        from clara.io.lp_parser import read_lp
        for name in converted:
            lp_file = LP_DIR / f"{name}.lp"
            try:
                problem = read_lp(lp_file)
                print(f"  {name}: {problem.num_variables} vars, {problem.num_constraints} constraints ✓")
            except Exception as e:
                print(f"  {name}: PARSE FAILED — {e}")
    except ImportError:
        print("  (clara not installed, skipping parser verification)")


if __name__ == "__main__":
    main()
