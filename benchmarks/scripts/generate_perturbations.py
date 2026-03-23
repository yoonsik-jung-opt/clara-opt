"""Generate systematic perturbations of base LP instances.

Usage:
    python benchmarks/scripts/generate_perturbations.py benchmarks/netlib/mps/afiro.mps
    python benchmarks/scripts/generate_perturbations.py --all-netlib
    python benchmarks/scripts/generate_perturbations.py --all-random

Output:
    benchmarks/instances/perturbations/{base_name}/*.lp + manifest.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"

SEEDS = [42, 123, 456]

# Perturbation specifications: (name, type, b_mag, c_mag)
PERTURBATION_SPECS = [
    # Type R
    ("R_small",  "R", 0.05, 0),
    ("R_medium", "R", 0.20, 0),
    ("R_large",  "R", 0.50, 0),
    ("R_single", "R_single", 0.20, 0),
    ("R_all",    "R_uniform", 0.10, 0),
    # Type C
    ("C_small",  "C", 0, 0.05),
    ("C_medium", "C", 0, 0.20),
    ("C_large",  "C", 0, 0.50),
    # Type RC
    ("RC_small",    "RC", 0.05, 0.05),
    ("RC_medium",   "RC", 0.20, 0.20),
    ("RC_large",    "RC", 0.50, 0.50),
    ("RC_asym_bc",  "RC", 0.05, 0.50),
    ("RC_asym_cb",  "RC", 0.50, 0.05),
]


def load_problem(filepath):
    """Load problem from .lp or .mps file."""
    path = Path(filepath)
    if path.suffix.lower() == ".mps":
        from clara.io.mps_parser import read_mps
        return read_mps(path)
    else:
        from clara.io.lp_parser import read_lp
        return read_lp(path)


def generate_perturbation(problem, spec_name, spec_type, b_mag, c_mag, seed):
    """Generate a perturbed LPProblem. Returns (new_c, new_b, delta_c, delta_b) or None."""
    rng = np.random.RandomState(seed)
    n = problem.num_variables
    m = len(problem.b)  # original constraint count

    c_orig = problem.c.copy()
    b_orig = problem.b.copy()
    delta_c = np.zeros(n)
    delta_b = np.zeros(m)

    # Type R perturbations
    if spec_type in ("R", "RC"):
        for i in range(m):
            if abs(b_orig[i]) > 1e-10:
                delta_b[i] = b_orig[i] * rng.uniform(-b_mag, b_mag)
            else:
                delta_b[i] = rng.uniform(-b_mag, b_mag)
    elif spec_type == "R_single":
        idx = rng.randint(m)
        if abs(b_orig[idx]) > 1e-10:
            delta_b[idx] = b_orig[idx] * rng.choice([-1, 1]) * b_mag
        else:
            delta_b[idx] = rng.choice([-1, 1]) * b_mag
    elif spec_type == "R_uniform":
        delta_b = b_orig * b_mag

    # Type C perturbations
    if spec_type in ("C", "RC"):
        for j in range(n):
            if abs(c_orig[j]) > 1e-10:
                delta_c[j] = c_orig[j] * rng.uniform(-c_mag, c_mag)
            else:
                delta_c[j] = rng.uniform(-c_mag, c_mag)

    new_b = b_orig + delta_b
    new_c = c_orig + delta_c

    # Validate: b_new should keep problem feasible (x=0 is feasible if b>=0 for <=)
    # For problems with equality constraints (paired rows), skip validation
    # Just ensure no extreme negative b
    if np.any(new_b < -1e6):
        return None

    return new_c, new_b, delta_c, delta_b


def write_perturbed_lp(filepath, problem, new_c, new_b, name):
    """Write perturbed problem as .lp file."""
    n = problem.num_variables
    m = len(new_b)
    var_names = problem.var_names

    with open(filepath, "w") as f:
        f.write(f"\\ Perturbed: {name}\n")
        sense = "Maximize" if problem.sense == "maximize" else "Minimize"
        f.write(f"{sense}\n obj:")
        for j in range(n):
            sign = " +" if j > 0 and new_c[j] >= 0 else " "
            f.write(f"{sign}{new_c[j]:.6f} {var_names[j]}")
        f.write("\n\nSubject To\n")
        for i in range(m):
            cname = problem.constraint_names[i] if i < len(problem.constraint_names) else f"c{i+1}"
            f.write(f" {cname}:")
            first = True
            for j in range(n):
                if abs(problem.A[i, j]) > 1e-12:
                    sign = " +" if not first and problem.A[i, j] >= 0 else " "
                    f.write(f"{sign}{problem.A[i, j]:.6f} {var_names[j]}")
                    first = False
            f.write(f" <= {new_b[i]:.6f}\n")
        f.write("\nEnd\n")


def process_base(filepath):
    """Generate all perturbations for a base instance."""
    problem = load_problem(filepath)
    base_name = Path(filepath).stem

    out_dir = PERTURBATIONS_DIR / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    generated = 0

    for spec_name, spec_type, b_mag, c_mag in PERTURBATION_SPECS:
        for seed in SEEDS:
            pert = generate_perturbation(problem, spec_name, spec_type, b_mag, c_mag, seed)
            if pert is None:
                continue

            new_c, new_b, delta_c, delta_b = pert
            inst_name = f"{base_name}_{spec_name}_s{seed}"
            lp_path = out_dir / f"{inst_name}.lp"

            write_perturbed_lp(lp_path, problem, new_c, new_b, inst_name)
            generated += 1

            results.append({
                "instance": inst_name,
                "base": base_name,
                "perturbation": spec_name,
                "type": spec_type.replace("_single", "").replace("_uniform", ""),
                "b_magnitude": b_mag,
                "c_magnitude": c_mag,
                "seed": seed,
                "delta_b_max": f"{np.max(np.abs(delta_b)):.6f}",
                "delta_c_max": f"{np.max(np.abs(delta_c)):.6f}",
            })

    # Write manifest
    if results:
        manifest = out_dir / "manifest.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"  {base_name}: {generated} perturbations → {out_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate LP perturbations")
    parser.add_argument("filepath", nargs="?", help="Base LP/MPS file")
    parser.add_argument("--all-netlib", action="store_true")
    parser.add_argument("--all-random", action="store_true")
    args = parser.parse_args()

    total = 0

    if args.filepath:
        total += process_base(args.filepath)

    if args.all_netlib:
        # Skip blend (Internal Simplex can't solve it)
        skip = {"blend"}
        for f in sorted(NETLIB_DIR.glob("*.mps")):
            if f.stem in skip:
                continue
            total += process_base(f)

    if args.all_random:
        # Read manifest to find solvable instances
        manifest = RANDOM_DIR / "manifest.csv"
        if manifest.exists():
            with open(manifest) as f:
                for row in csv.DictReader(f):
                    if row["match"] == "✓":
                        lp_path = RANDOM_DIR / f"{row['instance']}.lp"
                        if lp_path.exists():
                            total += process_base(lp_path)
        else:
            print("No random manifest found. Run generate_random_lp.py first.")

    if not args.filepath and not args.all_netlib and not args.all_random:
        # Default: Netlib only
        print("Generating perturbations for Netlib instances...")
        skip = {"blend"}
        for f in sorted(NETLIB_DIR.glob("*.mps")):
            if f.stem in skip:
                continue
            total += process_base(f)

    print(f"\nTotal perturbations generated: {total}")


if __name__ == "__main__":
    main()
