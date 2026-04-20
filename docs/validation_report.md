# CLARA Codebase Validation Report

**Date:** 2026-03-23
**Branch:** `feat/novelty-modules`
**Actual test count:** 328 collected, 1 collection error (see E1)
**Reviewer:** Automated static review via Claude

---

## A. Data Model

### A1. LPProblem fields
[PASS] `clara/model/problem.py` defines all required fields:
- `c`, `A`, `b` (numpy arrays, coerced in `__post_init__`)
- `var_names`, `constraint_names`
- `sense` ("maximize" | "minimize")
- `integer_vars`, `binary_vars` (sets)
- `upper_bounds`, `lower_bounds` (numpy arrays, defaulted in `__post_init__`)

[FAIL] `as_lp()` method is absent from `LPProblem`. The class has no serialization method by that name. (`problem.py` lines 10–64)

### A2. SolveState fields
[PASS] `clara/model/solve_state.py` provides all required fields:
- `optimal_value`, `is_optimal` (property), `status` (SolveStatus enum)
- `variables` (tuple of `VariableInfo` with `obj_coeff_range`)
- `sensitivity` (SensitivityRanges with `obj_coeff_ranges`, `rhs_ranges`)
- `basis_inverse` (Optional ndarray)
- `iteration_count`

### A3. ParameterChange and ChangeType
[PASS] `clara/reopt/types.py`:
- `ParameterChange` is a frozen dataclass with `change_type`, `delta_b`, `delta_c`.
- `ChangeType` enum contains: `TYPE_R`, `TYPE_C`, `TYPE_RC`, `TYPE_V`, `TYPE_X`, `TYPE_M`, `TYPE_A`, `TYPE_MULTI`. All eight required variants are present. (lines 29–47)

### A4. Frozen result dataclasses
[PASS] `AttributionResult` — `frozen=True` (line 175)
[PASS] `SensitivityRegion` — `frozen=True` (line 219)
[PASS] `ReoptResult` — `frozen=True` (line 139)
[PASS] `ParametricResult` — `frozen=True` (line 262)
[PASS] `DiffReport` — `frozen=True` (line 322)
[WARN] `MIPBoundResult` is not defined in any source file under `clara/`. A compiled `.pyc` for `mip_bound.py` exists in `__pycache__` but the source file is absent. Cannot validate frozen status. If this class is required, the source is missing.

### A5. `__init__.py` exports
[PASS] `clara/reopt/__init__.py` exports:
- `ChangeDetector`, `ImpactAnalyzer`, `Reoptimizer`, `DiffReporter`, `ChangeAttributor`, `SimultaneousRegionAnalyzer`

All six required public classes are present. (lines 3–8)

---

## B. Core Solver

### B1. Minimize negation in RevisedSimplex
[PASS] `clara/engine/simplex.py`:
- `__init__` (lines 76–79): when `problem.sense == "minimize"`, sets `self.c_full[:n] = -problem.c`.
- `_make_state` (lines 458–460): when `problem.sense == "minimize"`, un-negates: `optimal_value = -optimal_value`.

### B2. `_add_upper_bound_rows()`
[PASS] `clara/engine/simplex.py`:
- Static method `_add_upper_bound_rows()` exists (lines 96–122).
- Called in `__init__` at line 60: `A, b, con_names = self._add_upper_bound_rows(problem)`.

### B3. Big-M handling
[PASS] `clara/engine/simplex.py`:
- Negative RHS check at line 151 dispatches to `_solve_with_bigm(start_time)`.
- Artificials are added as columns with `-BIG_M` penalty, removed from the system after solving (lines 296–316).

### B4. Dual simplex
[PASS] `clara/engine/simplex.py`:
- `solve_dual()` exists (lines 321–330).
- `_is_dual_feasible()` exists (lines 404–414).

### B5. Branch-and-Bound
[PASS] `clara/engine/bnb.py`:
- LP relaxation solved at root node (line 141: `RevisedSimplex(problem).solve()`).
- Branching on most-fractional variable (lines 365–390).
- Incumbent update with pruning (lines 240–253, line 244: `queue.prune(incumbent_value)`).
- Bound pruning before and after node LP solve (lines 194–204, 226–237).

### B6. HiGHS sense-awareness
[PASS] `clara/engine/highs_backend.py`:
- Lines 53–56: `changeObjectiveSense(kMinimize)` when `problem.sense == "minimize"`, else `kMaximize`.

[FAIL] `clara/engine/highs_backend.py` ignores `problem.upper_bounds` and `problem.lower_bounds`. All variables are added with `h.addVar(0.0, highspy.kHighsInf)` (line 48), unconditionally. Problems with finite upper bounds or non-zero lower bounds will produce incorrect solutions from the HiGHS backend. (`highs_backend.py` line 48)

---

## C. Reoptimization Pipeline

### C1. ChangeDetector
[PASS] `clara/reopt/detector.py`:
- `TOLERANCE = 1e-10` (line 27).
- Structural changes (added/removed vars/constraints) classified before numeric changes (line 145).
- Returns `None` when no change detected (lines 105–106, 163).

### C2. ImpactAnalyzer — Oguz bound and inf delta handling
[PASS] `clara/reopt/analyzer.py`:
- Oguz formula implemented: `2 * delta / (1 + delta)` (line 181).
- `delta = float("inf")` when old coefficient is zero (line 176), mapped to `raw_bound = 1.0` (line 179).
- Wendell tightening: alpha computed from sensitivity ranges (lines 184–192), guards for `isinf(delta)` and `isinf(alpha)` (lines 194, 200).
- `c_j = 0` guard: `if abs(c_j) > 1e-10` before dividing (line 189). Division by zero prevented.

[WARN] Sensitivity check uses tolerance `1e-8` (lines 237, 250) while detector uses `1e-10`. Minor inconsistency, not a correctness issue.

### C3. Reoptimizer method dispatch
[PASS] `clara/reopt/reoptimizer.py` dispatches on `decision.recommended_method`:
- `"none"` → `_no_action`
- `"recompute"` → `_recompute`
- `"warm_start"` → `_warm_start` (which internally chooses primal or dual)
- `"parametric_lp"` → `_parametric_lp`
- else → `_scratch`

[WARN] The method string `"warm_start_dual"` is reported in `method_used` (line 226) but is not a recognized dispatch key in `reoptimize()`. The dispatch block (lines 56–65) does not handle `"warm_start_dual"` explicitly — it is an internal label set *after* dispatch inside `_warm_start`. This is not a bug (dispatch is correct) but the method string is misleading if passed back as a recommendation.

### C4. Parametric LP
[PASS] `clara/reopt/parametric.py`:
- `theta` runs from 0 to 1 (lines 82–95).
- Both primal breakpoints (`_find_primal_breakpoint`) and dual breakpoints (`_find_dual_breakpoint`) are computed each iteration.
- Pivots are performed at each breakpoint (lines 110–135).

### C5. DiffReport
[PASS] `clara/reopt/types.py` `DiffReport` includes:
- Variable and constraint change tuples.
- Bottleneck fields (`old_bottleneck`, `new_bottleneck`, `bottleneck_shifted`).
- Attribution section via optional `attribution` field.

[FAIL] `DiffReport.to_dict()` has dead code. The `return { ... }` statement ends at line 499, and the block that adds `d["attribution"]` (lines 500–513) is unreachable — it references an undeclared variable `d` and follows an already-returned dict literal. Attribution data is never included in the JSON output of `to_dict()`. (`clara/reopt/types.py` lines 499–513)

### C6. Attribution
[PASS] `clara/reopt/attribution.py`:
- `rhs_effect = yᵀΔb` (line 68).
- `obj_effect = Δcᵀx` (line 69).
- Interaction `Δc_Bᵀ B⁻¹ Δb` computed (lines 72–84).
- Shapley decomposition: `shapley_b = [(z_b - z_old) + (z_new - z_c)] / 2` (line 107), `shapley_c` (line 108).

### C7. SimultaneousRegion — Chebyshev norm
[PASS] `clara/reopt/sensitivity_region.py`:
- The Chebyshev center LP uses `norm_i = float(np.linalg.norm(H[i]))` (line 209), which is the L2 (Euclidean) norm. This is mathematically correct for inscribing a Euclidean ball in the constraint polyhedron.

[WARN] The polyhedron `H` is built from RHS perturbations only (lines 132–136: `dim = min(m, m_basis)`). Objective coefficient perturbations (Δc) are not included in the simultaneous region analysis, despite being part of the documented sensitivity region. The `_build_polyhedron` docstring acknowledges this limitation ("Simplified: only RHS perturbations").

---

## D. Edge Cases

### D1. Division by zero
[PASS] In `analyzer.py`, `c_j` guard prevents `/0` (line 189: `if abs(c_j) > 1e-10`).
[PASS] In `sensitivity_region.py`, `min_oat` guard prevents `/0` (line 63: `if min_oat > 1e-10 else 0.0`).
[PASS] In `attribution.py`, `delta_z` guard prevents `/0` in `nonlinearity` (line 123: `/ max(abs(delta_z), 1e-10)`).
[WARN] In `types.py` `AttributionResult.summary` (line 199) divides by `self.delta_z` without guarding for `delta_z == 0` inside the percent calculations (e.g., `self.rhs_effect / self.delta_z * 100`). This is inside an `if abs(self.delta_z) > 1e-10:` block (line 198), so the outer guard is present — [PASS].

### D2. Empty/None defense
[PASS] `SolveState` returns `SensitivityRanges({}, {})` on non-optimal status (`simplex.py` line 440).
[PASS] `Reoptimizer._recompute()` checks `B_inv is None` and falls back to `_scratch` (line 117).
[PASS] `ChangeAttributor.attribute()`: `delta_b` defaults to `np.zeros(m)` if `None` (line 57).

### D3. Numerical tolerances
[WARN] Mixed tolerances across modules:
- `PIVOT_TOL = 1e-10`, `OPTIMALITY_TOL = 1e-8` in `simplex.py` and `parametric.py`.
- `ChangeDetector.TOLERANCE = 1e-10`.
- `_sensitivity_check` uses `1e-10` for ignoring small deltas (line 230) but `1e-8` for range violation check (line 237).
- `DiffReporter.CHANGE_TOL = 1e-6`.
- Inconsistency is not critical but may cause edge cases where a "within range" delta flips status.

### D4. Iteration/node limits
[PASS] `simplex.py`: `MAX_ITERATIONS = 10_000` (line 34).
[PASS] `bnb.py`: `max_nodes` (default 10,000) and `max_time` (default 60s) (lines 112–116).
[PASS] `parametric.py`: `MAX_PIVOTS = 100` (line 34).

---

## E. Tests

### E1. Test file coverage per module
[PASS] Core modules have corresponding test files:
- `test_simplex.py` → `simplex.py`
- `test_bnb.py` → `bnb.py`
- `test_lp_parser.py` → `lp_parser.py`
- `test_mps_parser.py` → `mps_parser.py`
- `test_change_detector.py` → `detector.py`
- `test_impact_analyzer.py` → `analyzer.py`
- `test_reoptimizer.py` → `reoptimizer.py`
- `test_parametric_lp.py` → `parametric.py`
- `test_diff_report.py` → `diff_report.py`
- `test_attribution.py` → `attribution.py`
- `test_sensitivity_region.py` → `sensitivity_region.py`
- `test_dual_simplex.py` → `solve_dual()` / `_is_dual_feasible()`
- `test_cross_validation.py` → cross-validates internal vs HiGHS
- `test_cli.py` → `cli.py`

[FAIL] `tests/fixtures/test_albici.py` fails to **collect** with `ModuleNotFoundError: No module named 'scipy'`. The file imports `from scipy.optimize import linprog` at line 23. `scipy` is not listed as a dependency in `pyproject.toml`. 328 tests collect cleanly; this file contributes 0 because it errors out at import. CLAUDE.md claims "328 tests passing" — this is the exact collected count but the collection error means the Albici fixture suite is silently excluded from every run.

### E2. Golden values (Albici, Netlib)
[PASS] `tests/fixtures/test_albici.py` contains Albici et al. (2010) golden values (e.g., `EXPECTED_OPTIMAL = 33200/9`, reopt scenarios b1, b2, cost, columns, parametric). These tests are structurally correct but non-executable due to the scipy import failure (see E1).

[WARN] No Netlib golden-value tests were found in the test suite beyond the cross-validation runner benchmark script.

### E3. Edge cases
[WARN] `test_simplex.py` does not contain dedicated tests for `sense="minimize"`, finite `upper_bounds`, or explicit infeasibility scenarios. The file focuses entirely on the Albici base problem (maximize, no upper bounds). These paths are exercised in other test files (e.g., `test_dual_simplex.py` tests infeasibility triggering via negative RHS), but minimize and upper-bound coverage is sparse in the unit tests.

### E4. Test independence
[PASS] Tests use `@pytest.fixture` for problem setup; no shared mutable state observed across test classes.

---

## F. Benchmarks

### F1. match=✓ filtering in runners
[WARN] `benchmarks/scripts/run_scalability.py` does **not** read the `manifest.csv` generated by `generate_random_lp.py` and does not filter by `match=✓`. It generates problems from scratch inline (lines 46–59) using the same seed logic but runs all instances unconditionally. Other runners (`run_benchmark.py`, `run_reopt_benchmark.py`) do compute and record a `match` field but only for summary reporting, not pre-filtering.

### F2. upper_bounds in benchmark scripts
[PASS] `benchmarks/scripts/generate_random_lp.py`: `upper_bounds = x_feas * 3` at line 50. Written to `.lp` files via `write_lp_file` (lines 76–79). Passed to `LPProblem` (line 114).
[PASS] `benchmarks/scripts/run_scalability.py`: `upper_bounds = x_feas * 3` at line 57. Passed to `LPProblem` (line 59).

### F3. CSV filename consistency
[PASS] Cross-checked runner output filenames against `generate_all_tables.py` references:
- `run_cross_validation.py` → `exp1_cross_validation.csv` ✓
- `run_reopt_decision.py` → `exp2_reopt_decision.csv` ✓
- `run_scalability.py` → `exp6_scalability.csv` ✓

### F4. Multiprocessing safety
[PASS] `run_scalability.py`:
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` set to `"1"` before any imports (lines 1–4).
- All `clara.*` imports are inside `process_single()` (lines 39–43, 102–104), ensuring safe forking on macOS/Linux.

---

## G. CLI

### G1. `.lp` and `.mps` support
[PASS] `clara/cli.py` `_parse_file()` (lines 163–177):
- `if p.suffix.lower() == ".mps"`: routes to `read_mps()`.
- Otherwise: routes to `read_lp()`.
- All commands (`explain`, `solve`, `info`, `diff`) call `_parse_file()`, so `.mps` support is present for all commands.

### G2. `clara diff` full pipeline
[PASS] `clara/cli.py` `diff` command (lines 100–141):
- Imports and calls `ChangeDetector`, `ImpactAnalyzer`, `Reoptimizer`, `DiffReporter` in sequence.
- Runs full pipeline: detect → analyze → reoptimize → diff report.

---

## H. Documentation

### H1. CLAUDE.md test count vs actual
[FAIL] `CLAUDE.md` line 109 states: **"Test suite: 328 tests passing"**. The actual `pytest --co -q` output is `328 tests collected, 1 error` — the Albici fixture suite (`tests/fixtures/test_albici.py`) fails to collect due to `scipy` not being installed. The stated count of 328 matches the collected count but the collection error is not acknowledged and tests in that file never run.

### H2. pyproject.toml
[PASS] `pyproject.toml`:
- `version = "0.1.0"`.
- Dependencies: `numpy>=1.24`, `highspy>=1.7`, `click>=8.0`.
- Entry point: `clara = "clara.cli:main"`.
- Dev extras include `pytest`, `pytest-cov`, `hypothesis`, `ruff`.

[WARN] `scipy` is absent from `dependencies` and `dev` extras, yet `tests/fixtures/test_albici.py` imports it unconditionally. Either `scipy` should be added to dev dependencies or the import should be guarded with `pytest.importorskip("scipy")`.

[WARN] `pyproject.toml` declares `requires-python = ">=3.10"` but the runtime environment uses Python 3.9.12 (as reported by pytest). This is a version mismatch.

### H3. Type hints
[PASS] All reviewed modules use `from __future__ import annotations` and provide type hints on public methods. `reoptimizer.py`, `analyzer.py`, `attribution.py`, `sensitivity_region.py`, `parametric.py` all annotate return types and parameters.

---

## Summary

| Category | PASS | FAIL | WARN |
|----------|------|------|------|
| A. Data Model | 4 | 1 | 1 |
| B. Core Solver | 5 | 1 | 0 |
| C. Reoptimization Pipeline | 5 | 2 | 3 |
| D. Edge Cases | 6 | 0 | 1 |
| E. Tests | 3 | 2 | 2 |
| F. Benchmarks | 3 | 0 | 1 |
| G. CLI | 2 | 0 | 0 |
| H. Documentation | 2 | 2 | 3 |
| **Total** | **30** | **8** | **11** |

---

## Critical Issues (FAILs)

1. **[A1] `LPProblem.as_lp()` missing** — `clara/model/problem.py`: the `as_lp()` serialization method is not implemented. Any caller expecting this method will raise `AttributeError`.

2. **[B6] HiGHS backend ignores `upper_bounds` and `lower_bounds`** — `clara/engine/highs_backend.py` line 48: `h.addVar(0.0, highspy.kHighsInf)` is unconditional. Problems with finite upper bounds produce incorrect HiGHS results. The internal simplex correctly adds upper-bound rows via `_add_upper_bound_rows()`, but HiGHS silently discards these constraints.

3. **[C5] `DiffReport.to_dict()` dead code** — `clara/reopt/types.py` lines 499–513: the attribution block follows a `return` statement and is unreachable. Attribution data is never serialized to JSON, breaking the `to_json()` output for any `DiffReport` that has attribution data.

4. **[E1] `tests/fixtures/test_albici.py` collection error** — The file unconditionally imports `scipy` which is not installed and not in `pyproject.toml`. The entire Albici golden-value test class is silently skipped on every CI/local run.

5. **[E1/H1] CLAUDE.md test count misleading** — CLAUDE.md claims "328 tests passing" but the collection produces `328 tests collected, 1 error`. The Albici suite (≈30 tests) does not run, so the effective passing count is lower than advertised.

6. **[H1/H2] `scipy` missing from dependencies** — `pyproject.toml` does not declare `scipy` in `dev` extras, yet it is required by `tests/fixtures/test_albici.py`. Any fresh environment will fail to collect these tests.

7. **[H2] Python version mismatch** — `pyproject.toml` requires Python ≥ 3.10 but the environment runs Python 3.9.12. Code uses `from __future__ import annotations` which mitigates type annotation issues, but 3.9 support is not declared and `match` statements or 3.10+ stdlib features could silently fail.

8. **[C3/C5] Reoptimizer missing explicit `warm_start_dual` dispatch** — The method string `"warm_start_dual"` is set internally inside `_warm_start()` (line 226) as `method_used` in `ReoptResult`, but `reoptimize()` dispatch (lines 56–65) has no `elif method == "warm_start_dual"` branch. If an external caller passes `recommended_method="warm_start_dual"`, it falls through to `_scratch`. This is not triggered by the current `ImpactAnalyzer` but is a latent interface inconsistency.

---

*Report generated by automated static code review. No source files were modified.*
