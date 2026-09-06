# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-09-06

### Fixed
- **RHS sensitivity ranges for non-binding constraints.** HiGHS row-bound
  ranging reports, for a basic (non-binding) row, the range of the row
  activity rather than the basis-preserving range of the right-hand
  side, and the value does not even contain the current rhs. The
  backend now computes exact rhs ranges from the reconstructed basis
  inverse (`x_B + B^-1[:, k] * delta >= 0`), which also matches HiGHS
  for binding rows. This affected the simultaneous sensitivity region
  (OAT bounding box), the simultaneity ratio, and the first screening
  stage of the impact analyzer. A regression test re-solves at the
  range endpoints (`tests/test_rhs_ranges_exact.py`).
- **Certified recompute.** The zero-pivot recompute path now verifies
  dual feasibility of the retained basis under the new objective (in
  addition to primal feasibility) and falls back to a warm start
  otherwise, so skipped re-solves are exact by construction.
- Simultaneity ratio uses RHS tolerances only (Definition 3 of the
  paper); objective tolerances enter the joint region analysis.
- Paper figure script draws the asymmetric OAT box.

## [0.3.0] - 2026-08-23

### Changed
- **HiGHS is now the single solver backend.** After each solve, CLARA
  reconstructs the basis inverse B^-1 of the augmented system
  [A_aug | I] from the optimal basis reported by HiGHS, so all
  downstream modules (sensitivity region, basis robustness radius,
  reoptimizer, parametric tracer, attribution) operate on exact solver
  artifacts at production-solver speed.
- Warm-start reoptimization now passes the old basis to HiGHS as an
  advanced starting basis, forcing the dual (Type R) or primal
  (Type C) simplex strategy chosen by the impact analyzer.
- The parametric tracer remains a CLARA-native routine operating on
  the retained basis factorization; it now supports minimize-sense
  problems and reports native-sense objectives/duals.
- Attribution and all SolveStates now use the problem's native sense
  throughout (no internal sign flips).
- `EngineType` is now `HIGHS` or `BASIS_ROUTINE` (recompute /
  parametric outputs).

### Removed
- Internal educational Revised Simplex engine (`clara.engine.simplex`)
  and the internal branch-and-bound engine (`clara.engine.bnb`).
  `from clara.engine import solve` replaces
  `from clara.engine.simplex import solve`.
- CLI `--engine internal` option; MIP files are now rejected with an
  explanatory error.

### Added
- `clara.engine.standard_form` with the shared augmented-system
  helpers previously private to the internal engine.
- `HiGHSBackend.solve(problem, initial_basis=..., simplex_strategy=...)`
  advanced-basis warm-start API.
- B^-1 reconstruction validation test suite
  (`tests/test_cross_validation.py`), replacing the internal-vs-HiGHS
  cross-validation.
- Medium-scale Netlib warm-start benchmark (solver-level timing via
  `changeRowBounds`/`changeColCost` + `setBasis`) and refreshed
  benchmark suite ported to the HiGHS backend.

### Fixed
- MPS parser: RHS sections whose lines omit the RHS vector name
  (e.g. Netlib `blend`) were parsed as all-zero right-hand sides;
  the parser now detects the unnamed-vector form and reads the
  values correctly.
- Package metadata: corrected author name and repository URLs.

## [0.2.2] - 2026-04-27

### Removed
- Roadmap section from README (was speculative; future capabilities
  will be announced in release notes when ready)

## [0.2.1] - 2026-04-27

### Changed
- Repositioned package framing around factual-counterfactual duality
  and reoptimization (manuscript in preparation)
- Updated package description and keywords for PyPI discoverability
- Updated README to reflect new positioning

### Added
- "explainable-optimization" and "counterfactual-explanations" keywords
- "Topic :: Scientific/Engineering :: Information Analysis" classifier

### Note
- No API changes — fully backward compatible

## [0.2.0] - 2026-04-27

### Added
- Objective change attribution via first-order decomposition and Shapley values
- Simultaneous sensitivity regions via Chebyshev center of basis-preserving polyhedron
- Factual--counterfactual duality: basis robustness d₀ as lower bound on counterfactual cost
- MIP opportunity cost bound (LP relaxation bound, Oguz comparison)
- Parametric LP solver for Type RC compound changes
- Dual simplex for Type R warm-start reoptimization
- Full reoptimization pipeline: change detection, impact analysis, warm-start selection
- Structured diff reports comparing two SolveStates
- Branch-and-bound MIP solver (educational, cold-start)
- MPS parser (free-format, Netlib compatible)
- `clara diff` CLI command for reoptimization comparison
- Numerical diagnostics: κ(B), basis robustness d₀, degeneracy count
- Basis indices stored in SolveState for reliable warm-start

### Changed
- Repositioned package framing around factual-counterfactual duality
  and reoptimization (paper submitted to EJOR)
- Updated package description and keywords for PyPI discoverability
- Updated README to reflect new positioning
- HiGHS backend now passes upper/lower bounds correctly
- Sensitivity analysis uses augmented system dimensions consistently

### Added (packaging)
- "explainable-optimization", "counterfactual-explanations" keywords
- "Topic :: Scientific/Engineering :: Information Analysis" classifier

### Fixed
- Warm-start basis extraction: uses stored basis_indices instead of heuristic
- Big-M Phase I check skipped for warm-started solves
- B⁻¹ dimension padding in attribution, region, and parametric modules
- DiffReport.to_dict() dead code preventing attribution output
- Perturbation generator preserves upper bounds in LP files
- Wendell tolerance uses original (not augmented) constraint dimensions
- Analyzer handles infinite delta values without RuntimeWarning

## [0.1.0] - 2025-01-15

### Added
- Revised Simplex solver with dense B⁻¹ and Bland's rule
- HiGHS backend for large problems
- Sensitivity analysis (OAT ranges for RHS and objective coefficients)
- Explainer module: binding analysis, variable contributions, sensitivity classification
- LP parser (CPLEX format)
- CLI: `clara explain`, `clara solve`, `clara info`
- Cross-validation framework (internal vs HiGHS)
