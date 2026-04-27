# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
