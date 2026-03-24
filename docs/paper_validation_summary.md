# CLARA Paper-Code Validation Summary

Generated: 2026-03-24
Branch: feat/novelty-modules
Tests: 371 passing

## A-priority items

- **A1 (Chebyshev OAT box)**: VERIFIED — `_build_polyhedron()` builds H with primal block (-B⁻¹) + OAT box rows (I_m, -I_m from sensitivity ranges). Matches paper's 3m-row claim.
- **A2 (κ(B))**: IMPLEMENTED — `SolveState.condition_number` computed via `np.linalg.cond(B_inv)` in `simplex._make_state()`.
- **A3 (Degeneracy)**: IMPLEMENTED — `SolveState.degenerate_count` = count of x_B < 1e-8.
- **A4 (Bland's rule)**: VERIFIED — Default pivot selection in `_simplex_loop()` uses smallest index (Bland's rule). Comments at lines 167, 188.
- **A5 (d₀ basis robustness)**: IMPLEMENTED — `SolveState.basis_robustness_d0` = min(x_B[i] / ||B⁻¹_i||₂).
- **A6 (η nonlinearity)**: VERIFIED — `AttributionResult.nonlinearity` = |residual| / |Δz| at `attribution.py:127`.
- **A7 (Classification)**: VERIFIED — `sensitivity.py` computes bottleneck (max |dual|), fragile (min range_width), robust (max range_width).

## B-priority items (actual experimental values)

- **B1 (Instance statistics, n≤50 random LPs)**:
  - κ(B): median = 4.33 × 10¹, max = 1.50 × 10²
  - Degeneracy: 0% (random instances are non-degenerate by construction)
  - d₀: median = 0.53, min = 0.09
  - Note: Netlib instances will have different values (more degeneracy)

- **B3 (Attribution, Albici compound)**:
  - Δz = 6311.11
  - yᵀΔb = 677.78 (10.7%)
  - Δcᵀx = 4400.00 (69.7%)
  - Interaction = 400.00 (6.3%)
  - η = 0.132 (13.2% nonlinearity — basis changed)
  - Shapley: Δb = 1172.22, Δc = 5138.89 (sum = Δz ✓)

## C-priority items (formula verification)

- **C1 (Attribution theorem)**: VERIFIED — yᵀΔb + Δcᵀx + Δc_Bᵀ B⁻¹ Δb computed correctly. Residual indicates basis change.
- **C2 (Shapley split)**: VERIFIED — shapley_b + shapley_c = Δz within 1e-10.
- **C3 (Oguz bound)**: VERIFIED — 2δ/(1+δ) at `analyzer.py:181`. δ=inf handling at line 178.
- **C4 (Dual ratio test)**: VERIFIED — `_dual_simplex_loop()` at `simplex.py:327`. Correct leaving (most negative x_B) then entering (ratio rc_j/d_j for d_j < 0).
- **C5 (Parametric breakpoint)**: VERIFIED — `_find_primal_breakpoint()` and `_find_dual_breakpoint()` at `parametric.py`.

## Tests
- 371 tests passing, 0 failures
- New fields added: condition_number, degenerate_count, basis_robustness_d0

## Paper values to update
- κ(B) median: paper placeholder → actual 4.33 × 10¹ (for random LPs)
- Degeneracy: paper "15%" → actual 0% (random LPs; Netlib will differ)
- d₀ median: paper placeholder → actual 0.53
