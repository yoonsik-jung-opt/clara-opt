"""Simultaneous sensitivity region analysis.

Computes the largest inscribed ball (Chebyshev center) of the
polyhedron where simultaneous parameter changes preserve the basis.

One-at-a-time sensitivity ranges overestimate: a change within range
for each parameter individually may still break the basis when applied
simultaneously. The Chebyshev radius quantifies this gap.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from clara.model.problem import LPProblem
from clara.model.solve_state import BasisStatus, SolveState
from clara.reopt.types import SensitivityRegion


class SimultaneousRegionAnalyzer:
    """Compute the simultaneous sensitivity region and Chebyshev center."""

    def analyze(
        self,
        state: SolveState,
        problem: LPProblem,
        projection_pairs: Optional[list[tuple[int, int]]] = None,
    ) -> SensitivityRegion:
        """Analyze simultaneous sensitivity region.

        Args:
            state: Solved state with B⁻¹ and sensitivity ranges.
            problem: The LP problem.
            projection_pairs: Optional pairs of parameter indices for 2D projections.
        """
        n = problem.num_variables
        m = problem.num_constraints

        # One-at-a-time tolerances from existing sensitivity ranges
        oat_rhs = self._extract_oat_rhs(state, problem)
        oat_obj = self._extract_oat_obj(state, problem)

        all_oat = list(oat_rhs.values()) + list(oat_obj.values())
        finite_oat = [v for v in all_oat if v < 1e20]
        min_oat = min(finite_oat) if finite_oat else float("inf")

        # Build polyhedron Hδ ≤ h
        if state.basis_inverse is None:
            return SensitivityRegion(
                chebyshev_radius=0.0, chebyshev_center=None,
                min_oat_tolerance=min_oat, simultaneity_ratio=0.0,
                oat_rhs_tolerances=oat_rhs, oat_obj_tolerances=oat_obj,
            )

        H, h, param_names = self._build_polyhedron(state, problem)

        # Chebyshev center
        radius, center = self._chebyshev_center(H, h)

        ratio = radius / min_oat if min_oat > 1e-10 else 0.0

        # Projections
        projections = None
        if projection_pairs and radius > 0:
            projections = {}
            for i1, i2 in projection_pairs:
                verts = self._project_2d(H, h, i1, i2)
                if verts:
                    projections[(i1, i2)] = verts

        return SensitivityRegion(
            chebyshev_radius=radius,
            chebyshev_center=tuple(center) if center is not None else None,
            min_oat_tolerance=min_oat,
            simultaneity_ratio=ratio,
            oat_rhs_tolerances=oat_rhs,
            oat_obj_tolerances=oat_obj,
            projections=projections,
        )

    def _extract_oat_rhs(self, state: SolveState, problem: LPProblem) -> dict[str, float]:
        """One-at-a-time RHS tolerances from sensitivity ranges."""
        result = {}
        for i, con in enumerate(state.constraints):
            if i >= problem.num_constraints:
                break
            name = con.name
            if name in state.sensitivity.rhs_ranges:
                lo, hi = state.sensitivity.rhs_ranges[name]
                b_i = con.rhs
                tol_down = abs(b_i - lo) if not math.isinf(lo) else float("inf")
                tol_up = abs(hi - b_i) if not math.isinf(hi) else float("inf")
                result[name] = min(tol_down, tol_up)
        return result

    def _extract_oat_obj(self, state: SolveState, problem: LPProblem) -> dict[str, float]:
        """One-at-a-time objective coefficient tolerances."""
        result = {}
        for j, var in enumerate(state.variables):
            if j >= problem.num_variables:
                break
            name = var.name
            if name in state.sensitivity.obj_coeff_ranges:
                lo, hi = state.sensitivity.obj_coeff_ranges[name]
                c_j = problem.c[j]
                tol_down = abs(c_j - lo) if not math.isinf(lo) else float("inf")
                tol_up = abs(hi - c_j) if not math.isinf(hi) else float("inf")
                result[name] = min(tol_down, tol_up)
        return result

    def _build_polyhedron(
        self, state: SolveState, problem: LPProblem
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build Hδ ≤ h polyhedron for simultaneous changes.

        δ = [Δb₁,...,Δbₘ, Δc₁,...,Δcₙ]

        Primal block: B⁻¹ Δb component ≤ x_B[i] for each basic row
                     -B⁻¹ Δb component ≤ x_B[i] (non-negativity of new x_B)
                     → -B⁻¹[:,i] @ Δb ≤ x_B[i] for slope < 0
        Simplified: for each row i of B⁻¹, the change in x_B[i] must keep it ≥ 0.
        """
        B_inv = state.basis_inverse
        m_basis = B_inv.shape[0]
        n = problem.num_variables
        m = problem.num_constraints

        x_B = B_inv @ problem.b[:m_basis]

        # Parameter vector: [Δb₁..Δbₘ_basis]
        # (Simplified: only RHS perturbations for the polyhedron)
        # Full version would include Δc but that requires dual feasibility rows
        dim = min(m, m_basis)

        rows_H = []
        rows_h = []

        # Primal feasibility: B⁻¹ @ (b + Δb) ≥ 0
        # → -B⁻¹ @ Δb ≤ x_B
        for i in range(m_basis):
            row = np.zeros(dim)
            for j in range(dim):
                row[j] = -B_inv[i, j]
            rows_H.append(row)
            rows_h.append(float(x_B[i]))

        # Also bound from above using OAT ranges for each b_i
        # This ensures the polyhedron is bounded
        for j in range(dim):
            cname = problem.constraint_names[j] if j < len(problem.constraint_names) else ""
            if cname in state.sensitivity.rhs_ranges:
                lo, hi_val = state.sensitivity.rhs_ranges[cname]
                b_j = float(problem.b[j])
                # Δb_j ≤ hi - b_j
                if not math.isinf(hi_val):
                    row_up = np.zeros(dim)
                    row_up[j] = 1.0
                    rows_H.append(row_up)
                    rows_h.append(hi_val - b_j)
                # -Δb_j ≤ b_j - lo
                if not math.isinf(lo):
                    row_lo = np.zeros(dim)
                    row_lo[j] = -1.0
                    rows_H.append(row_lo)
                    rows_h.append(b_j - lo)

        if not rows_H:
            return np.zeros((1, dim)), np.ones(1), []

        H = np.array(rows_H)
        h = np.array(rows_h)
        param_names = [problem.constraint_names[i] if i < len(problem.constraint_names)
                       else f"b{i}" for i in range(dim)]

        return H, h, param_names

    def _chebyshev_center(
        self, H: np.ndarray, h: np.ndarray
    ) -> tuple[float, Optional[np.ndarray]]:
        """Find Chebyshev center via LP: max ε s.t. H@δ + ||H_i||*ε ≤ h."""
        try:
            import highspy
        except ImportError:
            return 0.0, None

        n_rows, dim = H.shape
        # Variables: [δ₁,...,δ_dim, ε]
        n_vars = dim + 1

        hi = highspy.Highs()
        hi.setOptionValue("output_flag", False)

        # Add variables: δ free, ε ≥ 0
        for j in range(dim):
            hi.addVar(-highspy.kHighsInf, highspy.kHighsInf)
        hi.addVar(0.0, highspy.kHighsInf)  # ε

        # Objective: maximize ε
        for j in range(dim):
            hi.changeColCost(j, 0.0)
        hi.changeColCost(dim, -1.0)  # minimize -ε = maximize ε
        hi.changeObjectiveSense(highspy.ObjSense.kMinimize)

        # Constraints: H_i @ δ + ||H_i||₁ * ε ≤ h_i
        for i in range(n_rows):
            norm_i = float(np.linalg.norm(H[i]))
            if norm_i < 1e-12:
                continue
            idx = list(range(dim)) + [dim]
            vals = list(H[i]) + [norm_i]
            hi.addRow(-highspy.kHighsInf, float(h[i]), len(idx), idx, vals)

        hi.run()

        if hi.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return 0.0, None

        sol = hi.getSolution()
        center = np.array(sol.col_value[:dim])
        radius = sol.col_value[dim]

        return max(float(radius), 0.0), center

    def _project_2d(
        self, H: np.ndarray, h: np.ndarray, i1: int, i2: int
    ) -> list[tuple[float, float]]:
        """Project polyhedron onto 2D plane (parameters i1, i2)."""
        try:
            import highspy
        except ImportError:
            return []

        n_rows, dim = H.shape
        if i1 >= dim or i2 >= dim:
            return []

        vertices = []
        n_dirs = 36
        for k in range(n_dirs):
            theta = 2 * math.pi * k / n_dirs
            ct, st = math.cos(theta), math.sin(theta)

            hi = highspy.Highs()
            hi.setOptionValue("output_flag", False)
            for j in range(dim):
                hi.addVar(-highspy.kHighsInf, highspy.kHighsInf)

            # Objective: maximize cos(θ)*δ_i1 + sin(θ)*δ_i2
            for j in range(dim):
                cost = 0.0
                if j == i1:
                    cost = -ct
                elif j == i2:
                    cost = -st
                hi.changeColCost(j, cost)
            hi.changeObjectiveSense(highspy.ObjSense.kMinimize)

            for i in range(n_rows):
                idx = [j for j in range(dim) if abs(H[i, j]) > 1e-12]
                vals = [float(H[i, j]) for j in idx]
                hi.addRow(-highspy.kHighsInf, float(h[i]), len(idx), idx, vals)

            hi.run()
            if hi.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                sol = hi.getSolution()
                vertices.append((sol.col_value[i1], sol.col_value[i2]))

        return vertices
