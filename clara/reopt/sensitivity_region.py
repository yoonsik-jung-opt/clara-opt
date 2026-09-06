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

        # The RHS-only region S is compared against RHS OAT tolerances
        # only (Definition 3 of the paper); objective tolerances enter the
        # joint analysis in analyze_joint.
        all_oat = list(oat_rhs.values())
        # Exclude zero tolerances (degenerate constraints) and infinities
        positive_oat = [v for v in all_oat if 1e-10 < v < 1e20]
        min_oat = min(positive_oat) if positive_oat else float("inf")

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

    def analyze_joint(
        self, state: SolveState, problem: LPProblem,
    ) -> dict:
        """Compute joint (Δb, Δc) Chebyshev region.

        S+ = { (δ_b, δ_c) : -B⁻¹ δ_b ≤ x_B,  -G δ_c ≤ c̄,  box constraints }

        Returns dict with chebyshev_radius_joint, chebyshev_radius_b_only, etc.
        """
        n = problem.num_variables
        m = problem.num_constraints

        if state.basis_inverse is None:
            return {"chebyshev_radius_joint": 0.0, "chebyshev_radius_b_only": 0.0, "ratio": 0.0}

        B_inv = state.basis_inverse
        import clara.engine.standard_form as standard_form
        A_aug, b_aug, _ = standard_form.add_upper_bound_rows(problem)
        m_aug = A_aug.shape[0]
        x_B = B_inv @ b_aug

        # Get basis indices (original n vars only)
        if state.basis_indices is not None:
            basis = list(state.basis_indices)
        else:
            return {"chebyshev_radius_joint": 0.0, "chebyshev_radius_b_only": 0.0, "ratio": 0.0}

        basis_set = set(basis)
        nonbasic = [j for j in range(n) if j not in basis_set]

        # Build G matrix: |N| × n
        G = np.zeros((len(nonbasic), n))
        c_bar = np.zeros(len(nonbasic))

        c_B = np.array([problem.c[j] if j < n else 0.0 for j in basis])
        y = c_B @ B_inv

        for idx, j in enumerate(nonbasic):
            a_j = A_aug[:, j]
            B_inv_a_j = B_inv @ a_j
            G[idx, j] = -1.0
            for l in range(m_aug):
                if basis[l] < n:
                    G[idx, basis[l]] = B_inv_a_j[l]
            # Reduced cost (for maximize, c̄_j ≤ 0 at optimality)
            c_bar[idx] = abs(problem.c[j] - float(y @ a_j))

        # OAT tolerances
        oat_rhs = self._extract_oat_rhs(state, problem)
        oat_obj = self._extract_oat_obj(state, problem)

        alpha_b_plus = np.array([oat_rhs.get(cn, 1e6) for cn in problem.constraint_names[:m]])
        alpha_b_minus = alpha_b_plus.copy()
        alpha_c_plus = np.array([oat_obj.get(vn, 1e6) for vn in problem.var_names[:n]])
        alpha_c_minus = alpha_c_plus.copy()

        # --- Build H matrix ---
        n_nb = len(nonbasic)
        dim = m + n  # δ = (δ_b, δ_c)

        rows_H, rows_h = [], []

        # (1) Primal: -B⁻¹[:, :m] δ_b ≤ x_B
        for i in range(m_aug):
            row = np.zeros(dim)
            for j in range(m):
                row[j] = -B_inv[i, j]
            rows_H.append(row)
            rows_h.append(float(x_B[i]))

        # (2) Dual: -G δ_c ≤ c̄
        for idx in range(n_nb):
            row = np.zeros(dim)
            for j in range(n):
                row[m + j] = -G[idx, j]
            rows_H.append(row)
            rows_h.append(float(c_bar[idx]))

        # (3) Box δ_b
        for j in range(m):
            r_up = np.zeros(dim); r_up[j] = 1.0
            r_lo = np.zeros(dim); r_lo[j] = -1.0
            rows_H.append(r_up); rows_h.append(float(alpha_b_plus[j]))
            rows_H.append(r_lo); rows_h.append(float(alpha_b_minus[j]))

        # (4) Box δ_c
        for j in range(n):
            r_up = np.zeros(dim); r_up[m + j] = 1.0
            r_lo = np.zeros(dim); r_lo[m + j] = -1.0
            rows_H.append(r_up); rows_h.append(float(alpha_c_plus[j]))
            rows_H.append(r_lo); rows_h.append(float(alpha_c_minus[j]))

        H = np.array(rows_H)
        h = np.array(rows_h)

        r_joint, center = self._chebyshev_center(H, h)

        # Also compute b-only radius for comparison
        region_b = self.analyze(state, problem)
        r_b = region_b.chebyshev_radius

        ratio = r_joint / r_b if r_b > 1e-10 else 0.0

        return {
            "chebyshev_radius_joint": r_joint,
            "chebyshev_radius_b_only": r_b,
            "ratio": ratio,
            "center_b": tuple(center[:m]) if center is not None else None,
            "center_c": tuple(center[m:]) if center is not None else None,
            "n_primal_rows": m_aug,
            "n_dual_rows": n_nb,
            "n_box_rows": 2 * m + 2 * n,
        }

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
        m_aug = B_inv.shape[0]
        n = problem.num_variables
        m = problem.num_constraints

        # Pad b to augmented dimensions (UB rows have their own RHS)
        import clara.engine.standard_form as standard_form
        _, b_aug, _ = standard_form.add_upper_bound_rows(problem)
        x_B = B_inv @ b_aug

        # Parameter vector: δ = [Δb₁..Δbₘ] (original constraints only)
        # Use B_inv[:, :m] since UB constraint RHS doesn't change
        dim = m

        rows_H = []
        rows_h = []

        # Primal feasibility: B⁻¹[:, :m] @ Δb ≤ x_B (for non-negativity)
        for i in range(m_aug):
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
