"""
Albici et al. (2010) — Expected Results for Test Validation
============================================================

Source: "Reoptimizations in Linear Programming", MPRA Paper No. 20091
All values verified against scipy.optimize.linprog with HiGHS backend.

Usage in tests:
    from fixtures.albici_expected import SCENARIOS
    assert abs(result.optimal_value - SCENARIOS["base"]["optimal_value"]) < 1e-6
"""

from dataclasses import dataclass


@dataclass
class ExpectedResult:
    optimal_value: float
    description: str
    change_type: str  # "none", "Type R", "Type C", "Type V"
    basis_preserved: bool  # Whether original basis remains optimal
    reopt_method: str  # "none", "dual_simplex", "primal_simplex", "check_reduced_cost"
    notes: str = ""


# ============================================================
# Problem Data (shared across all scenarios)
# ============================================================

A = [
    [1, 2, 1],    # S1: Department 1
    [1, 0, 3],    # S2: Department 2
    [2, 1, 2],    # S3: Department 3
    [0, 2, 3],    # S4: Department 4
]

ORIGINAL_RHS = [1200, 1400, 2000, 800]
ORIGINAL_OBJ = [3, 4, 5]

# B^-1 from final simplex tableau (basis: y2, x1, x2, x3)
# Rows correspond to: y2 (slack S2), x1, x2, x3
# Columns correspond to: y1, y2, y3, y4 (original slack variables)
BASIS_INVERSE = [
    [11/9,  1,    -10/9, -6/9],
    [1/9,   0,     4/9,  -3/9],
    [6/9,   0,    -3/9,   0  ],
    [-4/9,  0,     2/9,   3/9],
]

# ============================================================
# Scenarios
# ============================================================

SCENARIOS = {

    # --- Scenario 0: Original Problem ---
    "base": ExpectedResult(
        optimal_value=33200 / 9,  # ≈ 3688.8889
        description="Original production planning problem",
        change_type="none",
        basis_preserved=True,
        reopt_method="none",
    ),

    # --- Scenario b1: RHS change, basis preserved ---
    "reopt_b1": ExpectedResult(
        optimal_value=33100 / 9,  # ≈ 3677.7778
        description="RHS changed to (1300, 1200, 1800, 1000). "
                    "B^-1 * b' >= 0 everywhere, so basis is preserved. "
                    "No pivots needed — just recompute x_B = B^-1 * b'.",
        change_type="Type R",
        basis_preserved=True,
        reopt_method="none",
        notes="This is the ideal case for reoptimization: "
              "the change stays within the sensitivity range.",
    ),

    # --- Scenario b2: RHS change, basis broken ---
    "reopt_b2": ExpectedResult(
        optimal_value=4300.0,
        description="RHS changed to (1500, 1300, 2400, 800). "
                    "B^-1 * b' has negative component (y2 = -600/9 < 0), "
                    "so the solution is not admissible. "
                    "Dual simplex warm-start required.",
        change_type="Type R",
        basis_preserved=False,
        reopt_method="dual_simplex",
        notes="The Oguz bound does not apply here (RHS change, not obj coeff). "
              "But sensitivity range analysis correctly predicts basis breakage.",
    ),

    # --- Scenario c: Objective coefficient change ---
    "reopt_cost": ExpectedResult(
        optimal_value=24800 / 3,  # ≈ 8266.6667
        description="Objective changed to (8, 6, 7) from (3, 4, 5). "
                    "Reduced costs (Δj) no longer satisfy optimality. "
                    "Primal simplex warm-start required. "
                    "Result: x3 = 0, meaning P3 is not produced.",
        change_type="Type C",
        basis_preserved=False,
        reopt_method="primal_simplex",
        notes="Oguz bound applicable: δ = max relative change = |8-3|/3 = 5/3. "
              "Bound = 2δ/(1+δ) = (10/3)/(8/3) = 10/8 = 1.25 → 125% loss bound. "
              "Actual relative change = |8266.67 - C_old*x_old|/8266.67. "
              "Large δ makes the bound loose — reoptimization clearly needed.",
    ),

    # --- Scenario d: Column addition ---
    "reopt_columns": ExpectedResult(
        optimal_value=33700 / 9,  # ≈ 3744.4444
        description="Two new products P4 (cost=3, consumption=[2,0,1,0]) and "
                    "P5 (cost=5, consumption=[3,1,0,1]) added. "
                    "Check if new variables have negative reduced cost "
                    "(would improve objective). P5 enters basis.",
        change_type="Type V",
        basis_preserved=False,
        reopt_method="check_reduced_cost",
        notes="Paper's manual result (24800/6 ≈ 4133.33) contains arithmetic "
              "errors in multi-step tableau computation. "
              "Verified correct value: 33700/9 ≈ 3744.4444 by scipy/HiGHS. "
              "Improvement over base: +55.56 (+1.5%).",
    ),
}

# ============================================================
# Reoptimization Change Vectors (for Phase 2 Change Detector)
# ============================================================

CHANGES = {
    "b1": {
        "type": "Type R",
        "delta_b": [100, -200, -200, 200],  # b_new - b_old
        "delta_c": None,
        "new_columns": None,
    },
    "b2": {
        "type": "Type R",
        "delta_b": [300, -100, 400, 0],  # b_new - b_old
        "delta_c": None,
        "new_columns": None,
    },
    "cost": {
        "type": "Type C",
        "delta_b": None,
        "delta_c": [5, 2, 2],  # c_new - c_old
        "new_columns": None,
    },
    "columns": {
        "type": "Type V",
        "delta_b": None,
        "delta_c": None,
        "new_columns": {
            "x4": {"cost": 3, "column": [2, 0, 1, 0]},
            "x5": {"cost": 5, "column": [3, 1, 0, 1]},
        },
    },
}

# ============================================================
# Compound Change for Parametric LP Testing (Type RC)
# ============================================================

PARAMETRIC_TEST = {
    "description": "Simultaneous b and c change for parametric LP validation. "
                   "b: (1200,1400,2000,800) → (1500,1300,2400,800), "
                   "c: (3,4,5) → (8,6,7). "
                   "θ=0 is original, θ=1 is fully changed.",
    "delta_b": [300, -100, 400, 0],
    "delta_c": [5, 2, 2],
    "theta_0_optimal": 33200 / 9,    # ≈ 3688.8889
    "theta_1_optimal": 10000.0,       # combined b2 + cost change

    # Breakpoint analysis (verified analytically + binary search)
    # Basis {y2, x1, x2, x3} breaks at first of these:
    "breakpoints": [
        {
            "theta": 5 / 8,       # = 0.625
            "type": "primal",
            "description": "Basic var y2 (slack S2) hits 0. "
                           "y2(θ) = B⁻¹[0,:] @ (b+θΔb), slope = -177.78, "
                           "crosses zero at θ = 5/8.",
        },
        {
            "theta": 2 / 3,       # ≈ 0.6667
            "type": "dual",
            "description": "Reduced cost c̄_y4 hits 0. "
                           "c̄_y4(0) = 6/9 with slope -1, "
                           "crosses zero at θ = 2/3. "
                           "y4 (slack S4) would want to enter basis.",
        },
        {
            "theta": 4.0,
            "type": "primal",
            "description": "Basic var x3 hits 0, but θ > 1 so irrelevant.",
        },
    ],
    "first_breakpoint_theta": 5 / 8,  # = 0.625
    "first_breakpoint_type": "primal",

    # Sampled path (verified by scipy/HiGHS at each θ)
    "path_samples": [
        # (theta, optimal_value, x1, x2, x3)
        (0.0, 3688.8889, 755.56, 133.33, 177.78),
        (0.1, 4207.6667, 776.67, 140.00, 173.33),
        (0.2, 4748.4444, 797.78, 146.67, 168.89),
        (0.3, 5311.2222, 818.89, 153.33, 164.44),
        (0.4, 5896.0000, 840.00, 160.00, 160.00),
        (0.5, 6502.7778, 861.11, 166.67, 155.56),
        (0.6, 7131.5556, 882.22, 173.33, 151.11),
        # --- breakpoint at θ* = 0.625 (primal) then 0.667 (dual) ---
        (0.7, 7797.0000, 1050.00, 180.00, 0.00),
        (0.8, 8512.0000, 1066.67, 186.67, 0.00),
        (0.9, 9246.3333, 1083.33, 193.33, 0.00),
        (1.0, 10000.0000, 1100.00, 200.00, 0.00),
    ],
}
