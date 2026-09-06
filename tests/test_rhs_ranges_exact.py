"""Exact RHS ranging must agree with re-solving at the range endpoints.

Regression test for the misinterpretation of HiGHS row-bound ranging on
basic (non-binding) rows: the stored rhs range must contain the current
rhs, the optimal basis must survive a perturbation just inside the range,
and must change just outside a finite endpoint (nondegenerate instances).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from clara.engine.highs_backend import HiGHSBackend
from clara.model.problem import LPProblem

BACKEND = HiGHSBackend()


def random_lp(seed: int, n: int = 8, m: int = 12) -> LPProblem:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.5)
    xbar = rng.uniform(1, 10, n)
    s = rng.uniform(0.1, 10, m)
    return LPProblem(c=rng.uniform(0.1, 5, n), A=A, b=A @ xbar + s,
                     sense="maximize", upper_bounds=3 * xbar, name=f"t{seed}")


def basic_set(state) -> frozenset:
    return frozenset(state.basis_indices)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_rhs_ranges_contain_rhs_and_match_resolve(seed):
    prob = random_lp(seed)
    st = BACKEND.solve(prob)
    assert st.basis_inverse is not None
    if st.degenerate_count > 0:
        pytest.skip("degenerate instance; basis uniqueness not guaranteed")
    base = basic_set(st)
    for k, con in enumerate(st.constraints):
        lo, hi = st.sensitivity.rhs_ranges[con.name]
        assert lo <= prob.b[k] + 1e-9 and hi >= prob.b[k] - 1e-9, "range excludes current rhs"
        for endpoint, sign in ((hi, +1), (lo, -1)):
            if math.isinf(endpoint):
                continue
            width = abs(endpoint - prob.b[k])
            if width < 1e-6:
                continue
            b_in = prob.b.copy()
            b_in[k] = prob.b[k] + sign * 0.99 * width
            st_in = BACKEND.solve(LPProblem(c=prob.c, A=prob.A, b=b_in, sense=prob.sense,
                                            upper_bounds=prob.upper_bounds))
            assert basic_set(st_in) == base, f"basis changed inside range (row {k})"
            b_out = prob.b.copy()
            b_out[k] = prob.b[k] + sign * 1.05 * width
            st_out = BACKEND.solve(LPProblem(c=prob.c, A=prob.A, b=b_out, sense=prob.sense,
                                             upper_bounds=prob.upper_bounds))
            if st_out.is_optimal:
                assert basic_set(st_out) != base, f"basis unchanged outside range (row {k})"


def test_nonbinding_row_has_infinite_increase():
    prob = random_lp(7)
    st = BACKEND.solve(prob)
    for con in st.constraints:
        if con.basis_status.name == "BASIC" and con.slack > 1e-6:
            lo, hi = st.sensitivity.rhs_ranges[con.name]
            assert math.isinf(hi)
            assert abs((con.rhs - lo) - con.slack) < 1e-7
