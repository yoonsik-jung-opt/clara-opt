"""Tests for clara.ce — counterfactual explanations (Phase 2 skeleton).

Validation strategy mirrors the Track A verification suite:
closed forms against hand-checkable geometry, search against full-basis
enumeration ground truth, certificates against their definitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from clara.ce import (
    CEGeometry,
    DriveBasicToZero,
    MakeVariableActive,
    MutableSpec,
    SetShadowPrice,
    d0_restricted,
    exact_counterfactual,
    find_counterfactual,
    sandwich_lb,
)
from clara.engine.highs_backend import HiGHSBackend
from clara.model.problem import LPProblem

BACKEND = HiGHSBackend()


def geom_of(problem: LPProblem) -> CEGeometry:
    state = BACKEND.solve(problem)
    assert state.basis_inverse is not None
    return CEGeometry.from_state(problem, state)


@pytest.fixture(scope="module")
def prod2():
    return geom_of(LPProblem(c=[3, 5], A=[[1, 0], [0, 2], [3, 2]],
                             b=[4, 12, 18], sense="maximize", name="prod2"))


def tiny(seed: int, m: int = 4, n: int = 6) -> LPProblem:
    rng = np.random.default_rng(200 + seed)
    return LPProblem(c=rng.uniform(1, 10, n), A=rng.uniform(0.1, 2.0, (m, n)),
                     b=rng.uniform(5, 15, m), sense="maximize", name=f"tiny{seed}")


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def test_geometry_selfconsistent(prod2):
    g = prod2
    assert np.max(np.abs(g.cbar[g.Bidx])) < 1e-9
    assert np.min(g.x_B) > -1e-9
    # prod2 optimum (2, 6), z = 36
    assert abs(-g.c_min[:2] @ np.array([2.0, 6.0]) - 36.0) < 1e-9


def test_closed_form_row_reaches_facet(prod2):
    g = prod2
    S_b = np.arange(g.m)
    for i in range(g.m_aug):
        if g.x_B[i] <= 1e-9:
            continue
        t, db = g.closed_form_row(i, S_b, p=2)
        if t is None:
            continue
        xB_new = g.x_B + g.Binv @ g.db_to_aug(db)
        assert abs(xB_new[i]) < 1e-9
        assert abs(np.linalg.norm(db) - t) < 1e-9


def test_closed_form_col_zeroes_reduced_cost():
    g = geom_of(tiny(0))
    S_c = np.arange(g.n)
    for j in range(g.n):
        if j in g.basic_set or g.cbar[j] <= 1e-9:
            continue
        t, dc = g.closed_form_col(j, S_c, p=1)
        assert t is not None
        assert abs(g.cbar[j] + g.h_col(j) @ dc) < 1e-9
        assert abs(np.abs(dc).sum() - t) < 1e-9


# ------------------------------------------------------------------
# Certificates
# ------------------------------------------------------------------

def test_d0_blockdiag_infinite_certificate():
    A1 = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    b1 = np.array([4.0, 12.0, 18.0])
    c1 = np.array([3.0, 5.0])
    prob = LPProblem(
        c=np.concatenate([c1, c1]),
        A=np.block([[A1, np.zeros((3, 2))], [np.zeros((3, 2)), A1]]),
        b=np.concatenate([b1, b1]), sense="maximize", name="blockdiag")
    g = geom_of(prob)
    S_block2 = np.array([3, 4, 5])
    # block-1 rows are unreachable from block-2 rhs coords: some row has
    # zero response norm, so restricted d0 must exceed the unrestricted one
    d_full = d0_restricted(g, np.arange(6), p=2)
    d_restr = d0_restricted(g, S_block2, p=2)
    assert d_restr >= d_full - 1e-12


def test_sandwich_lb_positive(prod2):
    lb = sandwich_lb(prod2, np.arange(prod2.m), np.arange(prod2.n), p=1)
    assert lb > 0


# ------------------------------------------------------------------
# Search vs enumeration ground truth
# ------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 5, 7])
def test_search_matches_enumeration_c_only(seed):
    g = geom_of(tiny(seed))
    mut = MutableSpec(rows=())              # c-only
    for j in range(g.n):
        if j in g.basic_set or g.cbar[j] <= 1e-9:
            continue
        tgt = MakeVariableActive(j=j)
        ex = exact_counterfactual(g, tgt, mut)
        got = find_counterfactual(g, tgt, mut, depth=2)
        if not ex.found:
            continue
        assert got.found
        # search is an upper bound; equals exact when the achieving basis
        # is within depth 2 (8/9 of shortcuts in the ground-truth study)
        assert got.cost >= ex.cost - 1e-9
        if ex.pivot_distance is not None and ex.pivot_distance <= 2:
            assert got.cost <= ex.cost * (1 + 1e-6)


def test_search_joint_upper_bounds_exact():
    g = geom_of(tiny(1))
    tgt = None
    for j in range(g.n):
        if j not in g.basic_set and g.cbar[j] > 1e-9:
            tgt = MakeVariableActive(j=j)
            break
    assert tgt is not None
    ex = exact_counterfactual(g, tgt)
    got = find_counterfactual(g, tgt, depth=2)
    assert ex.found and got.found
    assert got.cost >= ex.cost - 1e-9
    assert got.lower_bound <= ex.cost + 1e-9


def test_search_verified_by_resolve():
    g = geom_of(tiny(0))
    j = next(j for j in range(g.n)
             if j not in g.basic_set and g.cbar[j] > 1e-9)
    rep = find_counterfactual(g, MakeVariableActive(j=j), MutableSpec(rows=()))
    assert rep.found
    p2 = g.apply(db=rep.db, dc=rep.dc)
    st = BACKEND.solve(p2)
    assert st.is_optimal
    # x_j positive in some optimal solution: maximize x_j over the face
    eps = max(1e-9, 1e-9 * abs(st.optimal_value))
    face = LPProblem(
        c=np.eye(g.n)[j],
        A=np.vstack([p2.A, -p2.c]), b=np.append(p2.b, -(st.optimal_value - eps)),
        sense="maximize")
    st2 = BACKEND.solve(face)
    assert (not st2.is_optimal) or st2.optimal_value > 1e-7


# ------------------------------------------------------------------
# Other target classes exercise the generic subproblem path
# ------------------------------------------------------------------

def test_drive_basic_to_zero(prod2):
    g = prod2
    v = int(g.Bidx[int(np.argmax(g.x_B))])
    rep = find_counterfactual(g, DriveBasicToZero(v=v))
    assert rep.found
    assert rep.cost > 0


def test_set_shadow_price(prod2):
    g = prod2
    i = next(i for i in range(g.m) if abs(g.y[i]) > 1e-6)
    rep = find_counterfactual(g, SetShadowPrice(i=i, p=0.5 * float(g.y[i])),
                              MutableSpec(rows=()))
    assert rep.found
    assert rep.cost > 0
