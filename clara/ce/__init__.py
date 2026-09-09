"""clara.ce — basis-geometric counterfactual explanations for LP (Track A).

Public API (experimental, Phase 2 skeleton):

    from clara.ce import (
        CEGeometry, MutableSpec,
        MakeVariableActive, DriveBasicToZero, SetShadowPrice,
        find_counterfactual, exact_counterfactual,
    )

    state = HiGHSBackend().solve(problem)
    geom = CEGeometry.from_state(problem, state)
    report = find_counterfactual(geom, MakeVariableActive(j=3))
"""

from clara.ce.certificates import d0_restricted, refined_lb, sandwich_lb
from clara.ce.exact import exact_counterfactual
from clara.ce.geometry import CEGeometry
from clara.ce.search import CEReport, find_counterfactual
from clara.ce.targets import (
    DriveBasicToZero,
    MakeVariableActive,
    MutableSpec,
    SetShadowPrice,
    Target,
)

__all__ = [
    "CEGeometry", "CEReport", "MutableSpec", "Target",
    "MakeVariableActive", "DriveBasicToZero", "SetShadowPrice",
    "find_counterfactual", "exact_counterfactual",
    "d0_restricted", "sandwich_lb", "refined_lb",
]
