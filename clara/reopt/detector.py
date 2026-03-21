"""Change Detector — compares two LPProblems and classifies parameter changes.

Based on Albici et al. (2010) taxonomy, extended with TYPE_A and TYPE_MULTI.

Usage:
    detector = ChangeDetector()
    change = detector.detect(old_problem, new_problem)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clara.model.problem import LPProblem
from clara.reopt.types import (
    ChangeType,
    IncompatibleProblemsError,
    ParameterChange,
)


class ChangeDetector:
    """Compare two LP problems and classify the parameter changes."""

    TOLERANCE = 1e-10

    def detect(self, old: LPProblem, new: LPProblem) -> Optional[ParameterChange]:
        """Compare old and new problem, return classified change.

        Args:
            old: The original problem.
            new: The modified problem.

        Returns:
            ParameterChange with change_type and relevant delta vectors,
            or None if the problems are identical.

        Raises:
            IncompatibleProblemsError: If problems have no common variables.
        """
        # Step 1: Variable set comparison
        old_vars = set(old.var_names)
        new_vars = set(new.var_names)
        added_vars = new_vars - old_vars
        removed_vars = old_vars - new_vars
        common_vars = old_vars & new_vars

        if not common_vars:
            raise IncompatibleProblemsError(
                f"No common variables between problems. "
                f"Old: {old.var_names}, New: {new.var_names}"
            )

        # Step 2: Constraint set comparison
        old_cons = set(old.constraint_names)
        new_cons = set(new.constraint_names)
        added_cons = new_cons - old_cons
        removed_cons = old_cons - new_cons
        common_cons = old_cons & new_cons

        # Step 3: Index maps
        old_var_idx = {name: i for i, name in enumerate(old.var_names)}
        new_var_idx = {name: i for i, name in enumerate(new.var_names)}
        old_con_idx = {name: i for i, name in enumerate(old.constraint_names)}
        new_con_idx = {name: i for i, name in enumerate(new.constraint_names)}

        # Compute delta_c for common variables
        delta_c = np.zeros(len(old.var_names))
        c_changed = False
        for name in common_vars:
            oi, ni = old_var_idx[name], new_var_idx[name]
            d = new.c[ni] - old.c[oi]
            delta_c[oi] = d
            if abs(d) > self.TOLERANCE:
                c_changed = True

        # Compute delta_b for common constraints
        delta_b = np.zeros(len(old.constraint_names))
        b_changed = False
        for name in common_cons:
            oi, ni = old_con_idx[name], new_con_idx[name]
            d = new.b[ni] - old.b[oi]
            delta_b[oi] = d
            if abs(d) > self.TOLERANCE:
                b_changed = True

        # Check A matrix changes for common vars × common constraints
        a_changed = False
        for con_name in common_cons:
            for var_name in common_vars:
                oi_c, ni_c = old_con_idx[con_name], new_con_idx[con_name]
                oi_v, ni_v = old_var_idx[var_name], new_var_idx[var_name]
                d = new.A[ni_c, ni_v] - old.A[oi_c, oi_v]
                if abs(d) > self.TOLERANCE:
                    a_changed = True
                    break
            if a_changed:
                break

        # Step 4: Structural flags
        has_structural = bool(added_vars or removed_vars or added_cons or removed_cons)

        if not has_structural and not c_changed and not b_changed and not a_changed:
            return None

        # Build new_columns for added variables
        new_columns = None
        if added_vars:
            new_columns = {}
            for name in sorted(added_vars):
                ni = new_var_idx[name]
                col = []
                for con_name in old.constraint_names:
                    if con_name in new_con_idx:
                        col.append(float(new.A[new_con_idx[con_name], ni]))
                    else:
                        col.append(0.0)
                new_columns[name] = {
                    "cost": float(new.c[ni]),
                    "column": col,
                }

        # Build new_rows for added constraints
        new_rows = None
        if added_cons:
            new_rows = {}
            for name in sorted(added_cons):
                ni = new_con_idx[name]
                coeffs = []
                for var_name in old.var_names:
                    if var_name in new_var_idx:
                        coeffs.append(float(new.A[ni, new_var_idx[var_name]]))
                    else:
                        coeffs.append(0.0)
                new_rows[name] = {
                    "coeffs": coeffs,
                    "rhs": float(new.b[ni]),
                }

        removed_row_list = sorted(removed_cons) if removed_cons else None

        # Classification
        if has_structural:
            if added_vars and not added_cons and not removed_cons and not b_changed and not c_changed:
                change_type = ChangeType.TYPE_V
            elif added_cons and not added_vars and not removed_vars and not b_changed and not c_changed:
                change_type = ChangeType.TYPE_X
            elif removed_cons and not added_vars and not added_cons and not b_changed and not c_changed:
                change_type = ChangeType.TYPE_M
            else:
                change_type = ChangeType.TYPE_MULTI
        elif a_changed:
            change_type = ChangeType.TYPE_A
        elif b_changed and c_changed:
            change_type = ChangeType.TYPE_RC
        elif b_changed:
            change_type = ChangeType.TYPE_R
        elif c_changed:
            change_type = ChangeType.TYPE_C
        else:
            return None

        return ParameterChange(
            change_type=change_type,
            delta_b=delta_b if b_changed else None,
            delta_c=delta_c if c_changed else None,
            new_columns=new_columns,
            new_rows=new_rows,
            removed_rows=removed_row_list,
        )
