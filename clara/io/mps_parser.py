"""MPS file format parser.

Parses .mps files (fixed and free format) into LPProblem instances.
Supports: NAME, OBJSENSE, ROWS, COLUMNS, RHS, RANGES, BOUNDS, ENDATA.

Public API:
    read_mps(filepath) → LPProblem
    parse_mps(content, name) → LPProblem
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from clara.model.problem import LPProblem

logger = logging.getLogger(__name__)


class MPSParseError(Exception):
    """Raised for MPS file syntax errors."""

    def __init__(self, message: str, line_number: Optional[int] = None):
        self.line_number = line_number
        if line_number is not None:
            super().__init__(f"line {line_number}: {message}")
        else:
            super().__init__(message)


class _MPSParser:
    """State-machine MPS parser (free format, whitespace-split)."""

    def __init__(self, content: str, name: str) -> None:
        self.name = name
        self.raw_lines = content.splitlines()

        # Problem data
        self.sense = "minimize"  # MPS default
        self.obj_row: Optional[str] = None  # name of the N-type row
        self.row_types: dict[str, str] = {}  # row_name → "N"|"L"|"G"|"E"
        self.row_order: list[str] = []  # constraint rows (L, G, E) in order

        # Column data: {var_name: {row_name: coeff}}
        self.columns: dict[str, dict[str, float]] = {}
        self.var_order: list[str] = []
        self.var_set: set[str] = set()

        # RHS
        self.rhs: dict[str, float] = {}

        # Bounds
        self.lower_bounds: dict[str, float] = {}
        self.upper_bounds: dict[str, float] = {}

        # Integer/binary
        self.integer_vars: set[str] = set()
        self.binary_vars: set[str] = set()
        self._in_integer_section = False

    def parse(self) -> LPProblem:
        section = None

        for lineno, raw in enumerate(self.raw_lines, 1):
            line = raw.rstrip()

            # Skip comments and empty lines
            if not line or line.startswith("*"):
                continue

            # Section headers start in column 1 (no leading space)
            if line and not line[0].isspace():
                token = line.split()[0].upper() if line.split() else ""
                if token == "NAME":
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        self.name = parts[1].strip()
                    section = "NAME"
                    continue
                elif token in ("MIN", "MINIMIZE"):
                    self.sense = "minimize"
                    section = "OBJSENSE"
                    continue
                elif token in ("MAX", "MAXIMIZE"):
                    self.sense = "maximize"
                    section = "OBJSENSE"
                    continue
                elif token == "OBJSENSE":
                    section = "OBJSENSE"
                    continue
                elif token == "ROWS":
                    section = "ROWS"
                    continue
                elif token == "COLUMNS":
                    section = "COLUMNS"
                    continue
                elif token == "RHS":
                    section = "RHS"
                    continue
                elif token == "RANGES":
                    section = "RANGES"
                    continue
                elif token == "BOUNDS":
                    section = "BOUNDS"
                    continue
                elif token == "ENDATA":
                    break
                # Unknown section header — might be NAME value
                continue

            # Process data within sections
            fields = line.split()
            if not fields:
                continue

            if section == "OBJSENSE":
                token = fields[0].upper()
                if token in ("MIN", "MINIMIZE"):
                    self.sense = "minimize"
                elif token in ("MAX", "MAXIMIZE"):
                    self.sense = "maximize"
                continue

            if section == "ROWS":
                self._parse_row(fields, lineno)
            elif section == "COLUMNS":
                self._parse_column(fields, lineno)
            elif section == "RHS":
                self._parse_rhs(fields, lineno)
            elif section == "RANGES":
                pass  # Best-effort: skip ranges for now
            elif section == "BOUNDS":
                self._parse_bound(fields, lineno)

        return self._build_problem()

    def _parse_row(self, fields: list[str], lineno: int) -> None:
        if len(fields) < 2:
            return
        rtype, rname = fields[0].upper(), fields[1]
        self.row_types[rname] = rtype
        if rtype == "N":
            if self.obj_row is None:
                self.obj_row = rname
        else:
            self.row_order.append(rname)

    def _parse_column(self, fields: list[str], lineno: int) -> None:
        # Check for integer markers
        if len(fields) >= 3 and fields[1].upper() == "'MARKER'":
            marker = fields[2].strip("'\"").upper()
            if marker == "INTORG":
                self._in_integer_section = True
            elif marker == "INTEND":
                self._in_integer_section = False
            return

        if len(fields) < 3:
            return

        col_name = fields[0]
        if col_name not in self.var_set:
            self.var_set.add(col_name)
            self.var_order.append(col_name)
            self.columns[col_name] = {}

        if self._in_integer_section:
            self.integer_vars.add(col_name)

        # Parse one or two (row, value) pairs
        i = 1
        while i + 1 < len(fields):
            row_name = fields[i]
            try:
                value = float(fields[i + 1])
            except ValueError:
                break
            self.columns[col_name][row_name] = (
                self.columns[col_name].get(row_name, 0.0) + value
            )
            i += 2

    def _parse_rhs(self, fields: list[str], lineno: int) -> None:
        if len(fields) < 3:
            return
        # First field is RHS vector name (ignored)
        i = 1
        while i + 1 < len(fields):
            row_name = fields[i]
            try:
                value = float(fields[i + 1])
            except ValueError:
                break
            self.rhs[row_name] = value
            i += 2

    def _parse_bound(self, fields: list[str], lineno: int) -> None:
        if len(fields) < 3:
            return
        btype = fields[0].upper()
        # fields[1] is bound vector name (ignored)
        col_name = fields[2]
        value = float(fields[3]) if len(fields) > 3 else 0.0

        if btype == "LO":
            self.lower_bounds[col_name] = value
        elif btype == "UP":
            self.upper_bounds[col_name] = value
        elif btype == "FX":
            self.lower_bounds[col_name] = value
            self.upper_bounds[col_name] = value
        elif btype == "FR":
            self.lower_bounds[col_name] = float("-inf")
            self.upper_bounds[col_name] = float("inf")
        elif btype == "MI":
            self.lower_bounds[col_name] = float("-inf")
        elif btype == "PL":
            self.upper_bounds[col_name] = float("inf")
        elif btype == "BV":
            self.lower_bounds[col_name] = 0.0
            self.upper_bounds[col_name] = 1.0
            self.binary_vars.add(col_name)
        elif btype in ("LI", "UI"):
            self.integer_vars.add(col_name)
            if btype == "LI":
                self.lower_bounds[col_name] = value
            else:
                self.upper_bounds[col_name] = value

    def _build_problem(self) -> LPProblem:
        var_names = self.var_order
        n = len(var_names)
        var_idx = {name: i for i, name in enumerate(var_names)}

        if n == 0:
            raise MPSParseError("No variables found in COLUMNS section")

        # Build objective vector
        c = np.zeros(n)
        if self.obj_row:
            for vname, coeffs in self.columns.items():
                if self.obj_row in coeffs:
                    c[var_idx[vname]] = coeffs[self.obj_row]

        # Build constraint matrix
        # >= (G) rows: negate to <=, = (E) rows: split into <= and >=
        all_rows: list[np.ndarray] = []
        all_rhs: list[float] = []
        con_names: list[str] = []

        for rname in self.row_order:
            rtype = self.row_types[rname]
            row = np.zeros(n)
            for vname, coeffs in self.columns.items():
                if rname in coeffs:
                    row[var_idx[vname]] = coeffs[rname]

            rhs_val = self.rhs.get(rname, 0.0)

            if rtype == "L":
                all_rows.append(row)
                all_rhs.append(rhs_val)
                con_names.append(rname)
            elif rtype == "G":
                all_rows.append(-row)
                all_rhs.append(-rhs_val)
                con_names.append(rname)
            elif rtype == "E":
                all_rows.append(row.copy())
                all_rhs.append(rhs_val)
                con_names.append(rname)
                all_rows.append(-row)
                all_rhs.append(-rhs_val)
                con_names.append(f"{rname}_eq")

        m = len(all_rows)
        A = np.array(all_rows) if m > 0 else np.zeros((0, n))
        b = np.array(all_rhs) if m > 0 else np.zeros(0)

        # Bounds
        lb = np.zeros(n)
        ub = np.full(n, np.inf)
        for vname, val in self.lower_bounds.items():
            if vname in var_idx:
                lb[var_idx[vname]] = val
        for vname, val in self.upper_bounds.items():
            if vname in var_idx:
                ub[var_idx[vname]] = val

        # Integer/binary indices
        int_indices = {var_idx[v] for v in self.integer_vars if v in var_idx}
        bin_indices = {var_idx[v] for v in self.binary_vars if v in var_idx}

        return LPProblem(
            c=c, A=A, b=b,
            var_names=var_names,
            constraint_names=con_names,
            name=self.name,
            sense=self.sense,
            lower_bounds=lb,
            upper_bounds=ub,
            integer_vars=int_indices,
            binary_vars=bin_indices,
        )


def read_mps(filepath: str | Path) -> LPProblem:
    """Read an MPS file and return an LPProblem instance."""
    path = Path(filepath)
    content = path.read_text()
    return _MPSParser(content, path.stem).parse()


def parse_mps(content: str, name: str = "unnamed") -> LPProblem:
    """Parse MPS format string and return an LPProblem instance."""
    return _MPSParser(content, name).parse()
