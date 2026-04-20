"""CPLEX LP file format parser.

Parses .lp files into LPProblem instances. Supports the standard CPLEX LP
subset: linear objectives, constraints (<=, >=, =), bounds, integer/binary
variable declarations.

Public API:
    read_lp(filepath) → LPProblem
    parse_lp(content, name) → LPProblem
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from clara.model.problem import LPProblem

logger = logging.getLogger(__name__)

# Section keyword mappings (lowercase)
_MAXIMIZE_KEYWORDS = {"maximize", "maximum", "max"}
_MINIMIZE_KEYWORDS = {"minimize", "minimum", "min"}
_OBJECTIVE_KEYWORDS = _MAXIMIZE_KEYWORDS | _MINIMIZE_KEYWORDS
_CONSTRAINT_KEYWORDS = {"subject to", "such that", "st", "s.t.", "st."}
_BOUNDS_KEYWORDS = {"bounds", "bound"}
_GENERAL_KEYWORDS = {"general", "generals", "gen"}
_BINARY_KEYWORDS = {"binary", "binaries", "bin"}
_SEMICONT_KEYWORDS = {"semi-continuous", "semi"}
_END_KEYWORDS = {"end"}

_ALL_SECTION_KEYWORDS = (
    _OBJECTIVE_KEYWORDS
    | _CONSTRAINT_KEYWORDS
    | _BOUNDS_KEYWORDS
    | _GENERAL_KEYWORDS
    | _BINARY_KEYWORDS
    | _SEMICONT_KEYWORDS
    | _END_KEYWORDS
)

# Regex for splitting "3.5x1" into coefficient and variable
_COEFF_VAR_RE = re.compile(
    r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([a-zA-Z_!][a-zA-Z0-9_.!#$%&()/,;?@`'{}|~]*)$"
)

# Regex for a valid variable name
_VAR_NAME_RE = re.compile(r"^[a-zA-Z_!#$%&()/,;?@`'{}|~][a-zA-Z0-9_.!#$%&()/,;?@`'{}|~]*$")

# Constraint sense operators
_SENSE_TOKENS = {"<=", ">=", "=", "<", ">", "=<", "=>"}


class LPParseError(Exception):
    """Raised for LP file syntax errors."""

    def __init__(
        self,
        message: str,
        line_number: int | None = None,
        line_content: str | None = None,
    ):
        self.line_number = line_number
        self.line_content = line_content
        parts = []
        if line_number is not None:
            parts.append(f"line {line_number}")
        parts.append(message)
        if line_content is not None:
            parts.append(f'  → "{line_content.strip()}"')
        super().__init__(": ".join(parts))


@dataclass
class _Constraint:
    """Internal constraint representation during parsing."""
    name: str
    terms: list[tuple[float, str]]  # (coefficient, variable_name)
    sense: str  # "<=", ">=", "="
    rhs: float


class _LPParser:
    """State-machine LP file parser."""

    def __init__(self, content: str, name: str) -> None:
        self.name = name
        self.raw_lines = content.splitlines()
        self.lines: list[tuple[int, str]] = []  # (original_line_number, stripped_content)
        self._preprocess()

        self.pos = 0
        self.sense: str = ""  # "maximize" or "minimize"
        self.obj_name: str = "obj"
        self.obj_terms: list[tuple[float, str]] = []
        self.obj_offset: float = 0.0

        self.constraints: list[_Constraint] = []
        self.bounds: dict[str, tuple[float, float]] = {}
        self.integer_vars: set[str] = set()
        self.binary_vars: set[str] = set()
        self.semicont_vars: set[str] = set()

        self._var_order: list[str] = []
        self._var_set: set[str] = set()
        self._constraint_names: set[str] = set()
        self._auto_constraint_idx = 0

    def _preprocess(self) -> None:
        """Strip comments and blank lines, preserving line numbers."""
        for i, raw in enumerate(self.raw_lines, 1):
            # Remove comments (everything after \)
            line = raw.split("\\")[0].strip()
            if line:
                self.lines.append((i, line))

    def _peek(self) -> tuple[int, str] | None:
        if self.pos < len(self.lines):
            return self.lines[self.pos]
        return None

    def _advance(self) -> tuple[int, str] | None:
        if self.pos < len(self.lines):
            result = self.lines[self.pos]
            self.pos += 1
            return result
        return None

    def _is_section_keyword(self, line: str) -> str | None:
        """Check if line is a section keyword. Returns normalized keyword or None."""
        low = line.lower().strip()
        # Two-word keywords
        for kw in _CONSTRAINT_KEYWORDS:
            if low == kw or low.startswith(kw + " ") or low.startswith(kw + "\t"):
                return "constraints"
        for kw in _SEMICONT_KEYWORDS:
            if low == kw:
                return "semicontinuous"
        for kw_set, section_name in [
            (_MAXIMIZE_KEYWORDS, "maximize"),
            (_MINIMIZE_KEYWORDS, "minimize"),
            (_BOUNDS_KEYWORDS, "bounds"),
            (_GENERAL_KEYWORDS, "general"),
            (_BINARY_KEYWORDS, "binary"),
            (_END_KEYWORDS, "end"),
        ]:
            if low in kw_set:
                return section_name
        return None

    def parse(self) -> LPProblem:
        """Parse the LP content and return an LPProblem."""
        if not self.lines:
            raise LPParseError("Empty or invalid LP file")

        # Parse objective section
        self._parse_objective_section()

        # Parse remaining sections
        while self.pos < len(self.lines):
            entry = self._peek()
            if entry is None:
                break
            lineno, line = entry
            section = self._is_section_keyword(line)
            if section == "constraints":
                self._advance()
                self._parse_constraints_section()
            elif section == "bounds":
                self._advance()
                self._parse_bounds_section()
            elif section == "general":
                self._advance()
                self._parse_var_list_section(self.integer_vars)
            elif section == "binary":
                self._advance()
                self._parse_var_list_section(self.binary_vars)
            elif section == "semicontinuous":
                self._advance()
                self._parse_var_list_section(self.semicont_vars)
            elif section == "end":
                self._advance()
                break
            else:
                # Might be continuation or unknown
                self._advance()

        if not self.constraints and not self.obj_terms:
            raise LPParseError("Expected SUBJECT TO section after objective")

        return self._build_lp_problem()

    def _parse_objective_section(self) -> None:
        """Parse MINIMIZE/MAXIMIZE and the objective expression."""
        entry = self._advance()
        if entry is None:
            raise LPParseError("Expected MINIMIZE or MAXIMIZE at start of file")

        lineno, line = entry
        section = self._is_section_keyword(line)
        if section not in ("maximize", "minimize"):
            raise LPParseError(
                "Expected MINIMIZE or MAXIMIZE at start of file",
                lineno, line,
            )
        self.sense = section

        # Collect objective expression lines until next section
        obj_text = self._collect_until_section()
        if not obj_text:
            return  # empty objective

        # Check for "name:" label
        first_line = obj_text[0][1]
        colon_idx = first_line.find(":")
        if colon_idx != -1:
            potential_name = first_line[:colon_idx].strip()
            if potential_name and _VAR_NAME_RE.match(potential_name):
                self.obj_name = potential_name
                obj_text[0] = (obj_text[0][0], first_line[colon_idx + 1:].strip())

        # Tokenize and parse
        tokens = self._tokenize_lines(obj_text)
        self.obj_terms, self.obj_offset = self._parse_linear_expr(tokens)

    def _parse_constraints_section(self) -> None:
        """Parse all constraints until next section."""
        while self.pos < len(self.lines):
            entry = self._peek()
            if entry is None:
                break
            lineno, line = entry
            if self._is_section_keyword(line) is not None:
                break

            # Collect lines for this constraint (until we find a sense operator and RHS)
            constraint_lines = []
            found_sense = False
            while self.pos < len(self.lines):
                entry = self._peek()
                if entry is None:
                    break
                ln, l = entry
                if self._is_section_keyword(l) is not None:
                    break
                self._advance()
                constraint_lines.append((ln, l))

                # Check if this line contains a sense operator followed by RHS
                if self._line_has_complete_constraint(l):
                    found_sense = True
                    break

            if not constraint_lines:
                break

            self._parse_single_constraint(constraint_lines)

    def _line_has_complete_constraint(self, line: str) -> bool:
        """Check if line contains a constraint sense with RHS."""
        tokens = line.split()
        for t in tokens:
            if t in _SENSE_TOKENS:
                return True
        return False

    def _parse_single_constraint(self, lines: list[tuple[int, str]]) -> None:
        """Parse a single constraint from collected lines."""
        if not lines:
            return

        # Join all lines into one string
        full_text = " ".join(l for _, l in lines)
        first_lineno = lines[0][0]

        # Check for "name:" label
        con_name = None
        colon_idx = full_text.find(":")
        if colon_idx != -1:
            potential_name = full_text[:colon_idx].strip()
            if potential_name and _VAR_NAME_RE.match(potential_name):
                # Make sure it's not part of an expression (e.g., not "3:" )
                if not potential_name[0].isdigit():
                    con_name = potential_name
                    full_text = full_text[colon_idx + 1:].strip()

        if con_name is None:
            self._auto_constraint_idx += 1
            con_name = f"c{self._auto_constraint_idx}"

        if con_name in self._constraint_names:
            raise LPParseError(
                f"Duplicate constraint name '{con_name}'",
                first_lineno,
            )
        self._constraint_names.add(con_name)

        # Find the sense operator
        sense, lhs_str, rhs_str = self._split_constraint(full_text, first_lineno, con_name)

        # Parse LHS
        tokens = self._tokenize_text(lhs_str)
        terms, offset = self._parse_linear_expr(tokens)

        # Parse RHS
        rhs_str = rhs_str.strip()
        if not rhs_str:
            raise LPParseError(
                f"Constraint '{con_name}' missing right-hand side value",
                first_lineno,
            )
        try:
            rhs = float(rhs_str) - offset
        except ValueError:
            raise LPParseError(
                f"Invalid RHS value '{rhs_str}' in constraint '{con_name}'",
                first_lineno,
            )

        # Normalize sense
        if sense in ("=<", "<"):
            sense = "<="
        elif sense in ("=>", ">"):
            sense = ">="

        self.constraints.append(_Constraint(
            name=con_name,
            terms=terms,
            sense=sense,
            rhs=rhs,
        ))

    def _split_constraint(
        self, text: str, lineno: int, name: str
    ) -> tuple[str, str, str]:
        """Split constraint text into (sense, lhs, rhs)."""
        # Try multi-char senses first
        for sense in ["<=", ">=", "=<", "=>"]:
            idx = text.find(sense)
            if idx != -1:
                return sense, text[:idx], text[idx + len(sense):]

        # Single-char
        for sense in ["<", ">", "="]:
            idx = text.find(sense)
            if idx != -1:
                # Make sure it's not part of a number (e.g., "1e-3")
                # Check that char before is not 'e' or 'E'
                if sense in "<>" and idx > 0 and text[idx - 1] in "eE":
                    continue
                return sense, text[:idx], text[idx + 1:]

        raise LPParseError(
            f"Constraint '{name}' missing inequality sense (<=, >=, =)",
            lineno,
        )

    def _parse_bounds_section(self) -> None:
        """Parse bounds section."""
        while self.pos < len(self.lines):
            entry = self._peek()
            if entry is None:
                break
            lineno, line = entry
            if self._is_section_keyword(line) is not None:
                break
            self._advance()
            self._parse_bound_line(line, lineno)

    def _parse_bound_line(self, line: str, lineno: int) -> None:
        """Parse a single bound line."""
        low = line.lower().strip()

        # Handle "var free"
        tokens = line.split()
        if len(tokens) >= 2 and tokens[-1].lower() == "free":
            var_name = tokens[0]
            self._register_var(var_name)
            if var_name in self.bounds:
                raise LPParseError(
                    f"Duplicate bound for variable '{var_name}'", lineno, line
                )
            self.bounds[var_name] = (float("-inf"), float("inf"))
            return

        # Handle double-bounded: "lb <= var <= ub"
        # Handle single: "var <= ub" or "var >= lb" or "var = val"
        parts = re.split(r"(<=|>=|=<|=>|<|>|=)", line)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == 5:
            # lb sense1 var sense2 ub
            lb_str, sense1, var_name, sense2, ub_str = parts
            lb = self._parse_bound_value(lb_str)
            ub = self._parse_bound_value(ub_str)
            self._register_var(var_name)
            if var_name in self.bounds:
                raise LPParseError(
                    f"Duplicate bound for variable '{var_name}'", lineno, line
                )
            self.bounds[var_name] = (lb, ub)
        elif len(parts) == 3:
            left, sense, right = parts
            # Determine which side is the variable
            if self._is_number_or_inf(left):
                # "lb <= var" or "lb >= var"
                var_name = right
                val = self._parse_bound_value(left)
                self._register_var(var_name)
                if sense in ("<=", "<", "=<"):
                    # lb <= var
                    existing = self.bounds.get(var_name, (0.0, float("inf")))
                    self.bounds[var_name] = (val, existing[1])
                elif sense in (">=", ">", "=>"):
                    # lb >= var → var <= lb
                    existing = self.bounds.get(var_name, (0.0, float("inf")))
                    self.bounds[var_name] = (existing[0], val)
                elif sense == "=":
                    self.bounds[var_name] = (val, val)
            else:
                # "var <= ub" or "var >= lb" or "var = val"
                var_name = left
                val = self._parse_bound_value(right)
                self._register_var(var_name)
                if sense in ("<=", "<", "=<"):
                    existing = self.bounds.get(var_name, (0.0, float("inf")))
                    self.bounds[var_name] = (existing[0], val)
                elif sense in (">=", ">", "=>"):
                    existing = self.bounds.get(var_name, (0.0, float("inf")))
                    self.bounds[var_name] = (val, existing[1])
                elif sense == "=":
                    self.bounds[var_name] = (val, val)
        else:
            raise LPParseError(f"Cannot parse bound", lineno, line)

    def _parse_bound_value(self, s: str) -> float:
        """Parse a bound value string, handling inf/infinity."""
        s = s.strip().lower()
        if s in ("+inf", "+infinity", "inf", "infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
        return float(s)

    def _is_number_or_inf(self, s: str) -> bool:
        """Check if string is a number or inf."""
        s = s.strip().lower()
        if s in ("+inf", "+infinity", "-inf", "-infinity", "inf", "infinity"):
            return True
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _parse_var_list_section(self, target_set: set[str]) -> None:
        """Parse a variable list section (General, Binary, Semi-Continuous)."""
        while self.pos < len(self.lines):
            entry = self._peek()
            if entry is None:
                break
            lineno, line = entry
            if self._is_section_keyword(line) is not None:
                break
            self._advance()
            for var_name in line.split():
                self._register_var(var_name)
                target_set.add(var_name)

    def _collect_until_section(self) -> list[tuple[int, str]]:
        """Collect lines until the next section keyword."""
        collected = []
        while self.pos < len(self.lines):
            entry = self._peek()
            if entry is None:
                break
            lineno, line = entry
            if self._is_section_keyword(line) is not None:
                break
            self._advance()
            collected.append((lineno, line))
        return collected

    def _tokenize_lines(self, lines: list[tuple[int, str]]) -> list[str]:
        """Tokenize multiple lines into a flat token list."""
        text = " ".join(l for _, l in lines)
        return self._tokenize_text(text)

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize a linear expression string."""
        # Insert spaces around + and - to help tokenization,
        # but be careful with scientific notation (e.g., 1.5e-3)
        result = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in "+-":
                # Check if this is part of scientific notation
                if i > 0 and text[i - 1] in "eE":
                    result.append(ch)
                else:
                    result.append(" ")
                    result.append(ch)
                    result.append(" ")
            else:
                result.append(ch)
            i += 1
        return "".join(result).split()

    def _parse_linear_expr(
        self, tokens: list[str]
    ) -> tuple[list[tuple[float, str]], float]:
        """Parse tokens into (terms, constant_offset).

        Returns:
            terms: list of (coefficient, variable_name)
            offset: constant term
        """
        terms: list[tuple[float, str]] = []
        offset = 0.0
        sign = 1.0
        coeff: float | None = None
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            if tok == "+":
                sign = 1.0
                i += 1
                continue
            elif tok == "-":
                sign = -1.0
                i += 1
                continue

            # Try to parse as number
            num = self._try_parse_number(tok)
            if num is not None:
                coeff = sign * num
                sign = 1.0
                i += 1
                continue

            # Try to parse as "3.5x1" (adjacent coeff+var)
            m = _COEFF_VAR_RE.match(tok)
            if m:
                c = float(m.group(1))
                var = m.group(2)
                if var:
                    self._register_var(var)
                    terms.append((sign * c, var))
                    sign = 1.0
                    coeff = None
                else:
                    coeff = sign * c
                    sign = 1.0
                i += 1
                continue

            # Must be a variable name
            if _VAR_NAME_RE.match(tok):
                self._register_var(tok)
                if coeff is not None:
                    terms.append((coeff, tok))
                else:
                    terms.append((sign, tok))
                    sign = 1.0
                coeff = None
                i += 1
                continue

            # If we had a pending coefficient with no variable, it's a constant
            if coeff is not None:
                offset += coeff
                coeff = None

            i += 1

        # Trailing constant
        if coeff is not None:
            offset += coeff

        return terms, offset

    def _try_parse_number(self, s: str) -> float | None:
        """Try to parse string as a number. Returns None if not a number."""
        try:
            return float(s)
        except ValueError:
            return None

    def _register_var(self, name: str) -> None:
        """Register a variable, preserving first-appearance order."""
        if name not in self._var_set:
            if len(name) > 255:
                raise LPParseError(f"Variable name too long ({len(name)} chars): '{name[:50]}...'")
            if name[0].isdigit() or name[0] == ".":
                raise LPParseError(f"Invalid variable name '{name}': cannot start with digit or period")
            self._var_set.add(name)
            self._var_order.append(name)

    def _build_lp_problem(self) -> LPProblem:
        """Convert parsed data into LPProblem."""
        var_names = self._var_order
        n = len(var_names)
        var_idx = {name: i for i, name in enumerate(var_names)}

        # Build objective vector
        c = np.zeros(n)
        for coeff, var in self.obj_terms:
            c[var_idx[var]] = coeff

        # Store sense — engine handles negation internally
        problem_sense = self.sense  # "maximize" or "minimize"

        # Separate constraints by type and convert >= to <=
        all_rows: list[np.ndarray] = []
        all_rhs: list[float] = []
        con_names: list[str] = []

        for con in self.constraints:
            row = np.zeros(n)
            for coeff, var in con.terms:
                row[var_idx[var]] += coeff

            if con.sense in ("<=", "<"):
                all_rows.append(row)
                all_rhs.append(con.rhs)
            elif con.sense in (">=", ">"):
                # Negate: ax >= b → -ax <= -b
                all_rows.append(-row)
                all_rhs.append(-con.rhs)
            elif con.sense == "=":
                # Equality: split into <= and >=
                all_rows.append(row.copy())
                all_rhs.append(con.rhs)
                all_rows.append(-row)
                all_rhs.append(-con.rhs)
                con_names.append(con.name)
                con_names.append(f"{con.name}_eq")
                continue

            con_names.append(con.name)

        m = len(all_rows)
        if m == 0:
            A = np.zeros((0, n))
            b = np.zeros(0)
        else:
            A = np.array(all_rows)
            b = np.array(all_rhs)

        # Build bounds arrays
        lower_bounds = np.zeros(n)
        upper_bounds = np.full(n, np.inf)
        for vname, (lo, hi) in self.bounds.items():
            if vname in var_idx:
                lower_bounds[var_idx[vname]] = lo
                upper_bounds[var_idx[vname]] = hi

        # Binary vars get [0, 1] bounds
        binary_indices: set[int] = set()
        for vname in self.binary_vars:
            if vname in var_idx:
                idx = var_idx[vname]
                binary_indices.add(idx)
                lower_bounds[idx] = 0.0
                upper_bounds[idx] = 1.0

        integer_indices: set[int] = set()
        for vname in self.integer_vars:
            if vname in var_idx:
                integer_indices.add(var_idx[vname])

        return LPProblem(
            c=c,
            A=A,
            b=b,
            var_names=list(var_names),
            constraint_names=con_names,
            name=self.name,
            sense=problem_sense,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            integer_vars=integer_indices,
            binary_vars=binary_indices,
        )


def read_lp(filepath: str | Path) -> LPProblem:
    """Read an LP file and return an LPProblem instance.

    Args:
        filepath: Path to .lp file.

    Returns:
        Parsed LPProblem ready for solving.

    Raises:
        LPParseError: If the file has syntax errors.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(filepath)
    content = path.read_text()
    return _LPParser(content, path.stem).parse()


def parse_lp(content: str, name: str = "unnamed") -> LPProblem:
    """Parse LP format string and return an LPProblem instance.

    Args:
        content: LP file content as string.
        name: Problem name.

    Returns:
        Parsed LPProblem ready for solving.

    Raises:
        LPParseError: If the content has syntax errors.
    """
    return _LPParser(content, name).parse()
