"""
CLARA LP Parser — Design Specification
========================================

Implements a CPLEX LP file format parser that converts .lp files into LPProblem.
Based on IBM CPLEX LP format specification (ICOS 22.1.0).

Reference: https://www.ibm.com/docs/en/icos/22.1.0?topic=cplex-lp-file-format-algebraic-representation

Scope: Standard CPLEX LP subset sufficient for LP, MIP, and educational use.
Explicitly excluded: Quadratic terms (QP/SOCP), piecewise linear, lazy constraints,
                     multi-objective, SOS (Special Ordered Sets), indicator constraints.
"""

# ============================================================
# 1. Sections to Parse (in order of appearance)
# ============================================================

SUPPORTED_SECTIONS = """
┌──────────────────┬──────────────────────────────────────────────────┬──────────┐
│ Section          │ Keywords (case-insensitive)                      │ Required │
├──────────────────┼──────────────────────────────────────────────────┼──────────┤
│ Objective        │ MINIMIZE, MAXIMIZE, MIN, MAX, MINIMUM, MAXIMUM  │ Yes      │
│ Constraints      │ SUBJECT TO, SUCH THAT, ST, S.T., ST.            │ Yes      │
│ Bounds           │ BOUNDS, BOUND                                    │ No       │
│ Integer vars     │ GENERAL, GENERALS, GEN                           │ No       │
│ Binary vars      │ BINARY, BINARIES, BIN                            │ No       │
│ Semi-continuous  │ SEMI-CONTINUOUS, SEMI                             │ No       │
│ End              │ END                                               │ Yes*     │
└──────────────────┴──────────────────────────────────────────────────┴──────────┘

* END is strongly recommended but parser should also handle EOF without END.
"""


# ============================================================
# 2. Grammar Rules
# ============================================================

GRAMMAR = """
# Comments
- Lines starting with \\ are comments (backslash)
- Everything after \\ on a line is ignored

# Variable names
- Alphanumeric + ! " # $ % & ( ) / , . ; ? @ _ ` ' { } | ~
- Cannot start with a digit or period
- Max 255 characters (CPLEX spec; we enforce 255)
- Avoid standalone 'e' or 'E' (exponential notation conflict)

# Objective function
  [name :] linear_expr

  Examples:
    Maximize
      obj: 3 x1 + 4 x2 + 5 x3

    Minimize
      cost: x1 + 2.5 x2 - x3

    Min
      3x + 4y                    (no name → default "obj")

# Constraints
  [name :] linear_expr sense rhs

  Sense: <=, >=, =   (also accepts <, >, =<, =>)
  RHS: a single number, must be on same line as sense

  Examples:
    Subject To
      S1: x1 + 2 x2 + x3 <= 1200
      S2: x1 + 3 x3 <= 1400
      c3: - x1 + x2 = 0
      x1 + x2 >= 100              (no name → auto-assign "c4")

  Multi-line constraints: a constraint continues until a sense indicator
  is found. Lines do NOT need to start with +/-.

  Range constraints (CPLEX extension):
    NOT SUPPORTED in v0.1. Raise ParseError with helpful message.

# Bounds
  Default: 0 <= x <= +inf for all variables

  Formats:
    lb <= var <= ub              (double-bounded)
    var <= ub                    (upper bound only)
    var >= lb                    (lower bound only)
    var = value                  (fixed variable)
    var free                     (unbounded: -inf <= var <= +inf)
    -inf <= var <= ub            (explicit -inf lower)

  Constants: +inf, +infinity, -inf, -infinity (case-insensitive)

  Examples:
    Bounds
      0 <= x1 <= 100
      x2 <= 50
      x3 >= 10
      x4 = 25
      x5 free

# Integer variables
  General
    x1 x2 x3                    (space-separated, can span multiple lines)

# Binary variables
  Binary
    y1 y2 y3                    (automatically bounded [0, 1])

# Semi-continuous variables
  Semi-Continuous
    z1 z2                       (value is 0 or within [lb, ub])
"""


# ============================================================
# 3. Linear Expression Parsing (핵심 파싱 로직)
# ============================================================

LINEAR_EXPR_RULES = """
A linear expression is a sequence of terms: [sign] [coeff] [varname]

Parsing rules:
  1. Tokenize by whitespace, keeping +/- as sign indicators
  2. Coefficient can be:
     - Explicit:    "3 x1"  → (3.0, "x1")
     - Implicit 1:  "+ x1"  → (1.0, "x1")
     - Implicit 1:  "- x1"  → (-1.0, "x1")
     - Adjacent:    "3x1"   → (3.0, "x1")  (no space between coeff and var)
     - Scientific:  "1.5e2 x1" → (150.0, "x1")
  3. Sign handling:
     - First term: sign optional (default +)
     - Subsequent terms: + or - required between terms
     - Adjacent signs: "+ -2 x1" → (-2.0, "x1")   [CPLEX allows this]
  4. Constant terms in objective: "3 x1 + 4 x2 + 10" → offset = 10

Edge cases to handle:
  - "x1" alone means 1.0 * x1
  - "-x1" means -1.0 * x1
  - "+x1" means 1.0 * x1
  - "3.5x1" (no space) means 3.5 * x1
  - "x1 + x2" (implicit coefficients)
  - Empty objective: "Minimize" followed by "Subject To" → zero objective

Edge cases to REJECT:
  - "x1 x2" without operator → ParseError (ambiguous: product or sum?)
  - "x1 * x2" → ParseError (nonlinear, not supported)
  - Parentheses "()" → ParseError (not supported in LP format)
"""


# ============================================================
# 4. Output: LPProblem Mapping
# ============================================================

MAPPING_TO_LPPROBLEM = """
Parser output fills these LPProblem fields:

  .name              ← filename stem (e.g., "albici_base" from "albici_base.lp")
  .sense             ← MINIMIZE or MAXIMIZE (from section keyword)
  .objective_name    ← from "name:" label, default "obj"
  .c                 ← numpy array of objective coefficients
  .obj_offset        ← constant term in objective (default 0.0)

  .A_ub              ← numpy matrix for <= constraints
  .b_ub              ← numpy array for <= RHS
  .A_eq              ← numpy matrix for = constraints
  .b_eq              ← numpy array for = RHS
  Note: >= constraints are converted to <= by negation

  .variable_names    ← ordered list of variable name strings
  .constraint_names  ← ordered list of constraint name strings

  .lower_bounds      ← numpy array (default 0.0 per variable)
  .upper_bounds      ← numpy array (default +inf per variable)

  .integer_vars      ← set of variable indices marked GENERAL
  .binary_vars       ← set of variable indices marked BINARY
  .semicont_vars     ← set of variable indices marked SEMI-CONTINUOUS

Variable ordering:
  Variables are ordered by first appearance in the file (objective first,
  then constraints). This matches CPLEX behavior.

Constraint ordering:
  Constraints are ordered by appearance. Auto-named as c1, c2, ... if unnamed.
"""


# ============================================================
# 5. Error Handling
# ============================================================

ERROR_HANDLING = """
class LPParseError(Exception):
    '''Raised for any LP file syntax error.'''
    def __init__(self, message: str, line_number: int | None = None,
                 line_content: str | None = None):
        self.line_number = line_number
        self.line_content = line_content
        super().__init__(self._format())

    def _format(self) -> str:
        parts = []
        if self.line_number:
            parts.append(f"line {self.line_number}")
        parts.append(self.args[0] if self.args else "unknown error")
        if self.line_content:
            parts.append(f'  → "{self.line_content.strip()}"')
        return ": ".join(parts) if len(parts) > 1 else parts[0]

Error categories with messages:
  - Missing objective section → "Expected MINIMIZE or MAXIMIZE at start of file"
  - Missing constraints section → "Expected SUBJECT TO section after objective"
  - Unknown section keyword → "Unknown section '{word}'. Expected one of: ..."
  - Invalid variable name → "Invalid variable name '{name}': cannot start with digit"
  - Duplicate variable in bounds → "Duplicate bound for variable '{name}' at line {n}"
  - Duplicate constraint name → "Duplicate constraint name '{name}'"
  - Missing RHS → "Constraint '{name}' missing right-hand side value"
  - Missing sense → "Constraint '{name}' missing inequality sense (<=, >=, =)"
  - Nonlinear term → "Nonlinear terms not supported: '{token}'"
  - Range constraint → "Range constraints not supported in v0.1"
  - Empty file → "Empty or invalid LP file"
  - No END keyword → Warning only (still parse, but log warning)
"""


# ============================================================
# 6. Public API
# ============================================================

API = """
# Primary function
def read_lp(filepath: str | Path) -> LPProblem:
    '''Read an LP file and return an LPProblem instance.

    Args:
        filepath: Path to .lp file.

    Returns:
        Parsed LPProblem ready for solving.

    Raises:
        LPParseError: If the file has syntax errors.
        FileNotFoundError: If the file doesn't exist.
    '''

# Alternative: parse from string (useful for tests)
def parse_lp(content: str, name: str = "unnamed") -> LPProblem:
    '''Parse LP format string and return an LPProblem instance.'''

# Module location
clara/io/lp_parser.py

# Re-export from clara/io/__init__.py
from clara.io.lp_parser import read_lp, parse_lp, LPParseError
"""


# ============================================================
# 7. Test Cases
# ============================================================

TEST_CASES = """
tests/test_lp_parser.py

1. Basic parsing
   - test_parse_albici_base          → 3 vars, 4 constraints, max, obj=3x1+4x2+5x3
   - test_parse_albici_reopt_b1      → same structure, different RHS
   - test_parse_albici_reopt_columns → 5 vars, 4 constraints
   - test_parse_minimize             → min direction
   - test_parse_no_bounds_section    → defaults to 0 <= x <= +inf
   - test_parse_named_constraints    → S1, S2 etc. preserved
   - test_parse_unnamed_constraints  → auto-named c1, c2, ...
   - test_parse_mixed_names          → some named, some auto

2. Objective
   - test_parse_named_objective      → "obj:" prefix
   - test_parse_unnamed_objective    → default name "obj"
   - test_parse_obj_with_offset      → "3 x1 + 4 x2 + 10" → offset=10
   - test_parse_single_var_obj       → "x1" alone → coeff 1.0

3. Coefficients & expressions
   - test_coeff_implicit_one         → "x1" → 1.0, "-x1" → -1.0
   - test_coeff_adjacent             → "3x1" → 3.0
   - test_coeff_float                → "2.5 x1" → 2.5
   - test_coeff_scientific           → "1.5e2 x1" → 150.0
   - test_coeff_negative             → "- 3 x1" → -3.0
   - test_multiline_constraint       → expression spanning two lines

4. Constraints
   - test_leq_constraint             → <=
   - test_geq_constraint             → >= → negated to <=
   - test_eq_constraint              → =
   - test_sense_variants             → <, >, =<, => all accepted

5. Bounds
   - test_double_bound               → "0 <= x1 <= 100"
   - test_upper_only                 → "x1 <= 50"
   - test_lower_only                 → "x1 >= 10"
   - test_fixed                      → "x1 = 25"
   - test_free                       → "x1 free" → (-inf, +inf)
   - test_explicit_inf               → "-inf <= x1 <= 100"
   - test_default_bounds             → no bounds section → [0, +inf]

6. MIP features
   - test_general_section            → integer_vars populated
   - test_binary_section             → binary_vars populated, bounds [0,1]
   - test_semicontinuous_section     → semicont_vars populated

7. Error cases
   - test_error_missing_objective    → LPParseError
   - test_error_missing_constraints  → LPParseError
   - test_error_invalid_var_name     → "3abc" as var name
   - test_error_duplicate_bound      → two bounds for same var
   - test_error_missing_rhs          → "x1 + x2 <="
   - test_error_nonlinear            → "x1 * x2"
   - test_error_empty_file           → LPParseError
   - test_error_no_end               → warning but still parses

8. Round-trip validation
   - test_albici_base_solve          → parse → Internal Simplex → opt = 33200/9
   - test_all_fixtures_parse         → all .lp files in fixtures/ parse without error

9. Edge cases
   - test_comments_ignored           → lines starting with \\
   - test_inline_comments            → content after \\
   - test_extra_whitespace           → tabs, multiple spaces
   - test_case_insensitive_keywords  → "maximize", "MAXIMIZE", "Maximize"
   - test_empty_lines_between        → blank lines between sections
   - test_long_variable_name         → 255-char name
"""


# ============================================================
# 8. Implementation Notes for Claude Code
# ============================================================

IMPLEMENTATION_NOTES = """
Architecture:
  - State machine parser: current_section tracks which section we're in
  - Single pass: read line by line, dispatch to section handler
  - Variables discovered on-the-fly, ordered by first appearance

Recommended structure:
  class _LPParser:
      def __init__(self, content: str, name: str):
          self.lines: list[str]
          self.line_idx: int = 0
          self.current_section: str = ""

          # Accumulated data
          self.sense: str = ""             # "maximize" or "minimize"
          self.obj_name: str = "obj"
          self.obj_terms: list[tuple[float, str]] = []
          self.obj_offset: float = 0.0

          self.constraints: list[_Constraint] = []
          self.bounds: dict[str, tuple[float, float]] = {}
          self.integer_vars: set[str] = set()
          self.binary_vars: set[str] = set()
          self.semicont_vars: set[str] = set()

          # Variable registry (preserves order)
          self._var_order: list[str] = []
          self._var_set: set[str] = set()

      def parse(self) -> LPProblem: ...
      def _parse_objective(self) -> None: ...
      def _parse_constraints(self) -> None: ...
      def _parse_bounds(self) -> None: ...
      def _parse_integer(self) -> None: ...
      def _parse_binary(self) -> None: ...
      def _parse_semicontinuous(self) -> None: ...
      def _parse_linear_expr(self, tokens) -> list[tuple[float, str]]: ...
      def _register_var(self, name: str) -> None: ...
      def _build_lp_problem(self) -> LPProblem: ...

  # Public API wraps this class:
  def read_lp(filepath) -> LPProblem:
      content = Path(filepath).read_text()
      return _LPParser(content, Path(filepath).stem).parse()

  def parse_lp(content, name="unnamed") -> LPProblem:
      return _LPParser(content, name).parse()

Performance:
  - This parser is NOT performance-critical (educational/small-medium problems)
  - For large problems, users should use HiGHS backend which has its own parser
  - No regex for main parsing loop — token-based for clarity
  - Regex acceptable for: coefficient extraction ("3.5x1" → "3.5", "x1")

Dependencies:
  - Only numpy (for array construction)
  - No external parsing libraries needed
"""