\ CLARA LP Parser — Comprehensive Test Fixture
\ Tests: comments, all bound types, mixed naming, MIP features,
\        multi-line expressions, implicit coefficients, scientific notation
\ Expected: 6 variables, 5 constraints, 2 integer vars, 1 binary var

Maximize
  profit: 3 x1 + 4.5 x2 - 2x3 + 1.5e1 y1 + y2 + 7 z1

Subject To
  capacity1: x1 + 2 x2 + x3 <= 1200
  capacity2: x1
             + 3 x3 <= 1400
  \ inline comment after a constraint
  balance: x1 - x2 = 0
  x1 + x2 + x3 + y1 + y2 + z1 >= 100
  demand: - x1 + 2 x2
          - 3 x3 + y1 <= 500

Bounds
  0 <= x1 <= 1000
  x2 <= 800
  x3 >= 10
  y1 = 50
  y2 free
  -inf <= z1 <= +infinity

General
  x1 x3

Binary
  y1

End
