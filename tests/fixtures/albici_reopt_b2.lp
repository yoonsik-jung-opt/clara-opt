\ Albici et al. (2010) - Reopt Scenario b2 (RHS change, basis broken)
\ RHS: (1500, 1300, 2400, 800)
\ Expected optimal: 4300.0

Maximize
 obj: 3 x1 + 4 x2 + 5 x3

Subject To
 S1: x1 + 2 x2 + x3 <= 1500
 S2: x1 + 3 x3 <= 1300
 S3: 2 x1 + x2 + 2 x3 <= 2400
 S4: 2 x2 + 3 x3 <= 800

Bounds
 x1 >= 0
 x2 >= 0
 x3 >= 0

End
