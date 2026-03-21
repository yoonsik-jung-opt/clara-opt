\ Albici et al. (2010) - Reopt Scenario b1 (RHS change, basis preserved)
\ RHS: (1300, 1200, 1800, 1000)
\ Expected optimal: 33100/9

Maximize
 obj: 3 x1 + 4 x2 + 5 x3

Subject To
 S1: x1 + 2 x2 + x3 <= 1300
 S2: x1 + 3 x3 <= 1200
 S3: 2 x1 + x2 + 2 x3 <= 1800
 S4: 2 x2 + 3 x3 <= 1000

Bounds
 x1 >= 0
 x2 >= 0
 x3 >= 0

End
