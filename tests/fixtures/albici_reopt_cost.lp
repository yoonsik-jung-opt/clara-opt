\ Albici et al. (2010) - Reopt Scenario cost (obj coeff change)
\ c: (8, 6, 7)
\ Expected optimal: 24800/3

Maximize
 obj: 8 x1 + 6 x2 + 7 x3

Subject To
 S1: x1 + 2 x2 + x3 <= 1200
 S2: x1 + 3 x3 <= 1400
 S3: 2 x1 + x2 + 2 x3 <= 2000
 S4: 2 x2 + 3 x3 <= 800

Bounds
 x1 >= 0
 x2 >= 0
 x3 >= 0

End
