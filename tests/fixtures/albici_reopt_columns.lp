\ Albici et al. (2010) - Reopt Scenario columns (2 new variables)
\ x4: cost=3, column=[2,0,1,0]
\ x5: cost=5, column=[3,1,0,1]
\ Expected optimal: 33700/9

Maximize
 obj: 3 x1 + 4 x2 + 5 x3 + 3 x4 + 5 x5

Subject To
 S1: x1 + 2 x2 + x3 + 2 x4 + 3 x5 <= 1200
 S2: x1 + 3 x3 + x5 <= 1400
 S3: 2 x1 + x2 + 2 x3 + x4 <= 2000
 S4: 2 x2 + 3 x3 + x5 <= 800

Bounds
 x1 >= 0
 x2 >= 0
 x3 >= 0
 x4 >= 0
 x5 >= 0

End
