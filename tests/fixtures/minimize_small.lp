\ Small minimize problem for cross-validation
\ min 2x1 + 3x2 s.t. x1 + x2 <= 10, x1 <= 6, x2 <= 8
\ Optimal: x1=0, x2=0, min value=0

Minimize
 cost: 2 x1 + 3 x2

Subject To
 total: x1 + x2 <= 10
 cap1: x1 <= 6
 cap2: x2 <= 8

End
