\ Albici et al. (2010) - Production Planning Example
\ Original problem: 4 departments, 3 products
\ Source: "Reoptimizations in Linear Programming", MPRA Paper No. 20091
\ Optimal value: 33200/9 ≈ 3688.8889

Maximize
 obj: 3 x1 + 4 x2 + 5 x3

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
