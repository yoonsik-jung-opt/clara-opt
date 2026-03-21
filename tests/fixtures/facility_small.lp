\ Small facility location for B&B testing
\ LP relaxation: 56.33, MIP optimal: 61.0

Minimize
 obj: 20 y1 + 25 y2 + 18 y3
    + 4 x11 + 6 x12 + 9 x13 + 8 x14 + 7 x15
    + 6 x21 + 3 x22 + 7 x23 + 4 x24 + 5 x25
    + 5 x31 + 7 x32 + 3 x33 + 6 x34 + 4 x35

Subject To
 d1: x11 + x21 + x31 = 1
 d2: x12 + x22 + x32 = 1
 d3: x13 + x23 + x33 = 1
 d4: x14 + x24 + x34 = 1
 d5: x15 + x25 + x35 = 1
 l11: x11 - y1 <= 0
 l12: x12 - y1 <= 0
 l13: x13 - y1 <= 0
 l14: x14 - y1 <= 0
 l15: x15 - y1 <= 0
 l21: x21 - y2 <= 0
 l22: x22 - y2 <= 0
 l23: x23 - y2 <= 0
 l24: x24 - y2 <= 0
 l25: x25 - y2 <= 0
 l31: x31 - y3 <= 0
 l32: x32 - y3 <= 0
 l33: x33 - y3 <= 0
 l34: x34 - y3 <= 0
 l35: x35 - y3 <= 0
 c1: x11 + x12 + x13 + x14 + x15 - 3 y1 <= 0
 c2: x21 + x22 + x23 + x24 + x25 - 2 y2 <= 0
 c3: x31 + x32 + x33 + x34 + x35 - 3 y3 <= 0

Bounds
 0 <= y1 <= 1
 0 <= y2 <= 1
 0 <= y3 <= 1

Binary
 y1 y2 y3

End
