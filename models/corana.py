#!/usr/bin/env python

"""
from test_corana.py
"""

def Corana(x):
    """
The costfunction for Corana's parabola. Eq. (11) of [1]
    """
    from math import pow
    from numpy import sign, floor
    r = 0
    for j in range(4):
        zj =  floor( abs(x[j]/0.2) + 0.49999 ) * sign(x[j]) * 0.2
        if abs(x[j]-zj) < 0.05:
            r += 0.15 * pow(zj - 0.05*sign(zj), 2) * d[j]
        else:
            r += d[j] * x[j] * x[j]
    return r

def Corana1(x):
    return Corana([x[0], 0, 0, 0])


# End of file
