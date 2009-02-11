#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
example.py
"""

from mystic.polytools import polyeval
from mystic.detools import *

ND = 9
NP = 80
MAX_GENERATIONS = ND*NP

min = [-100.0] * 9
max = [100.0] * 9

termination = VTR(0.01)

probability = 1.0
scale = 0.9

def cost(trial):
    """
The costfunction for the Chebyshev8 fitting problem. See tests/test_ffit.py
    """
    M=60 # number of evaluation points between [-1, 1]
    result=0.0
    x=-1.0
    dx = 2.0 / (M)
    for i in range(M+1):
        px = polyeval(trial, x)
        if px<-1 or px>1:
            result += (1 - px) * (1 - px)
        x += dx

    px = polyeval(trial, 1.2) - 72.661
    if px<0: result += px*px

    px = polyeval(trial, -1.2) - 72.661
    if px<0: result += px*px

    return result


# End of file
