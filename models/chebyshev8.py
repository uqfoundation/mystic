#!/usr/bin/env python

"""
Chebyshev8 functions
"""

def cost(trial):
    """
The costfunction for the Chebyshev8 fitting problem. See examples/test_ffit.py
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
