#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
 
""" NOTE:
due to pickling issues, cost function is provided w/o using a factory method.
(same as chebyshev8.py, except uses global target,polyeval,poly1d)
"""

from mystic.models.poly import chebyshev8coeffs as target
from mystic.math import polyeval, poly1d

def chebyshev8cost(trial,M=61):
    """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""

    result=0.0
    x=-1.0
    dx = 2.0 / (M-1)
    for i in range(M):
        px = polyeval(trial, x)
        if px<-1 or px>1:
            result += (1 - px) * (1 - px)
        x += dx

    px = polyeval(trial, 1.2) - polyeval(target, 1.2)
    if px<0: result += px*px

    px = polyeval(trial, -1.2) - polyeval(target, -1.2)
    if px<0: result += px*px

    return result

if __name__=='__main__':
    print(poly1d(target))
    print("")
    print(chebyshev8cost(target))
