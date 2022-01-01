#!/usr/bin/env python
#
# Problem definition:
# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# Original Matlab code written by A. Hedar (Nov. 23, 2005)
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm
# and ported to Python by Mike McKerns (December 2014)
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from g05 import objective, bounds, xs, ys

from mystic.constraints import as_constraint
from mystic.penalty import quadratic_inequality, quadratic_equality

def penalty1(x): # <= 0.0
    return x[2] - x[3] - 0.55

def penalty2(x): # <= 0.0
    return x[3] - x[2] - 0.55

def penalty3(x): # == 0.0
    from math import sin
    return abs(1000*(sin(-x[2]-.25) + sin(-x[3]-0.25)) + 894.8 - x[0])

def penalty4(x): # == 0.0
    from math import sin
    return abs(1000*(sin(x[2]-.25) + sin(x[2]-x[3]-0.25)) + 894.8 - x[1])

def penalty5(x): # == 0.0
    from math import sin
    return abs(1000*(sin(x[3]-.25) + sin(x[3]-x[2]-0.25)) + 1294.8)

@quadratic_inequality(penalty1, k=1e12)
@quadratic_inequality(penalty2, k=1e12)
@quadratic_equality(penalty3, k=1e12)
@quadratic_equality(penalty4, k=1e12)
@quadratic_equality(penalty5, k=1e12)
def penalty(x):
    return 0.0

solver = as_constraint(penalty)



if __name__ == '__main__':

    from mystic.solvers import buckshot, sparsity
    from mystic.math import almostEqual

    result = buckshot(objective, len(xs), npts=100, bounds=bounds, penalty=penalty, disp=False, full_output=True)
    #result = sparsity(objective, len(xs), npts=100, rtol=-10, bounds=bounds, penalty=penalty, disp=False, full_output=True)

    assert almostEqual(result[0], xs, rel=1e-1)
    assert almostEqual(result[1], ys, rel=1e-1)



# EOF
