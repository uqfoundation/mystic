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

from g01 import objective, bounds, xs, ys

from mystic.constraints import as_constraint
from mystic.penalty import quadratic_inequality

def penalty1(x): # <= 0.0
    return 2*x[0] + 2*x[1] + x[9] + x[10] - 10.0

def penalty2(x): # <= 0.0
    return 2*x[0] + 2*x[2] + x[9] + x[11] - 10.0

def penalty3(x): # <= 0.0
    return 2*x[1] + 2*x[2] + x[10] + x[11] - 10.0

def penalty4(x): # <= 0.0
    return -8*x[0] + x[9]

def penalty5(x): # <= 0.0
    return -8*x[1] + x[10]

def penalty6(x): # <= 0.0
    return -8*x[2] + x[11]

def penalty7(x): # <= 0.0
    return -2*x[3] - x[4] + x[9]

def penalty8(x): # <= 0.0
    return -2*x[5] - x[6] + x[10]

def penalty9(x): # <= 0.0
    return -2*x[7] - x[8] + x[11]

@quadratic_inequality(penalty1)
@quadratic_inequality(penalty2)
@quadratic_inequality(penalty3)
@quadratic_inequality(penalty4)
@quadratic_inequality(penalty5)
@quadratic_inequality(penalty6)
@quadratic_inequality(penalty7)
@quadratic_inequality(penalty8)
@quadratic_inequality(penalty9)
def penalty(x):
    return 0.0

solver = as_constraint(penalty)



if __name__ == '__main__':
    x = [0]*len(xs)

    from mystic.solvers import fmin_powell
    from mystic.math import almostEqual

    result = fmin_powell(objective, x0=x, bounds=bounds, penalty=penalty, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2)
    assert almostEqual(result[1], ys, tol=1e-2)



# EOF
