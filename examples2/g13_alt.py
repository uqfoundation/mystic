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

from g13 import objective, bounds, xs, _xs, x_s, xs_, ys

from mystic.constraints import as_constraint
from mystic.penalty import quadratic_equality

def penalty1(x): # = 0.0
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10.0

def penalty2(x): # = 0.0
    return x[1]*x[2] - 5.0*x[3]*x[4]

def penalty3(x): # = 0.0
    return x[0]**3 + x[1]**3 + 1.0

@quadratic_equality(penalty1)
@quadratic_equality(penalty2)
@quadratic_equality(penalty3)
def penalty(x):
    return 0.0

solver = as_constraint(penalty)



if __name__ == '__main__':

    from mystic.solvers import lattice
    from mystic.math import almostEqual

    result = lattice(objective, 5, [2]*5, bounds=bounds, penalty=penalty, ftol=1e-8, xtol=1e-8, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=5e-2) \
        or almostEqual(result[0], _xs, tol=5e-2) \
        or almostEqual(result[0], x_s, tol=5e-2) \
        or almostEqual(result[0], xs_, tol=5e-2)
    assert almostEqual(result[1], ys, rel=5e-2)

# EOF
