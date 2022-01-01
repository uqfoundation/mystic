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

from g11 import objective, bounds, xs, xs_, ys

from mystic.penalty import quadratic_equality
from mystic.constraints import with_penalty

@with_penalty(quadratic_equality, k=1e12)
def penalty(x): # == 0.0
    return x[1] - x[0]**2

from mystic.constraints import as_constraint

solver = as_constraint(penalty)



if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, constraints=solver, npop=40, xtol=1e-8, ftol=1e-8, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2) \
        or almostEqual(result[0], xs_, tol=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF
