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

from g03 import objective, bounds, xs, ys

from mystic.penalty import quadratic_equality
from mystic.constraints import with_penalty

#sum([x0**2, x1**2, x2**2]) - 1.0 = 0.0
@with_penalty(quadratic_equality)
def penalty(x): # == 0.0
    x = [xi**2 for xi in x] # x**2
    return abs(sum(x) - 1)

from mystic.constraints import as_constraint

solver = as_constraint(penalty)



if __name__ == '__main__':
    x = xs(3)
    y = ys(len(x))
    bounds = bounds(len(x))

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, npop=40, gtol=500, disp=False, full_output=True)

    assert almostEqual(result[0], x, tol=1e-2)
    assert almostEqual(result[1], y, tol=1e-2)



# EOF
