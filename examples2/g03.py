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

def objective(x):
    from numpy import prod, sqrt
    n = len(x)
    return -sqrt(n)*n * prod(x)

def bounds(len=3):
    return [(0.0,1.0)]*len

# with penalty='penalty' applied, solution is:
def xs(len=3):
    from math import sqrt
    return [1./sqrt(len)]*len
def ys(len=3):
    return objective(xs(len))

"""
for len(x) == 3, x* = 0.57735027 for all xi, y* = -1.0
for len(x) == 10, x* = 0.31622777 for all xi. y* = -0.00031623
for len(x) == 20, x* = 0.22360680 for all xi, y* = -8.73464054e-12
"""

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions

#sum([x0**2, x1**2, x2**2]) - 1.0 = 0.0
def equations(len=3):
    eqn = "\nsum(["
    for i in range(len):
        eqn += 'x%s**2, ' % str(i)
    return eqn[:-2]+"]) - 1.0 = 0.0\n"

def cf(len=3):
    return generate_constraint(generate_solvers(simplify(equations(len))))
def pf(len=3):
    return generate_penalty(generate_conditions(equations(len)))



if __name__ == '__main__':
    x = xs(10)
    y = ys(len(x))
    bounds = bounds(len(x))
    cf = cf(len(x))

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, constraints=cf, npop=40, gtol=500, disp=False, full_output=True)

    assert almostEqual(result[0], x, tol=1e-2)
    assert almostEqual(result[1], y, tol=1e-2)



# EOF
