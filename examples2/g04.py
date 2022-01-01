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
    x0,x1,x2,x3,x4 = x #XXX: allow x != 5?
    return 5.3578547*x2**2 + 0.8356891*x0*x4 + 37.293239*x0 - 40792.141

bounds = [(78,102),(33,45)] + [(27,45)]*3
# with penalty='penalty' applied, solution is:
xs = [78.0, 33.0, 29.9955776, 45.0, 36.7749999]
ys = -30665.488305434

def u(x):
    x0,x1,x2,x3,x4 = x
    return 85.334407 + 0.0056858*x1*x4 + 0.0006262*x0*x3 - 0.0022053*x2*x4

def v(x):
    x0,x1,x2,x3,x4 = x
    return 80.51249 + 0.0071317*x1*x4 + 0.0029955*x0*x1 + 0.0021813*x2*x2

def w(x):
    x0,x1,x2,x3,x4 = x
    return 9.300961 + 0.0047026*x2*x4 + 0.0012547*x0*x2 + 0.0019085*x2*x3


from mystic.penalty import quadratic_inequality

def penalty1(x): # <= 0.0
    return u(x) - 92.0

def penalty2(x): # <= 0.0
    return -u(x)

def penalty3(x): # <= 0.0
    return v(x) - 110.0

def penalty4(x): # <= 0.0
    return -v(x) + 90.0

def penalty5(x): # <= 0.0
    return w(x) - 25.0

def penalty6(x): # <= 0.0
    return -w(x) + 20.0

@quadratic_inequality(penalty1, k=1e10)
@quadratic_inequality(penalty2, k=1e10)
@quadratic_inequality(penalty3, k=1e10)
@quadratic_inequality(penalty4, k=1e10)
@quadratic_inequality(penalty5, k=1e10)
@quadratic_inequality(penalty6, k=1e10)
def penalty(x):
    return 0.0

from mystic.constraints import as_constraint

solver = as_constraint(penalty)



if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, npop=40, gtol=500, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF
