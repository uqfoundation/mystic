#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/least_square.py
# with Copyright 2011 Hakan Kjellerstrand hakank@bonetmail.com
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Least square optimization problem in Google or-tools.

  Solving a fourth grade least square equation.

  From the Swedish book 'Optimeringslara' [Optimization Theory],
  page 286f.
"""

p = 4
# number of points
npts = 14
# temperature
t = [20, 30, 80, 125, 175, 225, 275, 325, 360, 420, 495, 540, 630, 700]
# percentage gas
F = [0.0, 5.8, 14.7, 31.6, 43.2, 58.3, 78.4, 89.4, 96.4, 99.1, 99.5,
     99.9, 100.0, 100.0]

def objective(x):
    return sum([(F[i] - (sum([x[j] * t[i]**j for j in range(p + 1)])))
                                             for i in range(npts)])

bounds = [(-100,100)]*(p+1)
# with penalty='penalty' applied, solution is:
xs = [-1e2, -1e2, 5.55811273, -1.52656038e-2, 1.07572965e-5]
ys = -1046912.373722

#import random
#x0 = [random.randrange(i,j) for i,j in bounds]

# constraints
def penalty1(x): # == 0.0
    return sum([20**i * x[i] for i in range(p + 1)])

def penalty2(x): # == 0.0
    return (x[0] + sum([700.0**j * x[j] for j in range(1, p + 1)])) - 100.0

penalties = []
for i in range(npts):
    def dummy(x): # <= 0.0
        return -sum([j * x[j] * t[i]**(j - 1) for j in range(p + 1)])
    penalties.append(dummy)
# give the penalties names
p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13 = penalties

#XXX: better if numpy-ify the penalty functions?


from mystic.penalty import quadratic_equality, quadratic_inequality
from mystic.constraints import as_constraint

@quadratic_equality(penalty1)
@quadratic_equality(penalty2)
@quadratic_inequality(p0)
@quadratic_inequality(p1)
@quadratic_inequality(p2)
@quadratic_inequality(p3)
@quadratic_inequality(p4)
@quadratic_inequality(p5)
@quadratic_inequality(p6)
@quadratic_inequality(p7)
@quadratic_inequality(p8)
@quadratic_inequality(p9)
@quadratic_inequality(p10)
@quadratic_inequality(p11)
@quadratic_inequality(p12)
@quadratic_inequality(p13)
def penalty(x):
    return 0.0

solver = as_constraint(penalty)

def constraint(x):
    return x


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual
    from mystic.monitors import Monitor, VerboseMonitor
    mon = VerboseMonitor(10)#,10)

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, constraints=constraint, npop=40, ftol=1e-8, gtol=200, disp=True, full_output=True, cross=0.8, scale=0.9, itermon=mon)

    print(result[0])
    assert almostEqual(result[0], xs, rel=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)


# EOF
