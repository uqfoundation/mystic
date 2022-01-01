#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
    Fit linear and quadratic polynomial to noisy data:
               y(x) ~ a + b * x   --or--   y(x) ~ a + b * x + c * x**2

    where:
               0 >= x >= 4
               y(x) = y0(x) + yn
               y0(x) = 1.5 * exp(-0.2 * x) + 0.3
               yn = 0.1 * Normal(0,1)
"""
from numpy import polyfit, poly1d, linspace, exp
from numpy.random import normal
from mystic.math import polyeval
from mystic import reduced

# Create clean data.
x = linspace(0, 4.0, 100)
y0 = 1.5 * exp(-0.2 * x) + 0.3

# Add a bit of noise.
noise = 0.1 * normal(size=100) 
y = y0 + noise

@reduced(lambda x,y: abs(x)+abs(y))
def objective(coeffs, x, y):
    return polyeval(coeffs, x) - y

bounds  = [(None, None), (None, None)]
bounds_ = [(None, None), (None, None), (None, None)]

args = (x, y)

# 'solution' is:
xs  = polyfit(x, y, 1) 
xs_ = polyfit(x, y, 2) 
ys  = objective(xs, x, y)
ys_ = objective(xs_, x, y)


if __name__ == '__main__':

  from mystic.solvers import diffev2, fmin_powell
  from mystic.math import almostEqual
# from mystic.monitors import VerboseMonitor
# mon = VerboseMonitor(10)

  result = diffev2(objective, args=args, x0=bounds, bounds=bounds, npop=40, ftol=1e-8, gtol=100, disp=False, full_output=True)#, itermon=mon)
# print("%s %s" % (result[0], xs))
  assert almostEqual(result[0], xs, rel=1e-1)
  assert almostEqual(result[1], ys, rel=1e-1)

  result = fmin_powell(objective, args=args, x0=[0.0,0.0], bounds=bounds, disp=False, full_output=True)
# print("%s %s" % (result[0], xs))
  assert almostEqual(result[0], xs, rel=1e-1)
  assert almostEqual(result[1], ys, rel=1e-1)

# mon = VerboseMonitor(10)
  result = diffev2(objective, args=args, x0=bounds_, bounds=bounds_, npop=40, ftol=1e-8, gtol=100, disp=False, full_output=True)#, itermon=mon)
# print("%s %s" % (result[0], xs_))
  assert almostEqual(result[0], xs_, tol=1e-1)
  assert almostEqual(result[1], ys_, rel=1e-1)

  result = fmin_powell(objective, args=args, x0=[0.0,0.0,0.0], bounds=bounds_, disp=False, full_output=True)
# print("%s %s" % (result[0], xs_))
  assert almostEqual(result[0], xs_, tol=1e-1)
  assert almostEqual(result[1], ys_, rel=1e-1)


# EOF
