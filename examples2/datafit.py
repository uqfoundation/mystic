#!/usr/bin/env python
#
# Problem definition:
# Example in reference documentation for scipy.optimze
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
    Fit parameters to noisy data:
               y(x) ~ a * exp(-b * x) + c

    where:
               0 >= x >= 4
               y(x) = y0(x) + yn
               y0(x) = a0 * exp(-b0 * x) + c0
               a0,b0,c0 = 2.5,1.3,0.5
               yn = 0.2 * Normal(0,1)
"""
from numpy import exp, linspace
from numpy.random import normal
from mystic import reduced

def y0(coeffs, x):
    a,b,c = coeffs
    return a * exp(-b * x) + c

coeffs = (2.5, 1.3, 0.5)

# Create noisy data from the 'solution' parameters
x = linspace(0, 4, 50)
y = y0(coeffs, x) + 0.2 * normal(size=len(x))

@reduced(lambda x,y: abs(x)+abs(y))
def objective(coeffs, x, y):
    return y0(coeffs, x) - y

bounds = [(0, 10),(0, 10),(0, 10)]
args = (x,y)

# 'solution' is:
try:
    from scipy.optimize import curve_fit
    xs,pcov = curve_fit(lambda x,*coeffs: y0(coeffs,x), x, y, p0=[1,1,1])
except ImportError:
    xs = x0
ys = objective(xs, x, y)


if __name__ == '__main__':

  from mystic.solvers import diffev2
  from mystic.math import almostEqual
# from mystic.monitors import VerboseMonitor
# mon = VerboseMonitor(10)

  result = diffev2(objective, args=args, x0=bounds, bounds=bounds, npop=40, ftol=1e-8, gtol=100, disp=False, full_output=True)#, itermon=mon)
# print("%s %s" % (result[0], xs))
  assert almostEqual(result[0], xs, rel=2e-1)
  assert almostEqual(result[1], ys, rel=2e-1)

#XXX: how approximate the covariance matrix of estimates (pcov) w/ mystic?
#XXX: mystic should have leastsq


# EOF
