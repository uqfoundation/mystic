#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/a/71197490/2379433
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Minimize y = f(x):
  where y > g(x)

  with f(x) = x0^(sin(x0)) + x1^4 + 6x1^3 - 5x1^2 - 40x1 + 35
  and g(x) = 7x1 - x0 + 5
  in x = [0,10]
"""
import numpy as np
import mystic as my


def f(x):
    x0,x1 = x
    return x0**np.sin(x0) + x1**4 + 6*x1**3 - 5*x1**2 - 40*x1 + 35

def g(x):
    x0,x1 = x
    return 7*x1 - x0 + 5

def penalty(x):  # <= 0.0
    return g(x) - f(x)

@my.penalty.quadratic_inequality(penalty, k=1e12)
def p(x):
    return 0.0

# monitor
mon = my.monitors.VerboseMonitor(1,1)
my.solvers.fmin(f, [5,5], bounds=[(0,10)]*2, penalty=p, itermon=mon, disp=1)

my.log_reader(mon)

fig = my.model_plotter(f, mon, depth=True, scale=1.0, bounds="0:10, 0:10", out=True)

from matplotlib import cm
import matplotlib.pyplot as plt

x,y = my.scripts._parse_axes("0:10, 0:10", grid=True)
x, y = np.meshgrid(x, y)
z = 0*x
s,t = x.shape
for i in range(s):
    for j in range(t):
        xx,yy = x[i,j], y[i,j]
        z[i,j] = g([xx,yy])

z = np.log(4*z*1.0+1)+2 # scale=1.0

ax = fig.axes[0]
ax.contour(x, y, z, 50, cmap=cm.cool)
plt.show()
