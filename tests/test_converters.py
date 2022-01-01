#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
test _to_function and _to_objective in cost and gradient calculations
"""

import numpy as np
x = np.random.rand(10,3)

def cost(x):
   return sum(np.sin(x)**2) + 1

y = cost(x.T)

import mystic.math.interpolate as ip
fx = ip._to_function(cost)
assert (fx(*x.T) - cost(x.T) ).sum() < 0.0001

f = ip.interpf(x, y, method='linear', arrays=True)
cf = ip._to_objective(f)
assert (f(*x.T) - cf(x.T) ).sum() < 0.0001

assert f(*x[0].T) == cf(x[0].T)
assert fx(*x[0].T) == cost(x[0].T)

assert ip.gradient(x, f, method='linear').shape \
    == ip.gradient(x, fx, method='linear').shape


from mystic.models import rosen0der as rosen
y = rosen(x.T)

f = ip.interpf(x, y, method='linear')
fx = ip._to_function(rosen)
cf = ip._to_objective(f)

assert f(*x[0].T) == cf(x[0].T)
assert fx(*x[0].T) == rosen(x[0].T)
#print( fx(*x[0].T) - rosen(x[0].T) )

assert ip.gradient(x, f, method='linear').shape \
    == ip.gradient(x, fx, method='linear').shape

