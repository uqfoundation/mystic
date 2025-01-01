#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2024-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
test rbf interpolation
"""

import numpy as np
x = np.random.rand(1,3)

def cost(x):
   return sum(np.sin(x)**2) + 1

y = cost(x.T)

import mystic.math.interpolate as ip

# functions which are 1 at r = 0 reproduce a single datapoint
fx = ip.Rbf(*np.vstack((x.T, y)), function='gaussian', smooth=0.0)
assert (fx(*x.T) - y).sum() == 0.0
fx = ip.Rbf(*np.vstack((x.T, y)), function='multiquadric', smooth=0.0)
assert (fx(*x.T) - y).sum() == 0.0
fx = ip.Rbf(*np.vstack((x.T, y)), function='inverse', smooth=0.0)
assert (fx(*x.T) - y).sum() == 0.0

# functions which are 0 at r = 0 are singular with a single datapoint
fx = ip.Rbf(*np.vstack((x.T, y)), function='linear', smooth=-1e-100)
assert fx(*x.T).sum() == 0.0
fx = ip.Rbf(*np.vstack((x.T, y)), function='cubic', smooth=-1e-100)
assert fx(*x.T).sum() == 0.0
fx = ip.Rbf(*np.vstack((x.T, y)), function='quintic', smooth=-1e-100)
assert fx(*x.T).sum() == 0.0
fx = ip.Rbf(*np.vstack((x.T, y)), function='thin_plate', smooth=-1e-100)
assert fx(*x.T).sum() == 0.0

