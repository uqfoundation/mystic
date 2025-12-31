#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2025-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
import numpy as np
from mystic.tools import random_seed
random_seed(17)
from mystic.math import Distribution

size = 10
lb = [0]*4
ub = [10]*4
npts = 20
std = 1
dist = Distribution('numpy.random.normal', 0, std)
X = np.random.random(size=(size,4)) * 10
y = np.random.random(size=size) * 100

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(0,10)
ax.set_ylim3d(0,10)
ax.set_zlim3d(0,10)

from mystic.math.grid import fillpts, samplepts, errorpts
data = np.array(sorted(errorpts(lb, ub, npts, data=X, error=None, mtol=.05, dist=None)))#dist)))
#data = np.array(sorted(fillpts(lb, ub, npts, data=X, dist=None)))#dist)))
#data = np.array(sorted(samplepts(lb, ub, npts, dist=None)))#dist)))

a,b,c,d = data.T
print(a.min(),a.max())
print(b.min(),b.max())
print(c.min(),c.max())
print(d.min(),d.max())

a0,b0,c0,d0 = X.T
ax.scatter(a0,b0,c0,c=d0, cmap=plt.hot(), vmin=0,vmax=10, marker='s', edgecolors='k', s=30)
img = ax.scatter(a,b,c,c=d, cmap=plt.hot(), vmin=0,vmax=10, edgecolors='r', s=30)
fig.colorbar(img)
plt.show()
