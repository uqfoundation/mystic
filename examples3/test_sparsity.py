#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
finds npts that are rtol dist away from legacy data
"""
#XXX: could make similar that uses partitioning (based on monitor.y) ?
#XXX: could make similar that first samples, then uses hole-filling ?

#data=[[2,-6],[5,-7],[1,0],[1,3],[5,9],[2,2],[-10,-10],[-10,10],[10,-10]]
#data=[[0,0],[-10,-10],[-10,0],[10,0],[0,-10],[0,10],[10,-10],[-10,10],[10,10]]
data=[[0,0]]
#data = []

ndim = len(data[0]) if data else 2
lb,ub = [-10]*ndim, [10]*ndim

npts = 12
rtol = None # 4.5

#from mystic.math import Distribution
#from scipy.stats import norm
dist = None # Distribution(norm,0,2)


if __name__ == '__main__':
    from mystic.math import fillpts
    pts = fillpts(lb, ub, npts, data, rtol, dist)

    import numpy as np
    print(np.array(pts).reshape(-1,ndim))

    import matplotlib.pyplot as plt

    _x,_y = np.array(pts).reshape(-1,ndim).T
    x,y = np.array(data).reshape(-1,ndim).T

    plt.scatter(x=x,y=y)
    plt.scatter(x=_x,y=_y)
    plt.show()

