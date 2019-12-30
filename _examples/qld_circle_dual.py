#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Solve the dual form of test_circle.py.

Currently, it uses a package called "qld" that I wrote but not in
the repo. yet. It wraps IQP from bell-labs. (code not GPL and has export
restrictions.)
"""

from numpy import *
import matplotlib.pyplot as plt
from test_circle import sparse_circle, sv, x0, y0, R0
getpoints = sparse_circle.forward
import qld

def getobjective(H,f, x):
    return 0.5 * dot(dot(x,H),x) + dot(f,x)

def chop(x):
    if abs(x) > 1e-6:
        return x
    else:
        return 0

def round(x):
    return array([chop(y) for y in x])


def plot(xy, sv, x0, y0, R0, center, R):
    import matplotlib.pyplot as plt
    plt.plot(xy[:,0],xy[:,1],'k+')
    plt.plot(xy[sv,0],xy[sv,1],'ro')
    theta = arange(0, 2*pi, 0.02)
    plt.plot([center[0]],[center[1]],'bo')
    plt.plot([xy[sv0,0], center[0]],[xy[sv0,1], center[1]],'r--')
    plt.plot(R0 * cos(theta)+x0, R0*sin(theta)+y0, 'r-',linewidth=2)
    plt.plot(R * cos(theta)+center[0], R*sin(theta)+center[1], 'b-',linewidth=2)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    npt = 20
    from test_circle import xy
    npt1 = xy.shape[0]
    if npt is not npt1:
        xy = getpoints((x0,y0,R0),npt)
    else:
        pass
    Q = dot(xy, transpose(xy))
    f = -diag(Q)+10
    H = Q*2
    A = ones((1,npt))
    b = ones(1)
    x = qld.quadprog2(H, f, None, None, A, b, zeros(npt), ones(npt))

    center = dot(x,xy)
    print("center: %s" % center)
    # find support vectors (find numpy way please)
    
    sv = []
    for i,v in enumerate(x):
       if v > 0.001: sv.append(i)
    sv0 = sv[0]
    
    print(sv)
    R = linalg.norm(xy[sv0,:]-center)

    plot(xy, sv, x0, y0, R0, center, R)

# $Id$
# 
