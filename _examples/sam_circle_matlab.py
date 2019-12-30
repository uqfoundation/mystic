#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Solve the dual form of test_circle.py with matlab's quadprog (via sam)
"""

from numpy import *
import matplotlib.pyplot as plt
from test_circle import sv, xy, x0, y0, R0
import sam

# The dual problem verification begins here.
npt = xy.shape[0]
Q = dot(xy, transpose(xy))
#Q = zeros((npt,npt))
#for i in range(npt):
#   for j in range(npt):
##       Q[i,j] = dot(xy[i,:],xy[j,:])
#Q = array(Q)
f = -diag(Q)+10
H = Q*2
A = ones((1,npt))
b = ones(1)
sam.putarray('H',H);
sam.putarray('f',f);
sam.eval("npt = %d;" % npt);
sam.eval("al = quadprog(H,f,[],[],ones(1,npt),1,zeros(npt,1),ones(npt,1));")
alpha = sam.getarray('al').flatten()


def getobj(H,f, x):
    return 0.5 * dot(dot(x,H),x) + dot(f,x)

def chop(x):
    if abs(x) > 1e-6:
        return x
    else:
        return 0

x = array([chop(y) for y in alpha])
center = dot(alpha,xy)
# find support vectors (find numpy way please)
sv = []
for i,v in enumerate(x):
   if v > 0.001: sv.append(i)
sv0 = sv[0]
R = linalg.norm(xy[sv0,:]-center)

def plot():
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
    plot()

# $Id$
# 
# end of file
