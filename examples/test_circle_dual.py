#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Solve the dual form of test_circle.py.

Given a set of points in the plane, find the smallest circle
that contains them.

Requires:
  -- numpy, matplotlib

The matplotlib output will draw 
  -- a set of points inside a circle defined by x0,y0,R0 
  -- the circle (x0,y0) with rad R0
  -- the optimized circle with minimum R enclosing the points
"""

from numpy import *
import matplotlib.pyplot as plt
from test_circle import sparse_circle, circle, sv, x0, y0, R0

# a common objective function for solving a QP problem
# (see http://www.mathworks.com/help/optim/ug/quadprog.html)
def objective(x, H, f):
    return 0.5 * dot(dot(x,H),x) + dot(f,x)

# plot the data and results
def plot(xy, sv, x0, y0, R0, center, R):
    import matplotlib.pyplot as plt
    plt.plot(xy[:,0],xy[:,1],'k+')
    plt.plot(xy[sv,0],xy[sv,1],'bx',mfc="None")
    plt.plot([x0],[y0],'ro')
    plt.plot([center[0]],[center[1]],'bo')
    plt.plot([xy[sv[0],0], center[0]],[xy[sv[0],1], center[1]],'b--')
    c = circle(x0,y0,R0)
    plt.plot(c[:,0], c[:,1], 'r-', linewidth=2)
    c = circle(center[0],center[1],R)
    plt.plot(c[:,0], c[:,1], 'b-', linewidth=2)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    npt = 20
    from test_circle import xy
    npt1 = xy.shape[0]
    if npt is not npt1:
        xy = sparse_circle(x0,y0,R0,npt)
    else:
        pass

    # define a QP problem for the objective
    Q = dot(xy, transpose(xy))
    f = -diag(Q)+10
    H = Q*2
    # define lower and upper bounds on x
    LB = zeros(npt)
    UB = ones(npt)
    # define equality constraints (A*x == b)
    A = ones((1,npt))
    b = ones(1)

#   # generic: build a constraint where (A*x == b)
#   from mystic.symbolic import linear_symbolic, solve, \
#        generate_solvers as solvers, generate_constraint as constraint
#   norm = constraint(solvers(solve(linear_symbolic(A,b),target=['x0'])))

    # specific: build a constraint where (sum(x) == 1.0)
    from mystic.math.measures import normalize
    def norm(x):
        return normalize(x, mass=1.0)

    # solve the constrained quadratic programming problem
    from mystic.solvers import fmin_powell
    x = fmin_powell(objective, UB/npt, args=(H, f), bounds=list(zip(LB,UB)), \
                    constraints=norm, ftol=1e-8, disp=1)

    # find support vectors
    sv = where(x > 0.001)[0]
    print("support vectors: %s" % sv)

    # compare solved center and radius to generating center and radius
    center = dot(x,xy)
    R = linalg.norm(xy[sv[0],:]-center)
    print("x0, y0: (%f, %f) @ R0 = %f" % (x0,y0,R0))
    print("center: (%f, %f) @ R  = %f" % (center[0],center[1],R))

    plot(xy, sv, x0, y0, R0, center, R)


# EOF
