#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Support Vector Classification. Example 1
"""

from numpy import *
import pylab
from mystic.svctools import *

# a common objective function for solving a QP problem
# (see http://www.mathworks.com/help/optim/ug/quadprog.html)
def objective(x, H, f):
    return 0.5 * dot(dot(x,H),x) + dot(f,x)

c1 = array([[0., 0.],[1., 0.],[ 0.2, 0.2],[0.,1.]])
c2 = array([[0, 1.1], [1.1, 0.],[0, 1.5],[0.5,1.2],[0.8, 1.7]])

# the Kernel Matrix (with the linear kernel)
# Q = multiply.outer(X,X)   # NOTE: requires X is a list of scalars

XX = concatenate([c1,-c2])
nx = XX.shape[0]

# quadratic and linear terms of QP
Q = KernelMatrix(XX)
b = -1 * ones(nx)

H = Q
f = b
Aeq = concatenate([ones(c1.shape[0]), -ones(c2.shape[0])]).reshape(1,nx)
Beq = array([0.])
lb = zeros(nx)
ub = 99999 * ones(nx)

from mystic.symbolic import linear_symbolic, solve, \
     generate_solvers as solvers, generate_constraint as constraint
constrain = linear_symbolic(Aeq,Beq)
constrain = constraint(solvers(solve(constrain,target=['x0'])))

from mystic import supressed
@supressed(1e-5)
def conserve(x):
    return constrain(x)

#from mystic.monitors import VerboseMonitor
#mon = VerboseMonitor(1)

from mystic.solvers import diffev
alpha = diffev(objective, zip(lb,ub), args=(H,f), npop=nx*3, gtol=200, \
#              itermon=mon, \
               ftol=1e-8, bounds=zip(lb,ub), constraints=conserve, disp=1)

print 'solved x: ', alpha
print "constraint A*x == 0: ", inner(Aeq, alpha)
print "minimum 0.5*x'Hx + f'*x: ", objective(alpha, H, f)

# the labels and the points
X = concatenate([c1,c2])
y = concatenate([ones(c1.shape[0]), -ones(c2.shape[0])]).reshape(1,nx)

wv = WeightVector(alpha, X, y)
sv1, sv2 = SupportVectors(alpha,y,eps=1e-6)
bias = Bias(alpha, X, y)

ym = (y.flatten()<0).nonzero()[0]
yp = (y.flatten()>0).nonzero()[0]
ii = inner(wv, X)
bias2 = -0.5 *( max(ii[ym]) + min(ii[yp]) )

print 'weight vector: ', wv
print 'support vectors: ', sv1, sv2
print 'bias (from points): ', bias
print 'bias (with vectors): ', bias2

# plot data
pylab.plot(c1[:,0], c1[:,1], 'bo')
pylab.plot(c2[:,0], c2[:,1], 'yo')

# plot hyperplane: wv[0] x + wv[1] y + bias = 0
xmin,xmax,ymin,ymax = pylab.axis()
hx = array([floor(xmin-.1), ceil(xmax+.1)])
hy = -wv[0]/wv[1] * hx - bias/wv[1]
pylab.plot(hx, hy, 'k-')
#pylab.axis([xmin,xmax,ymin,ymax])
pylab.show()

# end of file
