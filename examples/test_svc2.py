#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Support Vector Classification. Example 2.

using meristem data from data files
"""

from numpy import *
import matplotlib.pyplot as plt
from mystic.svc import *
import os.path

# define the objective function to match standard QP solver
# (see: http://www.mathworks.com/help/optim/ug/quadprog.html)
# the objective funciton is very similar to the dual for SVC
# (see: http://scikit-learn.org/stable/modules/svm.html#svc)
def objective(x, Q, b):
    return 0.5 * dot(dot(x,Q),x) + dot(b,x)

# SETTINGS
reduced = True  # use a subset of the full data
overlap = False # reduce the distance between the datasets

# define the data points for each class
c1 = loadtxt(os.path.join('DATA','g1.pts'))
c2 = loadtxt(os.path.join('DATA','g2.pts'))

if reduced:
    c1 = c1[c1[:,0] > 245]
    c2 = c2[c2[:,0] < 280]
if overlap: c1[:,0] += 3 # make the datasets overlap a little

# define the full data set
X = concatenate([c1,c2]); nx = X.shape[0]
# define the labels (+1 for c1; -1 for c2)
y = concatenate([ones(c1.shape[0]), -ones(c2.shape[0])]).reshape(1,nx)

# build the Kernel Matrix
# get the QP quadratic and linear terms
XX = concatenate([c1,-c2])
Q = KernelMatrix(XX)  # Q_ij = K(x_i, x_j)
b = -ones(nx)         # b_i = 1  (in dual)

# build the constraints (y.T * x = 0.0)
# 1.0*x0 + 1.0*x1 + ... - 1.0*xN = 0.0
Aeq = y
Beq = array([0.])
# set the bounds
lb = zeros(nx)
ub = 1 * ones(nx)
_b = .1 * ones(nx) # good starting value if most solved xi should be zero

# build the constraints operator
from mystic.symbolic import linear_symbolic, solve, simplify, \
     generate_solvers as solvers, generate_constraint as constraint
constrain = linear_symbolic(Aeq,Beq)
#NOTE: HACK assumes a single equation of the form: '1.0*x0 + ... = 0.0\n'
x0,rhs = constrain.strip().split(' = ')
x0,xN = x0.split(' + ', 1) 
N,x0 = x0.split("*")
constrain = "{x0} = ({rhs} - ({xN}))/{N}".format(x0=x0, xN=xN, N=N, rhs=rhs)
#NOTE: end HACK (as mystic.symbolic.solve takes __forever__)
constrain = constraint(solvers(constrain))
#constrain = constraint(solvers(solve(constrain)))

from mystic import suppressed
@suppressed(5e-2)
def conserve(x):
    return constrain(x)

from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)

# solve the dual for alpha
from mystic.solvers import diffev
alpha = diffev(objective, list(zip(lb,_b)), args=(Q,b), npop=nx*3, gtol=200,\
               itermon=mon, \
               ftol=1e-8, bounds=list(zip(lb,ub)), constraints=conserve, disp=1)

print('solved x: %s' % alpha)
print("constraint A*x == 0: %s" % inner(Aeq, alpha))
print("minimum 0.5*x'Qx + b'*x: %s" % objective(alpha, Q, b))

# calculate weight vectors, support vectors, and bias
wv = WeightVector(alpha, X, y)
sv1, sv2 = SupportVectors(alpha,y,epsilon=1e-6)
bias = Bias(alpha, X, y)

ym = (y.flatten()<0).nonzero()[0]
yp = (y.flatten()>0).nonzero()[0]
ii = inner(wv, X)
bias2 = -0.5 *( max(ii[ym]) + min(ii[yp]) )

print('weight vector: %s' % wv)
print('support vectors: %s %s' % (sv1, sv2))
print('bias (from points): %s' % bias)
print('bias (with vectors): %s' % bias2)

# plot data
plt.plot(c1[:,0], c1[:,1], 'bo', markersize=5)
plt.plot(c2[:,0], c2[:,1], 'yo', markersize=5)

# plot hyperplane: wv[0] x + wv[1] y + bias = 0
xmin,xmax,ymin,ymax = plt.axis()
hx = array([floor(xmin-.1), ceil(xmax+.1)])
hy = -wv[0]/wv[1] * hx - bias/wv[1]
plt.plot(hx, hy, 'k-')
#plt.axis([xmin,xmax,ymin,ymax])

# plot the support points
plt.plot(XX[sv1,0], XX[sv1,1], 'bo', markersize=8)
plt.plot(-XX[sv2,0], -XX[sv2,1], 'yo', markersize=8)
#plt.axis('equal')
plt.show()

# end of file
