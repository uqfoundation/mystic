#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Support Vector Classification. Example 1
"""

from numpy import *
import matplotlib.pyplot as plt
from mystic.svc import *

# define the objective function to match standard QP solver
# (see: http://www.mathworks.com/help/optim/ug/quadprog.html)
# the objective function is very similar to the dual for SVC
# (see: http://scikit-learn.org/stable/modules/svm.html#svc)
def objective(x, Q, b):
    return 0.5 * dot(dot(x,Q),x) + dot(b,x)

# define the data points for each class
c1 = array([[0., 0.],[1., 0.],[ 0.2, 0.2],[0.,1.]])
c2 = array([[0, 1.1], [1.1, 0.],[0, 1.5],[0.5,1.2],[0.8, 1.7]])

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
ub = 99999 * ones(nx)

# build the constraints operator
from mystic.symbolic import linear_symbolic, solve, \
     generate_solvers as solvers, generate_constraint as constraint
constrain = linear_symbolic(Aeq,Beq)
constrain = constraint(solvers(solve(constrain,target=['x0'])))

from mystic import suppressed
@suppressed(1e-3)
def conserve(x):
    return constrain(x)

from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)

# solve the dual for alpha
from mystic.solvers import DifferentialEvolutionSolver as DESolver
from mystic.termination import Or, ChangeOverGeneration, CollapseAt
ndim = len(lb)
npop = nx*3
stop = Or(ChangeOverGeneration(1e-8,200),CollapseAt(0.0))
solver = DESolver(ndim,npop)
solver.SetRandomInitialPoints(min=lb,max=ub)
solver.SetStrictRanges(min=lb,max=ub)
solver.SetGenerationMonitor(mon)
solver.SetConstraints(conserve)
solver.SetTermination(stop)
solver.Solve(objective, ExtraArgs=(Q,b), disp=1)
alpha = solver.bestSolution

print('solved x: %s' % alpha)
print("constraint A*x == 0: %s" % inner(Aeq, alpha))
print("minimum 0.5*x'Qx + b'*x: %s" % solver.bestEnergy)

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
