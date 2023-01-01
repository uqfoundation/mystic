#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Support Vector Regression. Example 2
"""
#FIXME: works if x is a 1D array, should allow x as a 2D array
from numpy import *
import matplotlib.pyplot as plt
from mystic.svr import *

# define the objective function to match standard QP solver
# (see: http://www.mathworks.com/help/optim/ug/quadprog.html)
def objective(x, Q, b):
    return 0.5 * dot(dot(x,Q),x) + dot(b,x)

# define the data points (quadratic data with uniform scatter)
x = arange(-5, 5.001); nx = x.size
y = x*x + 8*random.rand(nx)
N = 2*nx

# build the Kernel Matrix (with custom k)
# get the QP quadratic term
X = concatenate([x,-x])
pk = lambda a1,a2: PolynomialKernel(a1,a2,2)
Q = KernelMatrix(X, kernel=pk)
# get the QP linear term
Y = concatenate([y,-y])
svr_epsilon = 4
b = Y + svr_epsilon * ones(Y.shape)

# build the constraints (y.T * x = 0.0)
# 1.0*x0 + 1.0*x1 + ... - 1.0*xN = 0.0
Aeq = concatenate([ones(nx), -ones(nx)]).reshape(1,N)
Beq = array([0.])
# set the bounds
lb = zeros(N)
ub = zeros(N) + 0.5
_b = zeros(N) + 0.1

# build the constraints operator
from mystic.symbolic import linear_symbolic, solve, \
     generate_solvers as solvers, generate_constraint as constraint
constrain = linear_symbolic(Aeq,Beq)
constrain = constraint(solvers(solve(constrain,target=['x0'])))

from mystic import suppressed
@suppressed(1e-5)
def conserve(x):
    return constrain(x)

from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)

# solve for alpha
from mystic.solvers import DifferentialEvolutionSolver as DESolver
from mystic.termination import Or, ChangeOverGeneration, CollapseAt
ndim = len(lb)
npop = 3*N
stop = Or(ChangeOverGeneration(1e-8,200),CollapseAt(0.0))
solver = DESolver(ndim,npop)
solver.SetRandomInitialPoints(min=lb,max=_b)
solver.SetStrictRanges(min=lb,max=ub)
solver.SetGenerationMonitor(mon)
solver.SetConstraints(conserve)
solver.SetTermination(stop)
solver.Solve(objective, ExtraArgs=(Q,b), disp=1)
alpha = solver.bestSolution

print('solved x: %s' % alpha)
print("constraint A*x == 0: %s" % inner(Aeq, alpha))
print("minimum 0.5*x'Qx + b'*x: %s" % solver.bestEnergy)

# calculate support vectors and regression function
sv1 = SupportVectors(alpha[:nx])
sv2 = SupportVectors(alpha[nx:])
R = RegressionFunction(x, y, alpha, svr_epsilon, pk)

print('support vectors: %s %s' % (sv1, sv2))

# plot data
plt.plot(x, y, 'k+', markersize=10)

# plot regression function and support
xx = arange(min(x),max(x),0.1)
plt.plot(xx,R(xx))
plt.plot(xx,R(xx)-svr_epsilon, 'r--')
plt.plot(xx,R(xx)+svr_epsilon, 'g--')
plt.plot(x[sv1],y[sv1],'ro')
plt.plot(x[sv2],y[sv2],'go')
plt.show()

# end of file
