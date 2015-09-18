#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Support Vector Regression. Example 2
"""

from numpy import *
import pylab
from mystic.svmtools import *

# a common objective function for solving a QP problem
# (see http://www.mathworks.com/help/optim/ug/quadprog.html)
def objective(x, H, f):
    return 0.5 * dot(dot(x,H),x) + dot(f,x)

# quadratic data plus scatter
x = arange(-5, 5.001)
y = x*x + 8*random.rand(x.size)

l = x.size
N = 2*l

# the Kernel Matrix 
X = concatenate([x,-x])
def pk(a1,a2):
    return (1+a1*a2)*(1+a1*a2)
Q = KernelMatrix(X, pk)

# the linear term for the QP
Y = concatenate([y,-y])
svr_epsilon = 4;
b = Y + svr_epsilon * ones(Y.size)

H = Q
f = b
Aeq = concatenate([ones(l), -ones(l)]).reshape(1,N)
Beq = array([0.])
lb = zeros(N)
ub = zeros(N) + 0.5

from mystic.symbolic import linear_symbolic, solve, \
     generate_solvers as solvers, generate_constraint as constraint
constrain = linear_symbolic(Aeq,Beq)
constrain = constraint(solvers(solve(constrain,target=['x0'])))

from mystic import supressed
@supressed(1e-5)
def conserve(x):
    return constrain(x)

from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)

from mystic.solvers import diffev
alpha = diffev(objective, zip(lb,.1*ub), args=(H,f), npop=N*3, gtol=200, \
               itermon=mon, \
               ftol=1e-5, bounds=zip(lb,ub), constraints=conserve, disp=1)

print 'solved x: ', alpha
print "constraint A*x == 0: ", inner(Aeq, alpha)
print "minimum 0.5*x'Hx + f'*x: ", objective(alpha, H, f)

sv1 = SupportVectors(alpha[:l])
sv2 = SupportVectors(alpha[l:])

R = RegressionFunction(x, y, alpha, svr_epsilon, pk)

print 'support vectors: ', sv1, sv2

# plot data
pylab.plot(x, y, 'k+', markersize=10)

# plot regression function and support
xx = arange(min(x),max(x),0.1)
pylab.plot(xx,R(xx))
pylab.plot(xx,R(xx)-svr_epsilon, 'r--')
pylab.plot(xx,R(xx)+svr_epsilon, 'g--')
pylab.plot(x[sv1],y[sv1],'ro')
pylab.plot(x[sv2],y[sv2],'go')
pylab.show()

# end of file
