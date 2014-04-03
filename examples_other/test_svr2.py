#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Support Vector Regression. Example 1
"""

from numpy import *
import qld
import pylab
from mystic.svmtools import *

x = arange(-5, 5.001)
y = x*x + random.rand(x.size)*8
#y = array([9.99, 10.93, 9.11, 6.50, 12.69, 14.09, 13.83, 15.86, 17.49, 14.20, 19.98])

l = x.size
N = 2*l

pylab.plot(x, y, 'k+')

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
Beq = array([0])
lb = zeros(N)
ub = zeros(N) + 0.5

alpha = qld.quadprog2(H, f, None, None, Aeq, Beq, lb, ub)
#alpha = 2*alpha

sv1 = SupportVectors(alpha[:l])
sv2 = SupportVectors(alpha[l:])

R = RegressionFunction(x, y, alpha, svr_epsilon, pk)

xx = arange(min(x),max(x),0.1)
pylab.plot(xx,R(xx))
pylab.plot(xx,R(xx)+svr_epsilon, 'k--')
pylab.plot(xx,R(xx)-svr_epsilon, 'k--')
pylab.plot(x[sv1],y[sv1],'ro')
pylab.plot(x[sv2],y[sv2],'go')

print alpha

pylab.show()

# $Id$
# 
# end of file
