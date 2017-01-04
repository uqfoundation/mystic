#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Support Vector Regression. Example 1
"""

from numpy import *
import qld
import pylab
from mystic.svr import *

from mystic.tools import random_seed
#random_seed(123)

x = arange(-5, 5.001)
y = x + random.rand(x.size)*7
#y = array([9.99, 10.93, 9.11, 6.50, 12.69, 14.09, 13.83, 15.86, 17.49, 14.20, 19.98])

l = x.size
N = 2*l

pylab.plot(x, y, 'k+')
#pylab.show()

# the Kernel Matrix (with the linear kernel)
X = concatenate([x,-x])
Q = multiply.outer(X,X)+1

# the linear term for the QP
Y = concatenate([y,-y])
svr_epsilon = 3;
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

R = RegressionFunction(x, y, alpha, svr_epsilon, LinearKernel)

pylab.plot(x,R(x))
pylab.plot(x,R(x)+svr_epsilon, 'k--')
pylab.plot(x,R(x)-svr_epsilon, 'k--')
pylab.plot(x[sv1],y[sv1],'ro')
pylab.plot(x[sv2],y[sv2],'go')

print alpha

pylab.show()

# $Id$
# 
# end of file
