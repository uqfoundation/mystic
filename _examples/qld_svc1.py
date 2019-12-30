#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Support Vector Classification. Example 1
"""

from numpy import *
import qld, quapro
import matplotlib.pyplot as plt
from mystic.svc import *

def myshow():
    import Image, tempfile
    name = tempfile.mktemp()
    plt.savefig(name,dpi=72)
    im = Image.open('%s.png' % name)
    im.show()

c1 = array([[0., 0.],[1., 0.],[ 0.2, 0.2],[0.,1.]])
c2 = array([[0, 1.1], [1.1, 0.],[0, 1.5],[0.5,1.2],[0.8, 1.7]])

plt.plot(c1[:,0], c1[:,1], 'ko')
plt.plot(c2[:,0], c2[:,1], 'ro')

# the Kernel Matrix (with the linear kernel)
# Q = multiply.outer(X,X) <--- unfortunately only works when X is a list of scalars...
# In Mathematica, this would be implemented simply via Outer[K, X, X, 1]

XX = concatenate([c1,-c2])
nx = XX.shape[0]

# quadratic and linear terms of QP
Q = KernelMatrix(XX)
b = -1 * ones(nx)

H = Q
f = b
Aeq = concatenate([ones(c1.shape[0]), -ones(c2.shape[0])]).reshape(1,nx)
Beq = array([0])
lb = zeros(nx)
ub = zeros(nx) + 99999

#alpha = qld.quadprog2(H, f, None, None, Aeq, Beq, lb, ub)
alpha, xxx= quapro.quadprog(H, f, None, None, Aeq=Aeq, beq=Beq, LB=lb, UB=ub)
print(alpha)
#alpha = array([fil(x) for x in alpha])
print("cons:%s" % inner(Aeq,alpha))
print("obj min: 0.5 * x'Hx + <x,f> %s" % 0.5*inner(alpha,inner(H,alpha))+inner(f,alpha))


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

print(wv)
print("%s %s" % (sv1, sv2))
print(bias)
print(bias2)

# Eqn of hyperplane:
# wv[0] x + wv[1] y + bias = 0
hx = array([0, 1.2])
hy = -wv[0]/wv[1] * hx - bias/wv[1]

plt.plot(hx, hy, 'k-')
myshow()

# $Id$
# 
# end of file
