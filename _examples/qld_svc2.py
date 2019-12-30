#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Support Vector Classification. Example 2.
meristem data.
"""

from numpy import *
from scipy.io import read_array
import qld, quapro
import matplotlib.pyplot as plt
from mystic.svc import *
import os.path, time

def myshow():
    import Image, tempfile
    name = tempfile.mktemp()
    plt.savefig(name,dpi=150)
    im = Image.open('%s.png' % name)
    im.show()

c1 = read_array(os.path.join('DATA','g1x.pts'))
c2 = read_array(os.path.join('DATA','g2.pts'))
c1[:,0] += 5 # to make the two sets overlap a little

# interchange ?
#c1, c2 = c2, c1

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
ub = zeros(nx) + 100000

#alpha = qld.quadprog2(H, f, None, None, Aeq, Beq, lb, ub)
alpha,xxx = quapro.quadprog(H, f, None, None, Aeq, Beq, lb, ub)
#t1 = time.time()
#for i in range(1000):
#   alpha,xxx = quapro.quadprog(H, f, None, None, Aeq, Beq, lb, ub)
#   #alpha = qld.quadprog2(H, f, None, None, Aeq, Beq, lb, ub)
#t2 = time.time()
#print("1000 calls to QP took %0.3f s" % (t2-t1))
print(alpha)

# the labels and the points
X = concatenate([c1,c2])
y = concatenate([ones(c1.shape[0]), -ones(c2.shape[0])]).reshape(1,nx)

wv = WeightVector(alpha, X, y)
sv1, sv2 = SupportVectors(alpha,y, eps=1e-6)
bias = Bias(alpha, X, y)

print(wv)
print("%s %s" % (sv1,sv2))
print(bias)

# Eqn of hyperplane:
# wv[0] x + wv[1] y + bias = 0

plt.plot(c1[:,0], c1[:,1], 'ko',markersize=2)
plt.plot(c2[:,0], c2[:,1], 'ro',markersize=2)
xmin,xmax,ymin,ymax = plt.axis()
hx = array([xmin, xmax])
hy = -wv[0]/wv[1] * hx - bias/wv[1]
plt.plot(hx, hy, 'k-')
plt.axis([xmin,xmax,ymin,ymax])
plt.plot(XX[sv1,0], XX[sv1,1], 'ko',markersize=5)
plt.plot(-XX[sv2,0], -XX[sv2,1], 'ro',markersize=5)
plt.axis('equal')
myshow()

# $Id$
# 
# end of file
