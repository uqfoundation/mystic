#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Support Vector Classification.

SMO prototype. 
"""

from numpy import *
import matplotlib.pyplot as plt
from mystic.svc import *

# a common objective function for solving a QP problem
# (see http://www.mathworks.com/help/optim/ug/quadprog.html)
def objective(x, H, f):
    return 0.5 * dot(dot(x,H),x) + dot(f,x)

c1 = array([[0., 0.],[1., 0.],[ 0.2, 0.2],[0.,1.]])
c2 = array([[0, 1.1], [1.1, 0.],[0, 1.5],[0.5,1.2],[0.8, 1.7]])

# the Kernel Matrix (with the linear kernel)
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
ub = 99999 * ones(nx)

from mystic.symbolic import linear_symbolic, solve, \
     generate_solvers as solvers, generate_constraint as constraint
constrain = linear_symbolic(Aeq,Beq)
constrain = constraint(solvers(solve(constrain,target=['x0'])))

from mystic import suppressed
@suppressed(1e-5)
def conserve(x):
    return constrain(x)

#from mystic.monitors import VerboseMonitor
#mon = VerboseMonitor(1)

from mystic.solvers import diffev
alpha = diffev(objective, list(zip(lb,ub)), args=(H,f), npop=nx*3, gtol=200, \
#              itermon=mon, \
               ftol=1e-8, bounds=list(zip(lb,ub)), constraints=conserve, disp=1)

print('solved x: %s' % alpha)
print("constraint A*x == 0: %s" % inner(Aeq, alpha))
print("minimum 0.5*x'Hx + f'*x: %s" % objective(alpha, H, f))


# let's play. We will need to bootstrap the SMO with an initial
# state that belongs to the feasible set. Because of the special structure
# of SVMs, the zero vector suffices. We will use that here. 
# More generally, an initial point can be obtained via solving an LP.

def getIndexSets(alpha, a, b, y):
    """See Kerthi and Gilbert, and MathSVM.m"""
    # it is either one loop, or five smart ops.
    # will do former so it can be translated into C
    I0,I1,I2,I3,I4 = [],[],[],[],[]
    for i, ali, ai, bi, yi in zip(range(y.size), alpha, a, b, y):
        if ali > ai and ali < bi:
          I0.append(i)
        elif ali == ai:
          if yi > 0:
             I1.append(i)
          else:
             I4.append(i)
        else:
          if yi < 0:
             I2.append(i)
          else:
             I3.append(i)
    return I0,I1,I2,I3,I4

def getIub(I0, I1, I2, I3, I4):
    return I0+I1+I2 

def getIlb(I0, I1, I2, I3, I4):
    return I0+I3+I4 

def QPOptimalQ(Iub, Ilb, F, tau):
    return max(F[Ilb]) - min(F[Iub]) <= tau

def ViolatingPairQ(Sub, Slb, i, j, F, tau):
    # Sub and Slb should now be sets
    if Sub.__contains__(i) and Slb.__contains__(j) and F[j] - F[i] > tau:
        return True
    elif Slb.__contains__(i) and Sub.__contains__(j) and F[i] - F[j] > tau:
        return True
    else:
        return False

def getViolatingPair(Iint, Iub, Ilb, F, tau):
    # The heuristic is to focus attention to Iint
    FIint = F[Iint]
    if FIint and max(FIint) - min(FIint) > tau:
         # the first index set (is nonempty and) has offending elements.
         ihigh,ilow = FIint.argmax(), FIint.argmin()
         return Iint[ilow], Iint[ihigh]
    else:
         # nothing within the first index set, need to scan
         # MathSVM is dumber than usual
         Sub,Slb = set(Iub), set(Ilb)
         l = F.size
         for i in range(l):
             for j in range(i,l):
                 if ViolatingPairQ(Sub, Slb, i, j, F, tau):
                      return i,j
    # the fact that this function is called means that it 
    # should never reach here
    return None


def smo_sub2d(Q, y, alpha, X, i, j, ym, yp):
    # 2d subproblem for SMO, varying only indices i and j
    a1,a2 = alpha[i], alpha[j]
    y1,y2 = y[i], y[j]
    wv = WeightVector(alpha, X, y)
    ii = inner(wv, X)
    bias = -0.5 * (max(ii[ym]) + min(ii[yp]))
    ay = transpose(alpha * y)
    TT = dot(Q, transpose(alpha * y)) + b
    E1, E2 = TT[i] - y1, TT[j] - y2
    k = Q[i,i] + Q[j,j] - 2. * Q[i,j]
    C = b[0] # note, i am now assuming that a=0, b=c
    if y1 != y2:
        U = max(0, a2-a1)
        V = min(C, C-a1 + a2)
        print("1 U/V: %s %s" % (U, V))
    else:
        U = max(0., a1+a2-C)
        V = min(C, a1+a2)
        print("2 U/V: %s %s" % (U, V))
    a2trial = a2 + y2*(E1-E2)/k
    if a2trial > V:
        a2new = V
    elif a2trial < U:
        a2new = U
    else:
        a2new = a2trial
    a1new = a1 + y1*y2*(a2-a2new)
    print("E1/E2: %s %s" % (E1,E2))
    print("C: %s" % C)
    return a1new, a2new

def QP_smo(Q, p, a, b, c, y, tau, a0, X):
    """\
Minimizes xQx + Px, 
   subject to a_i <= x_i <= b_i for all i
   and y.x = c

   X are the array of input points (their labels should be in y)
    """
    # initial alpha (must be feasible)
    alpha = a0  
    l = alpha.size
    ym,yp = (y<0).nonzero()[0], (y>0).nonzero()[0]
    while 1:
        Isets = getIndexSets(alpha, a, b, y)
        Iub = getIub(*Isets)
        Ilb = getIlb(*Isets)
        F = (dot(Q, alpha) + p)/y
        if QPOptimalQ(Iub, Ilb, F, tau):
            break
        B = getViolatingPair(Isets[0], Iub, Ilb, F, tau)
        # now solve the 2d subproblem
        aa = smo_sub2d(Q, y, alpha, X, B[0], B[1], ym,yp)
        print("aa: %s" % str(aa))
        alpha[B[0]], alpha[B[1]] = aa
        print(alpha)
        break

X = concatenate([c1,c2])
y = Aeq.flatten()
p,a,b,c = f, lb, ub, 0
QP_smo(Q, p, a, b, c, y, 0.01, a, X)

# end of file
