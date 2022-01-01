#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
The Twisted Gausssian in
"Shuffled Complex Evolution Metropolis" Algoritm of Vrugt et al. [1]

Reference:

[1] Jasper A. Vrugt, Hoshin V. Gupta, Willem Bouten, and Soroosh Sorooshian
A Shuffled Complex Evolution Metropolis algorithm for optimization and uncertainty assessment of hydrologic model parameters
WATER RESOURCES RESEARCH, VOL. 39, NO. 8, 1201, doi:10.1029/2002WR001642, 2003 
Link to paper: http://www.agu.org/pubs/crossref/2003/2002WR001642.shtml

[2] Vrugt JA, Nuallain , Robinson BA, Bouten W, Dekker SC, Sloot PM
Application of parallel computing to stochastic parameter estimation in environmental models
Computers & Geosciences, Vol. 32, No. 8. (October 2006), pp. 1139-1155.
Link to paper: http://www.science.uva.nl/research/scs/papers/archive/Vrugt2006b.pdf

"""

from mystic.scemtools import *
from numpy import zeros, identity, array
from numpy import random

# dimension of density function
n = 2

# number of parallel chains
q = 10

# number of points per complex
m = 100

m1 = zeros(n)
S1 = identity(n)
S1[0,0] = 100

def twist(X):
    b = 0.1
    Y = array(X)*1.
    Y[1] += b * X[0]**2 - 100. * b
    return Y

p = multinormal_pdf(m1,S1)
def target(X):
    return p(twist(X))     

def proposal(X):
    return random.multivariate_normal(X, 10. * identity(n))

def initpop(npts, ndim):
    return random.rand(npts, ndim) * 200. -1

a = initpop(q*m, n)
Cs = sort_and_deal(a, target, q)
Ck = Cs[0] # 0 for the first deal, -1 for the last
ak = [target(c) for c in Ck]
Sk = [ Ck[0] ]
Sak = [ ak[0] ]
L = 10000

if __name__=='__main__':
    from mystic.metropolis import *
    import time
    try:
        xrange
    except NameError:
        xrange = range

    t1 = time.time()
    for i in xrange(L):
        scem(Ck, ak, Sk, Sak, target, 0.1)
    t2 = time.time()
    print("SCEM 1 chain for x[%d] took %0.3f ms" % (len(Sk), (t2-t1)*1000))
    Sk = array(Sk)

    t1 = time.time()
    x = [ [0,10] ]
    for i in xrange(L):
        x.append(metropolis_hastings(proposal, target, x[-1]))
    t2 = time.time()
    print("2D Metropolis for x[%d] took %0.3f ms" % (len(x), (t2-t1)*1000))
    x = array(x)

    #import dill
    #dill.dump(x, open('twisted1.pkl','w'))
    #dill.dump(Sk, open('twisted1.pkl','w'))

    import matplotlib.pyplot as plt
    plt.plot(Sk[:,0],Sk[:,1],'r.')
    plt.plot(x[:,0] + 30,x[:,1],'b.')
    plt.show()

# end of file
