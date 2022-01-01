#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Implements the "Shuffled Complex Evolution Metropolis" Algoritm
of Vrugt et al. [1]

  
Reference:

[1] Jasper A. Vrugt, Hoshin V. Gupta, Willem Bouten, and Soroosh Sorooshian
A Shuffled Complex Evolution Metropolis algorithm for optimization and
uncertainty assessment of hydrologic model parameters,
WATER RESOURCES RESEARCH, VOL. 39, NO. 8, 1201, doi:10.1029/2002WR001642, 2003 
http://www.agu.org/pubs/crossref/2003/2002WR001642.shtml

[2] Vrugt JA, Nuallain , Robinson BA, Bouten W, Dekker SC, Sloot PM
Application of parallel computing to stochastic parameter estimation in
environmental models,
Computers & Geosciences, Vol. 32, No. 8. (October 2006), pp. 1139-1155.
http://www.science.uva.nl/research/scs/papers/archive/Vrugt2006b.pdf

"""

import numpy
try:
    getlimits = numpy.lib.getlimits
except AttributeError:
    getlimits = numpy.core.getlimits

TYPE_DOUBLE =  getlimits.finfo(getlimits.numeric.double)
TINY = TYPE_DOUBLE.tiny

def multinormal_pdf(mean, var):
    """var must be symmetric positive definite """
    import numpy
    vinv = numpy.linalg.inv(var)
    mu = numpy.array(mean).flatten()
    n = len(mu)

    # check that var is properly sized
    dum = numpy.dot(mu,var) - numpy.dot(var,mu)

    prefactor = 1./numpy.sqrt((2 * numpy.pi)**n  * numpy.linalg.det(var))
    def _(x):
        xm = numpy.array(x) - mu
        return prefactor *  numpy.exp(-0.5 * numpy.dot(xm,numpy.dot(vinv,xm)) )
    return _

def sequential_deal(inarray, n):
    """
- inarray: should be a set of N objects (the objects can be vectors themselves, 
  but inarray should be index-able like a list. It is coerced into a numpy array
  because the last operations requires that it is also indexable by a 'list.'
    
- it should have a length divisble by n, otherwise the reshape will fail 
  (this is a feature !) 

- sequential_deal(range(20), 5) wil return a 5 element list, each element being
  a 4-list of index. (see below)

>>> for l in sequential_deal(range(20),5):
...    print(l)
...
[ 0  5 10 15]
[ 1  6 11 16]
[ 2  7 12 17]
[ 3  8 13 18]
[ 4  9 14 19]

    """
    import numpy
    cards = numpy.array(inarray)
    N = len(cards)
    # this bit of numpy will give, for N=20, n = 5
    # ord = [ [0,5,10,15], [1,6,11,16], [2,7,12,17], [3,8,13,18], [4,9,14,19] ]
    ord = numpy.transpose(numpy.array(list(range(N))).reshape(N//n, n))
    return [cards[x] for x in ord] 

def sort_and_deal(cards, target, nplayers):
    import numpy
    c = numpy.array(list(map(target, cards)))
    o = list(reversed(numpy.argsort(c)))
    # from best to worst
    sorted_deck = cards[o]
    return sequential_deal(sorted_deck, nplayers)

def sort_ab_with_b(a, b, ord = -1):
    """default is descending..."""
    import numpy
    aa,bb = numpy.array(a), numpy.array(b)
    if ord == -1:
        o = list(reversed(bb.argsort()))
    else:
        o = b.argsort()
    return aa[o], bb[o]

def sort_complex0(c, a):
    # this is dumb, because c (i.e., a, are almost sorted)
    # should use the one below instead.
    # this is faster than sort_complex
    import numpy
    b = numpy.array(a)
    o = list(reversed(b.argsort()))
    return c[o], list(b[o])

def sort_complex(c, a):
    # this is dumb, because c (i.e., a, are almost sorted)
    # should use the one below instead.
    import numpy
    D = list(zip(a,c))
    D.sort(key=lambda x:x[0], reverse=True)
    return numpy.array([x[1] for x in D]), [x[0] for x in D]

def myinsert(a, x):
    # a is in descending order
    from bisect import bisect
    return bisect([-i for i in a],-x)

def sort_complex2(c, a):
    """
- c and a are partially sorted (either the last one is bad, or the first one)
- pos : 0 (first one out of order)
       -1 (last one out of order)
    """
    # find where new key is relative to the rest
    if a[0] < a[1]:
        ax, cx = a[0], c[0]
        ind = myinsert(a[1:],ax)
        a[0:ind], a[ind] = a[1:1+ind], ax
        c[0:ind], c[ind] = c[1:1+ind], cx
    elif a[-1] > a[-2]:
        ax,cx = a[-1], c[-1]
        ind = myinsert(a[0:-1],ax)
        a[ind+1:], a[ind] = a[ind:-1], ax
        c[ind+1:], c[ind] = c[ind:-1], cx
    return

def update_complex(Ck, ak, c, a, pos):
    """
- ak is sorted (descending)
- Ck[pos] and ak[pos] will be removed, and then
  c and a spliced in at the proper place
- pos is 0, or -1
    """
    if pos == 0:
        at = ak[1:]
    else:
        at = ak[:-1]

    p = myinsert(at, a)

    # not done yet.
    if pos == 0:
        Ck[0:p], ak[p] = Ck[1:p+1], ak[1:p+1]
        Ck[p], ak[p] = c, a
    else:
        Ck[0:p], ak[p] = Ck[1:p+1], ak[1:p+1]
        Ck[p], ak[p] = c, a

    return

def scem(Ck, ak, Sk, Sak, target, cn):
    """
This is the SCEM algorithm starting from line [35] of the reference [1].

- [inout] Ck is the kth 'complex' with m points. This should be an m by n array
  n being the dimensionality of the density function. i.e., the data
  are arranged in rows. 

  Ck is assumed to be sorted according to the target density.

- [inout] ak, the density of the points of Ck.

- [inout] Sk, the entire chain. (should be a list)

- [inout] Sak, the cost of the entire chain (should be a list)

  Sak would be more convenient to use if it is a numpy array, but we need
  to append to it frequently.

- [in] target: target density function

- [in] cn: jumprate. (see Paragraph 37 of [1.]

- The invariants: ak is always aligned with Ck, and are the cost of Ck
- Similarly, Sak is always aligned with Sk in the same way.

- On return... sort order in Ck/ak is destroyed. but see sort_complex2

    """
    import numpy
    from mystic.tools import random_state
    prng = random_state(module=numpy.random)

    # function level constants
    T = 100000. # predefined likelihood ratio. Paragraph 45 of [1.]

    # Sort Ck according to ak
    #Ck, ak = sort_complex0(Ck, ak) 
    #print("ak before: %s" % ak)
    #Ck, ak = sort_complex(Ck, ak) 
    sort_complex2(Ck, ak) 
    #print("ak after: %s" % ak)

    # number of points per complex
    M = Ck.shape[0]

    mu = numpy.mean(Ck,0) # row mean

    # (numpy.cov takes data in columns)
    Sigma = numpy.cov(numpy.transpose(Ck)) 

    # Gamma (line 35 of [1]. Best to Worst)
    Gamma = ak[0] / (ak[-1]+TINY)
   
    if len(Sak) >= M:
        meansak = numpy.mean(Sak[-M:])
    else:
        meansak = numpy.mean(Sak)
    alpha_k = numpy.mean(ak) / (meansak+TINY)
  
    if alpha_k < T:
         # Paragraph 37 of [1]
         basept = Sk[-1]
    else: # numpy.mean(asak) is very close to zero.
         # Paragraph 38 of [1]
         basept = mu

    # TODO should take a proposal instead !
    Yt = prng.multivariate_normal(basept, cn*cn * Sigma)
    cY = target(Yt)    

    # print("new/orig : %s %s" % (cY, Sak[-1]))

    r = min( cY / (Sak[-1]+TINY), 1)

    if prng.rand() <= r:
        Sk.append(Yt)
        Sak.append(cY)
        # Paragraph 43 of [1] (update the best of Ck)
        Ck[0], ak[0] = Sk[-1], Sak[-1]
    else:
        Sk.append(Sk[-1])
        Sak.append(Sak[-1])
        if Gamma > T and Sak[-1] > ak[-1]:
            # Paragraph 43 of [1] (update the worst of Ck)
            Ck[-1], ak[-1] = Sk[-1], Sak[-1]

    # mainly because of the logic in Paragraph [43] that updates the complex,
    # this function is not "functional" and modifies its argument instead.
    # hence no return value
    return     

def remix(Cs, As):
    """
Mixing and dealing the complexes.
The types of Cs and As are very important.... 
    """
    from mystic.tools import flatten_array
    q = len(Cs)
    c2, a2 =  list(flatten_array(Cs,1)), list(flatten_array(As,1))
    a,b = sort_ab_with_b(c2,a2)
    C2 = sequential_deal(a, q)
    A2 = [xx.tolist() for xx in sequential_deal(b, q)]
    return C2, A2

# end of file
