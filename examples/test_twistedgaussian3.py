#!/usr/bin/env python

"""
In test_twistedgaussian, we compared SCEM (one chain) with metropolis.

Now we will use n-chains SCEM / in pseudo parallel.
"""

from test_twistedgaussian import *

a = initpop(q*m, n)
b = map(target, a)
a,b = sort_ab_with_b(a,b)
Cs = sequential_deal(a, q)
As = [xx.tolist() for xx in sequential_deal(b, q)]

def scemmap(Q):
    Cs, As, Sk, Sak, target, cn = Q
    niter = 1000
    for i in xrange(niter):
        scem(Cs, As, Sk, Sak, target, 0.1)
    return Cs,As,Sk,Sak
        

if __name__=='__main__':
    from mystic.metropolis import *
    import time

    Sk = [ [Cs[i][0]] for i in range(q) ]
    Sak = [ [As[i][0]] for i in range(q) ]

    args = [(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1) for chain in range(q)]

    for iter in range(5):
       # this is parallel
       print iter+1

       res = map(scemmap, args)

       Cs = [x[0] for x in res]
       As = [x[1] for x in res]
       Sk = [x[2] for x in res]
       Sak = [x[3] for x in res]

       # need to gather and remix
       Cs , As = remix(Cs, As)

       args = [(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1) for chain in range(q)]

    
    from mystic import flatten_array

    Sk = [a[100:] for a in Sk] # throw away the first 100 pts of each chain
    sk = flatten_array(Sk,1)
    import scipy.io, cPickle
    print "Writing to data file"
    scipy.io.write_array(open('tg3.dat','w'), sk)
    #cPickle.dump(sk, open('tg3.dat','w'))
    
    try:
        import pylab
    except:
        print "No pylab"
    else:
        pylab.plot(sk[:,0],sk[:,1],'r.')
        pylab.show()

# end of file
