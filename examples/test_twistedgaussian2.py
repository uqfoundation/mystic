#!/usr/bin/env python

"""
In test_twistedgaussian, we compared SCEM (one chain) with metropolis.

Now we will use n-chains SCEM .
"""

from test_twistedgaussian import *

a = initpop(q*m, n)
b = map(target, a)
a,b = sort_ab_with_b(a,b)
Cs = sequential_deal(a, q)
As = [xx.tolist() for xx in sequential_deal(b, q)]

if __name__=='__main__':
    from mystic.metropolis import *
    import time

    Sk = [ [Cs[i][0]] for i in range(q) ]
    Sak = [ [As[i][0]] for i in range(q) ]

    for iter in range(5):
       # this is parallel
       print iter+1
       for chain in range(q):
           for i in xrange(1000):
              scem(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1)

       
       # need to gather and remix
       Cs , As = remix(Cs, As)
    
    from mystic import flatten_array

    Sk = [a[100:] for a in Sk] # throw away the first 100 pts of each chain
    sk = flatten_array(Sk,1)
    print len(sk)
    
    import pylab
    pylab.plot(sk[:,0],sk[:,1],'r.')
    pylab.show()

# end of file
