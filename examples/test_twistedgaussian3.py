#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
In test_twistedgaussian, we compared SCEM (one chain) with metropolis.

Now we will use n-chain SCEM / in parallel.
"""

from test_twistedgaussian import *

a = initpop(q*m, n)
b = list(map(target, a))
a,b = sort_ab_with_b(a,b)
Cs = sequential_deal(a, q)
As = [xx.tolist() for xx in sequential_deal(b, q)]

def scemmap(Q):
    from test_twistedgaussian import scem
    Cs, As, Sk, Sak, target, cn = Q
    niter = 1000
    for i in range(niter):
        scem(Cs, As, Sk, Sak, target, 0.1)
    return Cs,As,Sk,Sak
        

if __name__=='__main__':
    import time
    from mystic.metropolis import *
    try:
        xrange
    except NameError:
        xrange = range
    # if available, use a multiprocessing worker pool
    try:
        from pathos.helpers import freeze_support, shutdown
        freeze_support() # help Windows use multiprocessing
        from pathos.pools import ProcessPool as Pool
        map = Pool().map
    except ImportError:
        shutdown = lambda x=None:None
        pass

    Sk = [ [Cs[i][0]] for i in xrange(q) ]
    Sak = [ [As[i][0]] for i in xrange(q) ]

    args = [(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1) for chain in xrange(q)]

    for iter in xrange(5):
       # this is parallel
       print("iteration: %s" % str(iter+1))

       res = list(map(scemmap, args))

       Cs = [x[0] for x in res]
       As = [x[1] for x in res]
       Sk = [x[2] for x in res]
       Sak = [x[3] for x in res]

       # need to gather and remix
       Cs , As = remix(Cs, As)

       args = [(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1) for chain in xrange(q)]

    
    from mystic.tools import flatten_array

    Sk = [a[100:] for a in Sk] # throw away the first 100 pts of each chain
    sk = flatten_array(Sk,1)
    #import dill
    #print("Writing to data file")
    #dill.dump(sk, open('tg3.pkl','w'))
    shutdown()
    
    try:
        import matplotlib.pyplot as plt
    except:
        print("Install matplotlib for visualization")
    else:
        plt.plot(sk[:,0],sk[:,1],'r.')
        plt.show()

# end of file
