#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
In test_twistedgaussian, we compared SCEM (one chain) with metropolis.

Now we will use n-chain SCEM / in parallel.
"""

from test_twistedgaussian import *

a = initpop(q*m, n)
b = map(target, a)
a,b = sort_ab_with_b(a,b)
Cs = sequential_deal(a, q)
As = [xx.tolist() for xx in sequential_deal(b, q)]

def scemmap(Q):
    from test_twistedgaussian import scem
    Cs, As, Sk, Sak, target, cn = Q
    niter = 1000
    for i in xrange(niter):
        scem(Cs, As, Sk, Sak, target, 0.1)
    return Cs,As,Sk,Sak
        

if __name__=='__main__':
    import time
    from mystic.metropolis import *
    # if available, use a multiprocessing worker pool
    try:
        from pathos.helpers import freeze_support
        freeze_support() # help Windows use multiprocessing
        from pathos.pools import ProcessPool as Pool
        map = Pool().map
    except ImportError:
        pass

    Sk = [ [Cs[i][0]] for i in range(q) ]
    Sak = [ [As[i][0]] for i in range(q) ]

    args = [(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1) for chain in range(q)]

    for iter in range(5):
       # this is parallel
       print "iteration: %s" % str(iter+1)

       res = map(scemmap, args)

       Cs = [x[0] for x in res]
       As = [x[1] for x in res]
       Sk = [x[2] for x in res]
       Sak = [x[3] for x in res]

       # need to gather and remix
       Cs , As = remix(Cs, As)

       args = [(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1) for chain in range(q)]

    
    from mystic.tools import flatten_array

    Sk = [a[100:] for a in Sk] # throw away the first 100 pts of each chain
    sk = flatten_array(Sk,1)
    #import dill
    #print "Writing to data file"
    #dill.dump(sk, open('tg3.pkl','w'))
    
    try:
        import pylab
    except:
        print "Install matplotlib for visualization"
    else:
        pylab.plot(sk[:,0],sk[:,1],'r.')
        pylab.show()

# end of file
