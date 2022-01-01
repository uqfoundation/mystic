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

Now we will use n-chain SCEM .
"""

from test_twistedgaussian import *

a = initpop(q*m, n)
b = list(map(target, a))
a,b = sort_ab_with_b(a,b)
Cs = sequential_deal(a, q)
As = [xx.tolist() for xx in sequential_deal(b, q)]

if __name__=='__main__':
    from mystic.metropolis import *
    import time
    try:
        xrange
    except NameError:
        xrange = range

    Sk = [ [Cs[i][0]] for i in xrange(q) ]
    Sak = [ [As[i][0]] for i in xrange(q) ]

    for iter in xrange(5):
       # this is parallel
       print("iteration: %s" % str(iter+1))
       for chain in xrange(q):
           for i in xrange(1000):
              scem(Cs[chain], As[chain], Sk[chain], Sak[chain], target, 0.1)

       
       # need to gather and remix
       Cs , As = remix(Cs, As)
    
    from mystic.tools import flatten_array

    Sk = [a[100:] for a in Sk] # throw away the first 100 pts of each chain
    sk = flatten_array(Sk,1)
    #print("length of sk: %s" % len(sk))
    
    import matplotlib.pyplot as plt
    plt.plot(sk[:,0],sk[:,1],'r.')
    plt.show()

# end of file
