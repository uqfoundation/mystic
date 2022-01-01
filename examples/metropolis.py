#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
exercises the Metropolis-Hastings algorithm
"""

from mystic.metropolis import metropolis_hastings
from numpy import mean, cov

if __name__=='__main__':
    from numpy import random
    import time
    try:
        xrange
    except NameError:
        xrange = range

    def prop(x):
        return random.normal(x, 1)

    def target(x):
        if x <0 or x > 3:
            return 0
        return 1. + 0.31831/(0.0025 + (x-1)**2) + 45. *x - 10. * x**2

    t1 = time.time()
    x = [1]
    for i in xrange(100000):
        x.append(metropolis_hastings(prop, target, x[-1]) )
    t2 = time.time()
    print('Metropolis took %0.3f ms' % ((t2-t1)*1000 ))

    import matplotlib.pyplot as plt
    plt.hist(x,20)
    plt.show()


# end of file
