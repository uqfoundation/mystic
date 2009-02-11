#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
forward_model example
"""
from mystic.forward_model import *


if __name__=='__main__':
    from forward_mogi import ForwardMogiFactory
    import random
    from numpy import *
    xstations = array([random.uniform(-500,500) for i in range(300)])+1250.
    ystations =  0*xstations - 200.
    stations  = array((xstations, ystations))

    A = CostFactory()
    A.addModel(ForwardMogiFactory, 'mogi1', 4, outputFilter=PickComponent(2))
    A.addModel(ForwardMogiFactory, 'mogi2', 4, outputFilter=PickComponent(2))
    fe = A.getForwardEvaluator(stations)
    p = [random.random() for i in range(8)]
    c =  fe(p)
    print len(c)
    print sum(c).shape
    print A


# End of file
