#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
forward_model example
"""
from mystic.forward_model import *


if __name__=='__main__':
    from mystic.models import mogi; ForwardMogiFactory = mogi.ForwardFactory
    import random
    from numpy import *
    xstations = array([random.uniform(-500,500) for i in range(300)])+1250.
    ystations =  0*xstations - 200.
    stations  = array((xstations, ystations))

    A = CostFactory()
    A.addModel(ForwardMogiFactory, 4, 'mogi1', outputFilter=PickComponent(2))
    A.addModel(ForwardMogiFactory, 4, 'mogi2', outputFilter=PickComponent(2))
    fe = A.getForwardEvaluator(stations)
    p = [random.random() for i in range(8)]
    c =  fe(p)
    print(len(c))
    print(sum(c).shape)
    print(A)


# End of file
