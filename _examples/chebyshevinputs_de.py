#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
chebychevinputs_de.py -- cost function container module for differential evolution 
and the chebyshev function for testsolvers_pyre.py
"""

from mystic.models.poly import chebyshev8cost as cost
from mystic.termination import *
from mystic.strategy import *

ND = 9
NP = 80
maxiter = ND*NP

# used with SetRandomInitialPoints
min = [-100.0] * 9
max = [100.0] * 9


strategy = Best1Exp
#strategy = Best1Bin
#strategy = Rand1Exp

probability = 1.0
scale = 0.9

solverkwds = {'strategy': strategy, 'CrossProbability': probability,
              'ScalingFactor': scale}


#termination = VTR()
termination = CandidateRelativeTolerance()
#termination = ChangeOverGeneration()
#termination = NormalizedChangeOverGeneration()


# End of file
