#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
roseninputs.py -- inputs for testing the rosenbrock function for testsolvers_pyre.py
"""

from mystic.models import rosen as cost
from mystic.termination import *

ND = 3

# for Differential Evolution: 
NP = 30

from numpy import inf
from mystic.tools import random_seed
random_seed(123)

x0 = [0.8, 1.2, 0.5]

# used with SetStrictRanges
#min_bounds = [-0.999, -0.999, 0.999]     
#max_bounds = [200.001, 100.001, inf]    

termination = CandidateRelativeTolerance()
#termination = VTR()
#termination = ChangeOverGeneration()
#termination = NormalizedChangeOverGeneration()

# End of file
