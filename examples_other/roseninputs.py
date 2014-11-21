#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
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
