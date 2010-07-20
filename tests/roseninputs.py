#!/usr/bin/env python
#
# Alta Fang & Mike McKerns, Caltech

"""
roseninputs.py -- inputs for testing the rosenbrock function for testsolvers_pyre.py
"""

from mystic.models import rosen as cost
from mystic.termination import *

ND = 3

# for Differential Evolution: 
NP = 30

from numpy import inf
import random
random.seed(123)

x0 = [0.8, 1.2, 0.5]

# used with SetStrictRanges
#min_bounds = [-0.999, -0.999, 0.999]     
#max_bounds = [200.001, 100.001, inf]    

termination = CandidateRelativeTolerance()
#termination = VTR()
#termination = ChangeOverGeneration()
#termination = NormalizedChangeOverGeneration()

# End of file
