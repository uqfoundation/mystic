#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
dummy.py -- cost function container module for derun.py
"""

from mystic.models.poly import chebyshev8cost as cost
from mystic.termination import *
from mystic.strategy import *

ND = 9
NP = 80
MAX_GENERATIONS = ND*NP

min = [-100.0] * 9
max = [100.0] * 9

termination = VTR(0.01)

probability = 1.0
scale = 0.9


# End of file
