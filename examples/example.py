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
example.py
"""

from mystic.models.poly import chebyshev8cost as cost
from mystic.detools import *

ND = 9
NP = 80
MAX_GENERATIONS = ND*NP

min = [-100.0] * 9
max = [100.0] * 9

termination = VTR(0.01)

probability = 1.0
scale = 0.9


# End of file
