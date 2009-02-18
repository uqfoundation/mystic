#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1997-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


"""
A Simple Model-Independent Inversion Framework.
"""

# solvers
import differential_evolution, scipy_optimize

# strategies, termination conditions
import termination
import strategy

# monitors, function wrappers, and other tools
from tools import *


# end of file
