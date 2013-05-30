#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                 Mike McKerns, Alta Fang, & Patrick Hung, Caltech
#                        (C) 1997-2012  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# get version numbers, license, and long description
try:
    from info import this_version as __version__
    from info import readme as __doc__, license as __license__
except ImportError:
    msg = """First run 'python setup.py build' to build mystic."""
    raise ImportError(msg)

__author__ = 'Mike McKerns'

__doc__ = """
""" + __doc__

__license__ = """
""" + __license__

__all__ = ['solvers', 'termination', 'strategy', 'munge', 'tools', \
           'cache', 'constraints', 'penalty', 'coupler', 'symbolic']

# solvers
import solvers

# strategies, termination conditions
import termination
import strategy

# constraints and penalties
import constraints
import penalty
import coupler
import symbolic

# monitors, function wrappers, and other tools
import monitors
import munge
import cache
import tools

# backward compatibility
from tools import *


def license():
    """print license"""
    print __license__
    return

def citation():
    """print citation"""
    print __doc__[-521:-140]
    return

# end of file
