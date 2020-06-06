#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

# get version numbers, license, and long description
try:
    from mystic.info import this_version as __version__
    from mystic.info import readme as __doc__, license as __license__
except ImportError:
    msg = """First run 'python setup.py build' to build mystic."""
    raise ImportError(msg)

__author__ = 'Mike McKerns'

__doc__ = """
""" + __doc__

__license__ = """
""" + __license__

__all__ = ['solvers','termination','strategy','munge','tools','support', \
           'constraints','penalty','coupler','symbolic','monitors','license', \
           'citation','model_plotter','log_reader','collapse_plotter']

# solvers
import mystic.solvers as solvers

# strategies, termination conditions
import mystic.termination as termination
import mystic.strategy as strategy

# constraints and penalties
import mystic.constraints as constraints
import mystic.penalty as penalty
import mystic.coupler as coupler
import mystic.symbolic as symbolic

# monitors, function wrappers, and other tools
import mystic.monitors as monitors
import mystic.munge as munge
import mystic.tools as tools

# scripts
from mystic.scripts import model_plotter, log_reader, collapse_plotter
import mystic.support as support

# backward compatibility
from mystic.tools import *


def license():
    """print license"""
    print(__license__)
    return

def citation():
    """print citation"""
    print(__doc__[-510:-115])
    return

# end of file
