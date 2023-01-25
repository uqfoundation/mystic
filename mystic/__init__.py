#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

# author, version, license, and long description
try: # the package is installed
    from .__info__ import __version__, __author__, __doc__, __license__
except: # pragma: no cover
    import os
    import sys
    parent = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(parent)
    # get distribution meta info 
    from version import (__version__, __author__,
                         get_license_text, get_readme_as_rst)
    __license__ = get_license_text(os.path.join(parent, 'LICENSE'))
    __license__ = "\n%s" % __license__
    __doc__ = get_readme_as_rst(os.path.join(parent, 'README.md'))
    del os, sys, parent, get_license_text, get_readme_as_rst


__all__ = ['solvers','termination','strategy','munge','tools','support', \
           'penalty','coupler','symbolic','monitors','license','citation', \
           'constraints','model_plotter','log_reader','collapse_plotter', \
           'log_converter']

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
from mystic.scripts import model_plotter, log_reader, collapse_plotter, log_converter
import mystic.support as support

# backward compatibility
from mystic.tools import *


def license():
    """print license"""
    print(__license__)
    return

def citation():
    """print citation"""
    print(__doc__[-516:-118])
    return

# end of file
