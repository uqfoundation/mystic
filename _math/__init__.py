#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
math: mathematical functions and tools for use in mystic


Functions
=========

Mystic provides a set of mathematical functions that support various
advanced optimization features such as uncertainty analysis and
parameter sensitivity.  (These functions are to be provided in an
upcoming release.)


Tools
=====

Mystic also provides a set of mathematical tools that support advanced
features such as parameter space partitioning and monte carlo estimation.
These mathematical tools are provided::
    polyeval     -- fast evaluation of an n-dimensional polynomial
    poly1d       -- generate a 1d polynomial instance
    gridpts      -- generate a set of regularly spaced points
    samplepts    -- generate a set of randomly sampled points 
    almostEqual  -- test if equal within some absolute or relative tolerance


"""
from __future__ import absolute_import
# functions and tools
from .poly import polyeval, poly1d
from .grid import gridpts, samplepts
from .approx import almostEqual


# backward compatibility
from .approx import approx_equal
from . import discrete as dirac_measure
from . import distance as paramtrans

# end of file
