#!/usr/bin/env python
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
# functions and tools
from poly import polyeval, poly1d
from grid import gridpts, samplepts
from approx import almostEqual


# backward compatibility
from approx import approx_equal

# end of file
