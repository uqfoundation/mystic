#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
math: mathematical functions and tools for use in mystic


Functions
=========

Mystic provides a set of mathematical functions that support various
advanced optimization features such as uncertainty analysis and
parameter sensitivity.


Tools
=====

Mystic also provides a set of mathematical tools that support advanced
features such as parameter space partitioning and monte carlo estimation.
These mathematical tools are provided::

    polyeval     -- fast evaluation of an n-dimensional polynomial
    poly1d       -- generate a 1d polynomial instance
    gridpts      -- generate a set of regularly spaced points
    fillpts      -- generate a set of space-filling points
    samplepts    -- generate a set of randomly sampled points 
    tolerance    -- absolute difference plus relative difference
    almostEqual  -- test if equal within some absolute or relative tolerance
    Distribution -- generate a sampling distribution instance
"""
# functions and tools
from .poly import polyeval, poly1d
from .grid import gridpts, samplepts, fillpts
from .approx import almostEqual, tolerance


# backward compatibility
from .approx import approx_equal
from . import discrete as dirac_measure
from . import distance as paramtrans

__all__ = ['Distribution','polyeval','poly1d','gridpts','samplepts', \
           'fillpts','almostEqual','tolerance']

# distribution object
class Distribution(object):
    """
Sampling distribution for mystic optimizers
    """
    def __init__(self, generator=None, *args, **kwds):
        """
generate a sampling distribution with interface dist(size=None)

input::
    - generator: a 'distribution' method from scipy.stats or numpy.random
    - rng: a mystic.random_state object [default: random_state('numpy.random')]
    - args: positional arguments for the distribtution object
    - kwds: keyword arguments for the distribution object

note::
    this method only accepts numpy.random methods with the keyword 'size',
    and only accepts random_state objects built with module='numpy.random'
        """ #XXX: generate Distribution from list of Distributions?
        from mystic.tools import random_state
        rng = kwds.pop('rng', random_state(module='numpy.random'))
        if generator is None: generator = rng.random
        if getattr(generator, 'rvs', False): 
            d = generator(*args, **kwds)
            self.rvs = lambda size=None: d.rvs(size=size, random_state=rng)
        else:
            d = getattr(rng, generator.__name__)
            self.rvs = lambda size=None: d(size=size, *args, **kwds)
        return
    def __call__(self, size=None):
        """generate a sample of given size (tuple) from the distribution"""
        return self.rvs(size)

# end of file
