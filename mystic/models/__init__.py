#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
models: sample models and functions prepared for use in mystic


Functions
=========

Mystic provides a set of standard fitting functions that derive from
the function API found in ``mystic.models.abstract_model``. These standard
functions are provided::

    sphere     -- De Jong's spherical function
    rosen      -- Sum of Rosenbrock's function
    step       -- De Jong's step function
    quartic    -- De Jong's quartic function
    shekel     -- Shekel's function
    corana     -- Corana's function
    fosc3d     -- Trott's fOsc3D function
    nmin51     -- Champion's NMinimize test function
    griewangk  -- Griewangk's function
    zimmermann -- Zimmermann's function
    peaks      -- NAG's peaks function
    venkat91   -- Venkataraman's sinc function
    schwefel   -- Schwefel's function
    ellipsoid  -- Pohlheim's rotated hyper-ellipsoid function
    rastrigin  -- Rastrigin's function
    powers     -- Pohlheim's sum of different powers function
    ackley     -- Ackley's path function
    michal     -- Michalewicz's function
    branins    -- Branins's rcos function
    easom      -- Easom's function
    goldstein  -- Goldstein-Price's function
    paviani    -- Paviani's function
    wavy1      -- a simple sine-based multi-minima function
    wavy2      -- another simple sine-based multi-minima function


Models
======

Mystic also provides a set of example models that derive from the model API
found in ``mystic.models.abstract_model``. These standard models are provided::

    poly       -- 1d model representation for polynomials
    circle     -- 2d array representation of a circle
    lorentzian -- Lorentzian peak model
    br8        -- Bevington & Robinson's model of dual exponential decay
    mogi       -- Mogi's model of surface displacements from a point spherical
                  source in an elastic half space

Additionally, ``circle`` has been extended to provide three additional models,
each with different packing densities::

    dense_circle, sparse_circle, and minimal_circle

Further, ``poly`` provides additional models for 2nd, 4th, 6th, 8th, and 16th
order Chebyshev polynomials::

    chebyshev2, chebyshev4, chebyshev6, chebyshev8, chebyshev16

Also, ``rosen`` has been modified to provide models for the 0th and 1st
derivative of the Rosenbrock function::

    rosen0der, and rosen1der
"""
# base classes
from .abstract_model import AbstractModel, AbstractFunction

# models
from .poly import poly, chebyshev2,chebyshev4,chebyshev6,chebyshev8,chebyshev16
from .mogi import mogi
from .br8 import decay
from .lorentzian import lorentzian
from .circle import circle, dense_circle, sparse_circle, minimal_circle

# functions
from .functions import *
# from dejong import sphere, rosen, step, quartic, shekel
# from storn import corana, griewangk, zimmermann
# from wolfram import fosc3d, nmin51
# from nag import peaks
# from venkataraman import venkat91
# from wavy import wavy1, wavy2
# from pohlheim import schwefel, ellipsoid, rastrigin, powers, ackley
# from pohlheim import michal, branins, easom, goldstein
# from schittkowski import paviani

#shortcuts
#from poly import chebyshev8cost, chebyshev16cost
#from br8 import data as br8data
#from br8 import cost as br8cost
#from lorentzian import gendata, histogram

# end of file
