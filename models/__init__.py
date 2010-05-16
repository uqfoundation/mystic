#!/usr/bin/env python
"""
models: sample models and functions prepared for use in mystic


Functions
=========

Mystic provides a set of standard fitting functions that derive from
the function API found in `mystic.models.abstract_model`. These standard
functions are provided::
    rosen      -- Rosenbrock's function
    step       -- De Jong's step function
    quartic    -- De Jong's quartic function
    shekel     -- Shekel's function
    corana     -- Corana's function
    fosc3d     -- the fOsc3D Mathematica function
    griewangk  -- Griewangk's function
    zimmermann -- Zimmermann's function
    wavy1      -- a simple sine-based multi-minima function
    wavy2      -- another simple sine-based multi-minima function


Models
======

Mystic also provides a set of example models that derive from the model API
found in `mystic.models.abstract_model`. These standard models are provided::
    poly       -- 1d model representation for polynomials
    circle     -- 2d array representation of a circle
    lorentzian -- Lorentzian peak model
    br8        -- Bevington & Robinson's model of dual exponential decay
    mogi       -- Mogi's model of surface displacements from a point spherical
                  source in an elastic half space

Additionally, circle has been extended to provide three additional models,
each with different packing densities::
    - dense_circle, sparse_circle, and minimal_circle

Poly also provides two additional models, each for 8th and 16th order
Chebyshev polynomials::
    - chebyshev8, chebyshev16


"""
# base classes
from abstract_model import AbstractModel, AbstractFunction

# models
from poly import poly, chebyshev8, chebyshev16
from mogi import mogi
from br8 import decay
from lorentzian import lorentzian
from circle import circle, dense_circle, sparse_circle, minimal_circle

# functions
from dejong import rosen, step, quartic, shekel
from corana import corana
from fosc3d import fosc3d
from griewangk import griewangk
from zimmermann import zimmermann
from wavy import wavy1, wavy2

#shortcuts
#from poly import chebyshev8cost, chebyshev16cost
#from br8 import data as br8data
#from br8 import cost as br8cost
#from corana import corana1d, corana2d, corana3d
#from lorentzian import gendata, histogram

# end of file
