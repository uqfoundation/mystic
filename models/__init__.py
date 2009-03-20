#!/usr/bin/env python

"""
models and functions used in testing & examples
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
#from poly import polyeval, poly1d
#from poly import chebyshev8cost, chebyshev16cost
#from br8 import data as br8data
#from br8 import cost as br8cost
#from corana import corana1d, corana2d, corana3d
#from lorentzian import gendata, histogram

# end of file
