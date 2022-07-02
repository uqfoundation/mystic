#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
convert bound instances into functions
"""
# from dejong import sphere, rosen, step, quartic, shekel
def sphere(x):
    from mystic.models.dejong import sphere; return sphere(x)
from .dejong import sphere as model
sphere.__doc__ = model.__doc__

def rosen(x):
    from mystic.models.dejong import rosen; return rosen(x)
from .dejong import rosen as model
rosen.__doc__ = model.__doc__

def rosen0der(x):
    from mystic.models.dejong import rosen0der; return rosen0der(x)
from .dejong import rosen0der as model
rosen0der.__doc__ = model.__doc__

def rosen1der(x):
    from mystic.models.dejong import rosen1der; return rosen1der(x)
from .dejong import rosen1der as model
rosen1der.__doc__ = model.__doc__

def step(x):
    from mystic.models.dejong import step; return step(x)
from .dejong import step as model
step.__doc__ = model.__doc__

def quartic(x):
    from mystic.models.dejong import quartic; return quartic(x)
from .dejong import quartic as model
quartic.__doc__ = model.__doc__

def shekel(x):
    from mystic.models.dejong import shekel; return shekel(x)
from .dejong import shekel as model
shekel.__doc__ = model.__doc__

# from storn import corana, griewangk, zimmermann
def corana(x):
    from mystic.models.storn import corana; return corana(x)
from .storn import corana as model
corana.__doc__ = model.__doc__

def griewangk(x):
    from mystic.models.storn import griewangk; return griewangk(x)
from .storn import griewangk as model
griewangk.__doc__ = model.__doc__

def zimmermann(x):
    from mystic.models.storn import zimmermann; return zimmermann(x)
from .storn import zimmermann as model
zimmermann.__doc__ = model.__doc__

# from wolfram import fosc3d, nmin51
def fosc3d(x):
    from mystic.models.wolfram import fosc3d; return fosc3d(x)
from .wolfram import fosc3d as model
fosc3d.__doc__ = model.__doc__

def nmin51(x):
    from mystic.models.wolfram import nmin51; return nmin51(x)
from .wolfram import nmin51 as model
nmin51.__doc__ = model.__doc__

# from nag import peaks
def peaks(x):
    from mystic.models.nag import peaks; return peaks(x)
from .nag import peaks as model
peaks.__doc__ = model.__doc__

# from venkataraman import venkat91
def venkat91(x):
    from mystic.models.venkataraman import venkat91; return venkat91(x)
from .venkataraman import venkat91 as model
venkat91.__doc__ = model.__doc__

# from wavy import wavy1, wavy2
def wavy1(x):
    from mystic.models.wavy import wavy1; return wavy1(x)
from .wavy import wavy1 as model
wavy1.__doc__ = model.__doc__

def wavy2(x):
    from mystic.models.wavy import wavy2; return wavy2(x)
from .wavy import wavy2 as model
wavy2.__doc__ = model.__doc__

# from pohlheim import schwefel, ellipsoid, rastrigin, powers, ackley
# from pohlheim import michal, branins, easom, goldstein
def schwefel(x):
    from mystic.models.pohlheim import schwefel; return schwefel(x)
from .pohlheim import schwefel as model
schwefel.__doc__ = model.__doc__

def ellipsoid(x):
    from mystic.models.pohlheim import ellipsoid; return ellipsoid(x)
from .pohlheim import ellipsoid as model
ellipsoid.__doc__ = model.__doc__

def rastrigin(x):
    from mystic.models.pohlheim import rastrigin; return rastrigin(x)
from .pohlheim import rastrigin as model
rastrigin.__doc__ = model.__doc__

def powers(x):
    from mystic.models.pohlheim import powers; return powers(x)
from .pohlheim import powers as model
powers.__doc__ = model.__doc__

def ackley(x):
    from mystic.models.pohlheim import ackley; return ackley(x)
from .pohlheim import ackley as model
ackley.__doc__ = model.__doc__

def michal(x):
    from mystic.models.pohlheim import michal; return michal(x)
from .pohlheim import michal as model
michal.__doc__ = model.__doc__

def branins(x):
    from mystic.models.pohlheim import branins; return branins(x)
from .pohlheim import branins as model
branins.__doc__ = model.__doc__

def easom(x):
    from mystic.models.pohlheim import easom; return easom(x)
from .pohlheim import easom as model
easom.__doc__ = model.__doc__

def goldstein(x):
    from mystic.models.pohlheim import goldstein; return goldstein(x)
from .pohlheim import goldstein as model
goldstein.__doc__ = model.__doc__

# from schittkowski import paviani
def paviani(x):
    from mystic.models.schittkowski import paviani; return paviani(x)
from .schittkowski import paviani as model
paviani.__doc__ = model.__doc__

# clean up
del model


# EOF
