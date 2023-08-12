#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
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

note::
    generator may be a method object or a string of 'module.object';
    similarly, rng may be a random_state object or a string of 'module'

note::
    Distributions d1,d2 may be combined by adding data (i.e. d1(n) + d2(n)),
    or by adding probabilitiies as Distribution(d1,d2); the former uses
    the addition operator and produces a new unnormalized Distribution,
    while the latter produces a new Distribution which randomly chooses from
    the Distributions provided

note::
    a normalization factor can be incorporated through the multiplication
    or division operator, and is stored in the Distribution as 'norm'
        """ #XXX: generate Distribution from list of Distributions?
        self.norm = kwds.pop('norm', 1) + 0
        if isinstance(generator, Distribution):
            if kwds:
                msg = 'keyword arguments are invalid with {0} instance'.format(self.__class__.__name__)
                raise TypeError(msg)
            if not args:
                self._type = generator._type
                self.rvs = generator.rvs
                self.repr = generator.repr
                self.norm *= generator.norm
                return
            # args can only support additional distribution instances
            for arg in args:
                if not isinstance(arg, Distribution): # raise TypeError
                    generator += arg
            # use choice from multiple distributions
            import numpy as np
            generator = (generator,) + args
            rep = lambda di: "{0}".format(di).split("(",1)[-1][:-1] if di._type == 'join' else "{0}".format(di)
            sig = ', '.join(rep(i) for i in generator)
            self.repr = lambda cls,fac: ("{0}({1}".format(cls, sig) + (')' if fac == 1 else ', norm={0})'.format(fac)))
            self.rvs = lambda size=None: np.choose(np.random.choice(range(len(generator)), size=size), tuple(d(size) for d in generator))
            self._type = 'join'
            return
        from mystic.tools import random_state
        rng = kwds.pop('rng', random_state(module='numpy.random'))
        if isinstance(rng, str): rng = random_state(module=rng)
        mod = 'numpy.random'
        if generator is None:
            generator = rng.random
            mod = rng.__name__
        elif isinstance(generator, str):
            from importlib import import_module
            if '.' in generator:
                mod,generator = generator.rsplit('.', 1) 
                mod = import_module(mod)
            else:
                mod = rng
            generator = getattr(mod, generator)
            mod = mod.__name__
        if getattr(generator, 'rvs', False): 
            d = generator(*args, **kwds)
            self.rvs = lambda size=None: d.rvs(size=size, random_state=rng)
            name = getattr(generator, 'name', None) #XXX: also try __name__?
            mod = 'scipy.stats' #XXX: assumed due to 'd.rvs'
        else:
            d = getattr(rng, generator.__name__)
            self.rvs = lambda size=None: d(size=size, *args, **kwds)
            name = generator.__name__
            mod = rng.__name__
        name = "'{0}.{1}'".format(mod, name) if name else ""
        sig = ', '.join(str(i) for i in args) 
        kwd = ', '.join("{0}={1}".format(i,j) for i,j in kwds.items())
        #nrm = '' if self.norm == 1 else 'norm={0}'.format(self.norm)
        #kwd = '{0}, {1}'.format(kwd, nrm) if (kwd and nrm) else (kwd or nrm)
        sig = '{0}, {1}'.format(sig, kwd) if (sig and kwd) else (sig or kwd)
        if name and sig: name += ", "
        #sig = ", rng='{0}')".format(rng.__name__)
        self.repr = lambda cls,fac: ("{0}({1}".format(cls, name) + sig + ('' if fac == 1 else ((', ' if (name or sig) else '') + 'norm={0}'.format(fac))) + ')')
        self._type = 'base'
        return
    def __call__(self, size=None):
        """generate a sample of given size (tuple) from the distribution"""
        return self.norm * self.rvs(size)
    def __repr__(self):
        return self.repr(self.__class__.__name__, self.norm)
    def __add__(self, dist):
        if not isinstance(dist, Distribution):
            msg = "unsupported operand type(s) for +: '{0}' and '{1}'".format(self.__class__.__name__, type(dist))
            raise TypeError(msg)
        # add data from multiple distributions
        new = Distribution()
        first = "{0}".format(self)
        second = "{0}".format(dist)
        if self._type == 'add': first = first.split("(",1)[-1][:-1]
        if dist._type == 'add': second = second.split("(",1)[-1][:-1]
        new.repr = lambda cls,fac: ("{0}({1} + {2}".format(cls, first, second) + (')' if fac == 1 else ', norm={0})'.format(fac)))
        new.rvs = lambda size=None: (self(size) + dist(size))
        new._type = 'add'
        new.norm = 1
        return new
    def __mul__(self, norm):
        new = Distribution()
        new.repr = self.repr
        new.rvs = self.rvs
        new._type = 'base'
        new.norm = self.norm * norm
        return new
    __rmul__ = __mul__
    def __truediv__(self, denom):
        new = Distribution()
        new.repr = self.repr
        new.rvs = self.rvs
        new._type = 'base'
        new.norm = self.norm / denom
        return new
    def __floordiv__(self, denom):
        new = Distribution()
        new.repr = self.repr
        new.rvs = self.rvs
        new._type = 'base'
        new.norm = self.norm // denom
        return new
    """
    def __mul__(self, dist):
        if not isinstance(dist, Distribution):
            msg = "unsupported operand type(s) for *: '{0}' and '{1}'".format(self.__class__.__name__, type(dist))
            raise TypeError(msg)
        # use conflation of multiple distributions
        new = Distribution()
        norm = lambda x: x/sum(x) #FIXME: what is the formula...?
        #func = lambda x,y: (x*y)/(x+y)
        #new.rvs = lambda size=None: func(self(size),dist(size))
        new.rvs = lambda size=None: norm(self(size) * dist(size))
        new.repr = lambda cls: "{0}({1} * {2})".format(cls, self, dist)
        return new
    """


# end of file
