#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Classes for discrete measure data objects.
Includes point_mass, measure, product_measure, and scenario classes.
"""
# Adapted from seesaw2d.py in branches/UQ/math/examples2/ 
# For usage example, see seesaw2d_inf_example.py .

from mystic.math.measures import impose_expectation, impose_expected_variance
from mystic.math.measures import impose_expected_mean_and_variance
from mystic.math.measures import impose_mean, impose_spread, impose_variance
from mystic.math.measures import impose_weight_norm

__all__ = ['point_mass','measure','product_measure','scenario',\
           '_mimic','_uniform_weights','_list_of_measures','compose',\
           'decompose','unflatten','flatten','bounded_mean',\
           'norm_wts_constraintsFactory','mean_y_norm_wts_constraintsFactory',\
           'impose_feasible','impose_valid']

class point_mass(object):
  """a point mass object with weight and position

Args:
    position (tuple(float)): position of the point mass
    weight (float, default=1.0): weight of the point mass

Attributes:
    position (tuple(float)): position of the point mass
    weight (float): weight of the point mass
"""

  def __init__(self, position, weight=1.0):
    self.weight = weight
    self.position = position
    return

  def __repr__(self):
    return "(%s @%s)" % (self.weight, self.position)

  def __rms(self): # square root of sum of squared positions
    from math import sqrt
    return sqrt(sum([i**2 for i in self.position]))

  # interface
  rms = property(__rms, doc='readonly: square root of the sum of squared position')

  pass

class measure(list):  #FIXME: meant to only accept point_masses...
  """a 1-d collection of point masses forming a 'discrete measure'

Args:
    iterable (list): a list of ``mystic.math.discrete.point_mass`` objects

Notes:
    - assumes only contains ``mystic.math.discrete.point_mass`` objects
    - assumes ``measure.n = len(measure.positions) == len(measure.weights)``
    - relies on constraints to impose notions such as ``sum(weights) == 1.0``
"""

  def support_index(self, tol=0):
    """get the indices where there is support (i.e. non-zero weight)

Args:
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the list of indices where there is support
"""
    from .measures import support_index
    return support_index(self.weights, tol)

  def support(self, tol=0):
    """get the positions with non-zero weight (i.e. support)

Args:
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the list of positions with support
"""
    from .measures import support
    return support(self.positions, self.weights, tol)

  def __weights(self):
    return [i.weight for i in self]

  def __positions(self):
    return [i.position for i in self]

  def __n(self):
    return len(self)

  def __mass(self):
    return sum(self.weights)
    #from mystic.math.measures import norm
    #return norm(self.weights)  # normalized by self.npts

  def __mean(self):
    from mystic.math.measures import mean
    return mean(self.positions, self.weights) 

  def __range(self):
    from mystic.math.measures import spread
    return spread(self.positions)

  def __variance(self):
    from mystic.math.measures import variance
    return variance(self.positions, self.weights)

  def __set_weights(self, weights):
    for i in range(len(weights)):
      self[i].weight = weights[i]
    return

  def __set_positions(self, positions):
    for i in range(len(positions)):
      self[i].position = positions[i]
    return

  def normalize(self):
    """normalize the weights to 1.0

Args:
    None

Returns:
    None
"""
    self.positions, self.weights = impose_weight_norm(self.positions, self.weights)
    return

  def __set_mean(self, m):
    self.positions = impose_mean(m, self.positions, self.weights)
    return

  def __set_range(self, r):
    self.positions = impose_spread(r, self.positions, self.weights)
    return

  def __set_variance(self, v):
    self.positions = impose_variance(v, self.positions, self.weights)
    return

  def maximum(self, f):
    """calculate the maximum for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the maximum value of ``f`` over all measure positions
"""
    from .measures import maximum
    positions = [(i,) for i in self.positions]
    return maximum(f, positions)

  def ess_maximum(self, f, tol=0.):
    """calculate the maximum for the support of a given function

Args:
    f (func): a function that takes a list and returns a number
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the maximum value of ``f`` over all measure positions with support
"""
    from .measures import ess_maximum
    positions = [(i,) for i in self.positions]
    return ess_maximum(f, positions, self.weights, tol)

  def minimum(self, f):
    """calculate the minimum for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the minimum value of ``f`` over all measure positions
"""
    from .measures import minimum
    positions = [(i,) for i in self.positions]
    return minimum(f, positions)

  def ess_minimum(self, f, tol=0.):
    """calculate the minimum for the support of a given function

Args:
    f (func): a function that takes a list and returns a number
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the minimum value of ``f`` over all measure positions with support
"""
    from .measures import ess_minimum
    positions = [(i,) for i in self.positions]
    return ess_minimum(f, positions, self.weights, tol)

  def ptp(self, f):
    """calculate the spread for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the spread of the values of ``f`` over all measure positions
"""
    from .measures import ptp
    positions = [(i,) for i in self.positions]
    return ptp(f, positions)

  def ess_ptp(self, f, tol=0.):
    """calculate the spread for the support of a given function

Args:
    f (func): a function that takes a list and returns a number
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the spread of the values of ``f`` over all measure positions with support
"""
    from .measures import ess_ptp
    positions = [(i,) for i in self.positions]
    return ess_ptp(f, positions, self.weights, tol)

  def expect(self, f):
    """calculate the expectation for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the expectation of ``f`` over all measure positions
""" #XXX: maybe more natural if f takes a positional value x, not a list x ?
    from mystic.math.measures import expectation
    positions = [(i,) for i in self.positions]
    return expectation(f, positions, self.weights)

  def expect_var(self, f):
    """calculate the expected variance for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the expected variance of ``f`` over all measure positions
""" #XXX: maybe more natural if f takes a positional value x, not a list x ?
    from mystic.math.measures import expected_variance
    positions = [(i,) for i in self.positions]
    return expected_variance(f, positions, self.weights)

  #NOTE: backward incompatible 08/26/18: (expected=(m,D),...) --> (m,...,tol=D)
  def set_expect(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose an expectation on the measure by adjusting the positions

Args:
    expected (float): target expected mean
    f (func): a function that takes a list and returns a number
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``c' = constraints(c)``,
        where ``c`` is a product measure, and ``c'`` is a product measure
        where the encoded constaints are satisfied.
    tol (float, default=None): maximum allowable deviation from ``expected``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    None

Notes:
    Expectation ``E`` is calculated by minimizing ``mean(f(x)) - expected``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target mean ``expected``. If ``tol`` is not
    provided, then a relative deviation of 1% of ``expected`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from.

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter.
""" #XXX: maybe more natural if f takes a positional value x, not a list x ?
    #XXX: maybe also natural c' = constraints(c) where c is a measure ?
    m = expected
    if constraints:  # then adjust interface for 'impose_expectation'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    positions = impose_expectation(m, f, [self.npts], bounds, self.weights, \
                                   constraints=cnstr, **kwds)
    from numpy import array
    self.positions = list(array(positions)[:,0])
   #from numpy import squeeze
   #self.positions = list(squeeze(positions))
    return

  def set_expect_var(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose an expected variance on the measure by adjusting the positions

Args:
    expected (float): target expected variance
    f (func): a function that takes a list and returns a number
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``c' = constraints(c)``,
        where ``c`` is a product measure, and ``c'`` is a product measure
        where the encoded constaints are satisfied.
    tol (float, default=None): maximum allowable deviation from ``expected``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    None

Notes:
    Expected var ``E`` is calculated by minimizing ``var(f(x)) - expected``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target variance ``expected``. If ``tol`` is not
    provided, then a relative deviation of 1% of ``expected`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from.

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter.
""" #XXX: maybe more natural if f takes a positional value x, not a list x ?
    #XXX: maybe also natural c' = constraints(c) where c is a measure ?
    m = expected
    if constraints:  # then adjust interface for 'impose_expected_variance'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    positions = impose_expected_variance(m, f, [self.npts], bounds, \
                                self.weights, constraints=cnstr, **kwds)
    from numpy import array
    self.positions = list(array(positions)[:,0])
   #from numpy import squeeze
   #self.positions = list(squeeze(positions))
    return

  def set_expect_mean_and_var(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose expected mean and var on the measure by adjusting the positions

Args:
    expected (tuple(float)): ``(expected mean, expected var)``
    f (func): a function that takes a list and returns a number
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``c' = constraints(c)``,
        where ``c`` is a product measure, and ``c'`` is a product measure
        where the encoded constaints are satisfied.
    tol (float, default=None): maximum allowable deviation from ``expected``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    None

Notes:
    Expected mean ``E`` and expected variance ``R`` are calculated by
    minimizing the sum of the absolute values of ``mean(f(x)) - m`` and
    ``variance(f(x)) - v`` over the given *bounds*, and will terminate when
    ``E`` and ``R`` are found within tolerance ``tol`` of the target mean ``m``
    and variance ``v``, respectively. If ``tol`` is not provided, then a
    relative deviation of 1% of ``max(m,v)`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter
""" #XXX: maybe more natural if f takes a positional value x, not a list x ?
    #XXX: maybe also natural c' = constraints(c) where c is a measure ?
    m,v = expected
    if constraints:  # then adjust interface for 'impose_expected_mean_and_var'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    positions = impose_expected_mean_and_variance((m,v), f, [self.npts], \
                       bounds, self.weights, constraints=cnstr, **kwds)
    from numpy import array
    self.positions = list(array(positions)[:,0])
   #from numpy import squeeze
   #self.positions = list(squeeze(positions))
    return

  # interface
  weights = property(__weights, __set_weights, doc='a list of weights for all point masses in the measure')
  positions = property(__positions, __set_positions, doc='a list of positions for all point masses in the measure')
  ###XXX: why not use 'points' also/instead?
  npts = property(__n, doc='readonly: the number of point masses in the measure')
  mass = property(__mass, doc='readonly: the sum of the weights')
  range = property(__range, __set_range, doc='``|max - min|`` for the positions')
  center_mass = property(__mean, __set_mean, doc='sum of ``weights * positions``')
  var = property(__variance, __set_variance, doc='``mean(|positions - mean(positions)|**2)``')

  # backward compatibility
  coords = positions
  get_expect = expect
  mean = center_mass
  pass

class product_measure(list):  #FIXME: meant to only accept sets...
  """a N-d measure-theoretic product of discrete measures

Args:
    iterable (list): a list of ``mystic.math.discrete.measure`` objects

Notes:
    - all measures are treated as if they are orthogonal
    - assumes only contains ``mystic.math.discrete.measure`` objects
    - assumes ``len(product_measure.positions) == len(product_measure.weights)``
    - relies on constraints to impose notions such as ``sum(weights) == 1.0``
    - relies on constraints to impose expectation (within acceptable deviation)
    - positions are ``(xi,yi,zi)`` with weights ``(wxi,wyi,wzi)``, where weight
      ``wxi`` at ``xi`` should be the same for each ``(yj,zk)``.  Similarly
      for each ``wyi`` and ``wzi``.
"""
  def __val(self):
    raise NotImplementedError("'value' is undefined in a measure")

  def __pts(self):
    return [i.npts for i in self]

  def __wts(self):
    return [i.weights for i in self]

  def __pos(self):
    return [i.positions for i in self]

  def __mean(self):
    return [i.center_mass for i in self]

  def __set_mean(self, center_masses):
    [i._measure__set_mean(center_masses[m]) for (m,i) in enumerate(self)]
   #for i in range(len(center_masses)):
   #  self[i].center_mass = center_masses[i]
    return

  def __n(self):
    from numpy import product
    return product(self.pts)

  def support_index(self, tol=0):
    """get the indices where there is support (i.e. non-zero weight)

Args:
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the list of indices where there is support
"""
    from .measures import support_index
    return support_index(self.weights, tol)

  def support(self, tol=0): #XXX: better if generated positions only when needed
    """get the positions with non-zero weight (i.e. support)

Args:
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the list of positions with support
"""
    from .measures import support
    return support(self.positions, self.weights, tol)

  def __weights(self):
    from mystic.math.measures import _pack
    from numpy import product
    weights = _pack(self.wts)
    _weights = []
    for wts in weights:
      _weights.append(product(wts))
    return _weights

  def __positions(self):
    from mystic.math.measures import _pack
    return _pack(self.pos)

  def __set_positions(self, positions):
    from mystic.math.measures import _unpack
    positions = _unpack(positions, self.pts)
    for i in range(len(positions)):
      self[i].positions = positions[i]
    return

 #def __get_center(self):
 #  return self.__center

 #def __get_delta(self):
 #  return self.__delta

  def __mass(self):
    return [self[i].mass for i in range(len(self))]

  def maximum(self, f): #XXX: return max of all or return all max?
    """calculate the maximum for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the maximum value of ``f`` over all measure positions
"""
    return max([i.maximum(f) for i in self])

  def minimum(self, f): #XXX: return min of all or return all min?
    """calculate the minimum for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the minimum value of ``f`` over all measure positions
"""
    return min([i.minimum(f) for i in self])

  def ptp(self, f): #XXX: return max of all or return all ptp?
    """calculate the spread for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the spread for values of ``f`` over all measure positions
"""
    return max([i.ptp(f) for i in self])

  def ess_maximum(self, f, tol=0.): #XXX: return max of all or return all max?
    """calculate the maximum for the support of a given function

Args:
    f (func): a function that takes a list and returns a number
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the maximum value of ``f`` over all measure positions with support
"""
    return max([i.ess_maximum(f, tol) for i in self])

  def ess_minimum(self, f, tol=0.): #XXX: return min of all or return all min?
    """calculate the minimum for the support of a given function

Args:
    f (func): a function that takes a list and returns a number
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the minimum value of ``f`` over all measure positions with support
"""
    return min([i.ess_minimum(f, tol) for i in self])

  def ess_ptp(self, f, tol=0.): #XXX: return max of all or return all ptp?
    """calculate the spread for the support of a given function

Args:
    f (func): a function that takes a list and returns a number
    tol (float, default=0.0): tolerance, where any ``weight <= tol`` is zero

Returns:
    the spread of values of ``f`` over all measure positions with support
"""
    return max([i.ess_ptp(f, tol) for i in self])

  def expect(self, f):
    """calculate the expectation for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the expectation of ``f`` over all measure positions
"""
    from mystic.math.measures import expectation
    return expectation(f, self.positions, self.weights)

  def expect_var(self, f):
    """calculate the expected variance for a given function

Args:
    f (func): a function that takes a list and returns a number

Returns:
    the expected variance of ``f`` over all measure positions
"""
    from mystic.math.measures import expected_variance
    return expected_variance(f, self.positions, self.weights)

  #NOTE: backward incompatible 08/26/18: (expected=(m,D),...) --> (m,...,tol=D)
  def set_expect(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose an expectation on the measure by adjusting the positions

Args:
    expected (float): target expected mean
    f (func): a function that takes a list and returns a number
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``c' = constraints(c)``,
        where ``c`` is a product measure, and ``c'`` is a product measure
        where the encoded constaints are satisfied.
    tol (float, default=None): maximum allowable deviation from ``expected``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    None

Notes:
    Expectation ``E`` is calculated by minimizing ``mean(f(x)) - expected``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target mean ``expected``. If ``tol`` is not
    provided, then a relative deviation of 1% of ``expected`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from.

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter.
"""
    #self.__center = m
    #self.__delta = D
    m = expected
    if constraints:  # then adjust interface for 'impose_expectation'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    self.positions = impose_expectation(m, f, self.pts, bounds, self.weights, \
                                        constraints=cnstr, **kwds)
    return

  def set_expect_var(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose an expected variance on the measure by adjusting the positions

Args:
    expected (float): target expected variance
    f (func): a function that takes a list and returns a number
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``c' = constraints(c)``,
        where ``c`` is a product measure, and ``c'`` is a product measure
        where the encoded constaints are satisfied.
    tol (float, default=None): maximum allowable deviation from ``expected``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    None

Notes:
    Expected var ``E`` is calculated by minimizing ``var(f(x)) - expected``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target variance ``expected``. If ``tol`` is not
    provided, then a relative deviation of 1% of ``expected`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from.

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter.
"""
    #self.__center = m
    #self.__delta = D
    m = expected
    if constraints:  # then adjust interface for 'impose_expected_variance'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    self.positions = impose_expected_variance(m, f, self.pts, bounds, \
                             self.weights, constraints=cnstr, **kwds)
    return


  def set_expect_mean_and_var(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose expected mean and var on the measure by adjusting the positions

Args:
    expected (tuple(float)): ``(expected mean, expected var)``
    f (func): a function that takes a list and returns a number
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``c' = constraints(c)``,
        where ``c`` is a product measure, and ``c'`` is a product measure
        where the encoded constaints are satisfied.
    tol (float, default=None): maximum allowable deviation from ``expected``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    None

Notes:
    Expected mean ``E`` and expected variance ``R`` are calculated by
    minimizing the sum of the absolute values of ``mean(f(x)) - m`` and
    ``variance(f(x)) - v`` over the given *bounds*, and will terminate when
    ``E`` and ``R`` are found within tolerance ``tol`` of the target mean ``m``
    and variance ``v``, respectively. If ``tol`` is not provided, then a
    relative deviation of 1% of ``max(m,v)`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter
"""
    #self.__center = m
    #self.__delta = D
    m,v = expected
    if constraints:  # then adjust interface for 'impose_expected_mean_and_var'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    self.positions = impose_expected_mean_and_variance((m,v), f, self.pts, \
                          bounds, self.weights, constraints=cnstr, **kwds)
    return


  def pof(self, f):
    """calculate probability of failure for a given function

Args:
    f (func): a function returning True for 'success' and False for 'failure'

Returns:
    the probabilty of failure, a float in ``[0.0,1.0]``

Notes:
    - the function ``f`` should take a list of ``positions`` (for example,
      ``scenario.positions`` or ``product_measure.positions``) and return a
      single value (e.g. 0.0 or False)
"""
    u = 0.0
    set = zip(self.positions, self.weights)
    for x in set:
      if f(x[0]) <= 0.0: #XXX: should f return a bool or a float?
        u += x[1]
    return u
  # for i in range(self.npts):
  #   #if f(self.positions[i]) > 0.0:  #NOTE: f(x) > 0.0 yields prob of success
  #   if f(self.positions[i]) <= 0.0:  #NOTE: f(x) <= 0.0 yields prob of failure
  #     u += self.weights[i]
  # return u  #XXX: does this need to be normalized?

  def sampled_minimum(self, f, npts=10000):
    """use sampling to calculate ess_minimum for a given function

Args:
    f (func): a function that takes a list and returns a number
    npts (int, default=10000): the number of point masses sampled from the
        underlying discrete measures

Returns:
    the sampled ess_minimum, a float

Notes:
    - the function ``f`` should take a list of ``positions`` (for example,
      ``scenario.positions`` or ``product_measure.positions``) and return a
      single value (e.g. 0.0)
"""
    from mystic.math.samples import _minimum_given_samples
    pts = self.sampled_support(npts)
    return _minimum_given_samples(f, pts)

  def sampled_ptp(self, f, npts=10000):
    """use sampling to calculate ess_|maximum - minimum| for a given function

Args:
    f (func): a function that takes a list and returns a number
    npts (int, default=10000): the number of point masses sampled from the
        underlying discrete measures

Returns:
    the sampled |ess_maximum - ess_minimum|, a float

Notes:
    - the function ``f`` should take a list of ``positions`` (for example,
      ``scenario.positions`` or ``product_measure.positions``) and return a
      single value (e.g. 0.0)
"""
    from mystic.math.samples import _ptp_given_samples
    pts = self.sampled_support(npts)
    return _ptp_given_samples(f, pts)

  def sampled_expect(self, f, npts=10000):
    """use sampling to calculate expected value for a given function

Args:
    f (func): a function that takes a list and returns a number
    npts (int, default=10000): the number of point masses sampled from the
        underlying discrete measures

Returns:
    the expected value, a float

Notes:
    - the function ``f`` should take a list of ``positions`` (for example,
      ``scenario.positions`` or ``product_measure.positions``) and return a
      single value (e.g. 0.0)
"""
    from mystic.math.samples import _expectation_given_samples
    pts = self.sampled_support(npts)
    return _expectation_given_samples(f, pts)

  def sampled_maximum(self, f, npts=10000):
    """use sampling to calculate ess_maximum for a given function

Args:
    f (func): a function that takes a list and returns a number
    npts (int, default=10000): the number of point masses sampled from the
        underlying discrete measures

Returns:
    the ess_maximum, a float

Notes:
    - the function ``f`` should take a list of ``positions`` (for example,
      ``scenario.positions`` or ``product_measure.positions``) and return a
      single value (e.g. 0.0)
"""
    from mystic.math.samples import _maximum_given_samples
    pts = self.sampled_support(npts)
    return _maximum_given_samples(f, pts)

  def sampled_pof(self, f, npts=10000):
    """use sampling to calculate probability of failure for a given function

Args:
    f (func): a function returning True for 'success' and False for 'failure'
    npts (int, default=10000): the number of point masses sampled from the
        underlying discrete measures

Returns:
    the probabilty of failure, a float in ``[0.0,1.0]``

Notes:
    - the function ``f`` should take a list of ``positions`` (for example,
      ``scenario.positions`` or ``product_measure.positions``) and return a
      single value (e.g. 0.0 or False)
"""
    from mystic.math.samples import _pof_given_samples
    pts = self.sampled_support(npts)
    return _pof_given_samples(f, pts)

  def sampled_support(self, npts=10000): ##XXX: was 'def support'
    """randomly select support points from the underlying discrete measures

Args:
    npts (int, default=10000): the number of sampled points

Returns:
    a list of ``len(product measure)`` lists, each of length ``len(npts)``
"""
    from mystic.math.measures import weighted_select as _select
    pts = []
    for i in range(npts):
      # for a single trial, select positions from all sets
      pts.append( [_select(set.positions, set.weights) for set in self] )

    # convert pts to len(prod_meas) lists, each of len(npts)
    from numpy import transpose
    return transpose(pts)  #XXX: assumes 'positions' is a list of floats

  def update(self, params):
    """update the product measure from a list of parameters

Args:
    params (list(float)): parameters corresponding to N 1D discrete measures

Returns:
    self (measure): the product measure itself

Notes:
    The dimensions of the product measure will not change upon update, and
    it is assumed *params* either corresponds to the correct number of
    weights and positions for the existing ``product_measure``, or *params*
    has additional values (typically output values) which will be ignored.
    It is assumed that ``len(params) >= 2 * sum(product_measure.pts)``.

    If ``product_measure.pts = (M, N, ...)``, then it is assumed that
    ``params = [wx1, ..., wxM, x1, ..., xM, wy1, ..., wyN, y1, ..., yN, ...]``.
    Thus, *params* should have ``M`` weights and ``M`` corresponding positions,
    followed by ``N`` weights and ``N`` corresponding positions, with this
    pattern followed for each new dimension of the desired product measure.
"""
    pts = self.pts
    _len = 2 * sum(pts)

    if len(params)  >  _len:  # if Y-values are appended to params
      params, values  =  params[:_len], params[_len:]

    pm = unflatten(params, pts)
    zo = pm.count([])
    self[:] = pm[:len(self) - zo] + self[len(pm) - zo:]
    return self

  def load(self, params, pts):
    """load the product measure from a list of parameters

Args:
    params (list(float)): parameters corresponding to N 1D discrete measures
    pts (tuple(int)): number of point masses in each of the discrete measures

Returns:
    self (measure): the product measure itself

Notes:
    To append ``len(pts)`` new discrete measures to the product measure,
    it is assumed *params* either corresponds to the correct number of
    weights and positions specified by *pts*, or *params* has additional
    values (typically output values) which will be ignored. It is assumed
    that ``len(params) >= 2 * sum(product_measure.pts)``.

    Given the value of ``pts = (M, N, ...)``, it is assumed that
    ``params = [wx1, ..., wxM, x1, ..., xM, wy1, ..., wyN, y1, ..., yN, ...]``.
    Thus, *params* should have ``M`` weights and ``M`` corresponding positions,
    followed by ``N`` weights and ``N`` corresponding positions, with this
    pattern followed for each new dimension of the desired product measure.
"""
    _len = 2 * sum(pts)
    if len(params)  >  _len:  # if Y-values are appended to params
      params, values  =  params[:_len], params[_len:]

    self.extend( unflatten(params, pts) )
    return self

  def flatten(self):
    """convert a product measure to a single list of parameters

Args:
    None

Returns:
    a list of parameters

Notes:
    Given ``product_measure.pts = (M, N, ...)``, then the returned list is
    ``params = [wx1, ..., wxM, x1, ..., xM, wy1, ..., wyN, y1, ..., yN, ...]``.
    Thus, *params* will have ``M`` weights and ``M`` corresponding positions,
    followed by ``N`` weights and ``N`` corresponding positions, with this
    pattern followed for each new dimension of the desired product measure.
"""
    params = flatten(self)
    return params

  #XXX: name stinks... better as "non_redundant"? ...is really a helper
  def differs_by_one(self, ith, all=True, index=True):
    """get the coordinates where the associated binary string differs
by exactly one index

Args:
    ith (int): the target index
    all (bool, default=True): if False, only return results for indices < ``i``
    index (bool, default=True): if True, return the indices of the results
        instead of the results themselves

Returns:
    the coordinates where the associated binary string differs by one, or
    if *index* is True, return the corresponding indices
"""
    from mystic.math.compressed import index2binary, differs_by_one
    b = index2binary(list(range(self.npts)), self.npts)
    return differs_by_one(ith, b, all, index) 

  def select(self, *index, **kwds):
    """generate product measure positions for the selected position indices

Args:
    index (tuple(int)): tuple of position indicies

Returns:
    a list of product measure positions for the selected indices

Examples:
    >>> r
    [[9, 8], [1, 3], [4, 2]]
    >>> r.select(*range(r.npts))
    [(9, 1, 4), (8, 1, 4), (9, 3, 4), (8, 3, 4), (9, 1, 2), (8, 1, 2), (9, 3, 2), (8, 3, 2)]
    >>>
    >>> _pack(r)
    [(9, 1, 4), (8, 1, 4), (9, 3, 4), (8, 3, 4), (9, 1, 2), (8, 1, 2), (9, 3, 2), (8, 3, 2)]

Notes:
    This only works for product measures of dimension ``2^K``
"""
    from mystic.math.compressed import index2binary, binary2coords
    v = index2binary(list(index), self.npts)
    return binary2coords(v, self.pos, **kwds)
    #XXX: '_pack' requires resorting ([::-1]) so that indexing is wrong.
    #     Better if modify mystic's pack to match sorting of binary strings ?

 #__center = None
 #__delta = None

  # interface
  npts = property(__n, doc='readonly: the total number of point masses in the product measure')
  weights = property(__weights, doc='a list of weights for all point masses in the product measure')
  positions = property(__positions, __set_positions, doc='a list of positions for all point masses in the product measure')
  center_mass = property(__mean, __set_mean, doc='sum of ``weights * positions``')
 #center = property(__get_center ) #FIXME: remove c.center and c.delta... or
 #delta = property(__get_delta )   #       replace with c._params (e.g. (m,D))
 #expect = property(__expect, __set_expect )
  mass = property(__mass, doc='readonly: a list of weight norms')
  pts = property(__pts, doc='readonly: the number of point masses for each discrete mesure')
  wts = property(__wts, doc='readonly: a list of weights for each discrete mesure')
  pos = property(__pos, doc='readonly: a list of positions for each discrete mesure')

  # backward compatibility
  coords = positions
  get_expect = expect
  pass


class scenario(product_measure):  #FIXME: meant to only accept sets...
  """a N-d product measure with associated data values

A scenario is a measure-theoretic product of discrete measures that also
includes a list of associated values, with the values corresponding to
measured or synthetic data for each measure position.  Each point mass
in the product measure is paired with a value, and thus, essentially, a
scenario is equivalent to a ``mystic.math.legacydata.dataset`` stored in 
a ``product_measure`` representation.

Args:
    pm (mystic.math.discrete.product_measure, default=None): a product measure
    values (list(float), default=None): values associated with each position

Notes:
    - all measures are treated as if they are orthogonal
    - relies on constraints to impose notions such as ``sum(weights) == 1.0``
    - relies on constraints to impose expectation (within acceptable deviation)
    - positions are ``(xi,yi,zi)`` with weights ``(wxi,wyi,wzi)``, where weight
      ``wxi`` at ``xi`` should be the same for each ``(yj,zk)``.  Similarly
      for each ``wyi`` and ``wzi``.
"""
  def __init__(self, pm=None, values=None):
    super(product_measure,self).__init__()
    if pm: 
      pm = product_measure(pm)
      self.load(pm.flatten(), pm.pts)
    if not values: values = []
    self.__Y = values # storage for values of s.positions
    return

  def __values(self):
    return self.__Y

  def __set_values(self, values):
    self.__Y = values[:]
    return

  def mean_value(self):  # get mean of y's
    """calculate the mean of the associated values for a scenario

Args:
    None

Returns:
    the weighted mean of the scenario values
"""
    from mystic.math.measures import mean
    return mean(self.values, self.weights)

  def set_mean_value(self, m):  # set mean of y's
    """set the mean for the associated values of a scenario

Args:
    m (float): the target weighted mean of the scenario values

Returns:
    None
"""
    from mystic.math.measures import impose_mean
    self.values = impose_mean(m, self.values, self.weights)
    return

  def valid_wrt_model(self, model, blamelist=False, pairs=True, \
                                   all=False, raw=False, **kwds):
    """check for scenario validity with respect to the model

Args:
    model (func): the model function, ``y' = F(x')``.
    blamelist (bool, default=False): if True, indicate the infeasible points.
    pairs (bool, default=True): if True, indicate indices of infeasible points.
    all (bool, default=False): if True, get results for each individual point.
    raw (bool, default=False): if False, get boolean results (i.e. non-float).
    ytol (float, default=0.0): maximum acceptable difference ``|y - F(x')|``.
    xtol (float, default=0.0): maximum acceptable difference ``|x - x'|``.
    cutoff (float, default=ytol): zero out distances less than cutoff.
    hausdorff (bool, default=False): hausdorff ``norm``, where if given,
        then ``ytol = |y - F(x')| + |x - x'|/norm``.

Notes:
    *xtol* defines the n-dimensional base of a pilar of height *ytol*,
    centered at each point. The region inside the pilar defines the space
    where a "valid" model must intersect. If *xtol* is not specified, then
    the base of the pilar will be a dirac at ``x' = x``. This function
    performs an optimization for each ``x`` to find an appropriate ``x'``.

    *ytol* is a single value, while *xtol* is a single value or an iterable.
    *cutoff* takes a float or a boolean, where ``cutoff=True`` will set the
    value of *cutoff* to the default. Typically, the value of *cutoff* is
    *ytol*, 0.0, or None. *hausdorff* can be False (e.g. ``norm = 1.0``),
    True (e.g. ``norm = spread(x)``), or a list of points of ``len(x)``.

    While *cutoff* and *ytol* are very tightly related, they play a distinct
    role; *ytol* is used to set the optimization termination for an acceptable
    ``|y - F(x')|``, while *cutoff* is applied post-optimization.

    If we are using the *hausdorff* norm, then *ytol* will set the optimization
    termination for an acceptable ``|y - F(x')| + |x - x'|/norm``, where the
    ``x`` values are normalized by ``norm = hausdorff``.
"""
    from mystic.math.legacydata import dataset 
    data = dataset() 
    data.load(self.positions, self.values)
   #data.lipschitz = L
    for i in range(len(data)):
      data[i].id = i
    return data.valid(model, blamelist=blamelist, pairs=pairs, \
                                       all=all, raw=raw, **kwds)

  def short_wrt_self(self, L, blamelist=False, pairs=True, \
                              all=False, raw=False, **kwds):
    """check for shortness with respect to the scenario itself

Args:
    L (float): the lipschitz constant.
    blamelist (bool, default=False): if True, indicate the infeasible points.
    pairs (bool, default=True): if True, indicate indices of infeasible points.
    all (bool, default=False): if True, get results for each individual point.
    raw (bool, default=False): if False, get boolean results (i.e. non-float).
    tol (float, default=0.0): maximum acceptable deviation from shortness.
    cutoff (float, default=tol): zero out distances less than cutoff.

Notes:
    Each point x,y can be thought to have an associated double-cone with slope
    equal to the lipschitz constant. Shortness with respect to another point is
    defined by the first point not being inside the cone of the second. We can
    allow for some error in shortness, a short tolerance *tol*, for which the
    point x,y is some acceptable y-distance inside the cone. While very tightly
    related, *cutoff* and *tol* play distinct roles; *tol* is subtracted from
    calculation of the lipschitz_distance, while *cutoff* zeros out the value
    of any element less than the *cutoff*.
"""
    from mystic.math.legacydata import dataset 
    data = dataset() 
    data.load(self.positions, self.values)
    data.lipschitz = L
    for i in range(len(data)):
      data[i].id = i
    return data.short(blamelist=blamelist, pairs=pairs, \
                                           all=all, raw=raw, **kwds)

  def short_wrt_data(self, data, L=None, blamelist=False, pairs=True, \
                                         all=False, raw=False, **kwds):
    """check for shortness with respect to the given data

Args:
    data (list): a list of data points or dataset to compare against.
    L (float, default=None): the lipschitz constant, if different than in data.
    blamelist (bool, default=False): if True, indicate the infeasible points.
    pairs (bool, default=True): if True, indicate indices of infeasible points.
    all (bool, default=False): if True, get results for each individual point.
    raw (bool, default=False): if False, get boolean results (i.e. non-float).
    tol (float, default=0.0): maximum acceptable deviation from shortness.
    cutoff (float, default=tol): zero out distances less than cutoff.

Notes:
    Each point x,y can be thought to have an associated double-cone with slope
    equal to the lipschitz constant. Shortness with respect to another point is
    defined by the first point not being inside the cone of the second. We can
    allow for some error in shortness, a short tolerance *tol*, for which the
    point x,y is some acceptable y-distance inside the cone. While very tightly
    related, *cutoff* and *tol* play distinct roles; *tol* is subtracted from
    calculation of the lipschitz_distance, while *cutoff* zeros out the value
    of any element less than the *cutoff*.
"""
    from mystic.math.legacydata import dataset 
    _self = dataset() 
    _self.load(self.positions, self.values)
    _self.lipschitz = data.lipschitz
    for i in range(len(_self)):
      _self[i].id = i
    return _self.short(data, L=L, blamelist=blamelist, pairs=pairs, \
                                               all=all, raw=raw, **kwds)

  def set_feasible(self, data, cutoff=0.0, bounds=None, constraints=None, \
                                                  with_self=True, **kwds):
    """impose shortness with respect to the given data points

This function attempts to minimize the infeasibility between observed *data*
and the scenario of synthetic data by perforing an optimization on ``w,x,y``
over the given *bounds*.

Args:
    data (mystic.math.discrete.scenario): a dataset of observed points
    cutoff (float, default=0.0): maximum acceptable deviation from shortness
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``x' = constraints(x)``,
        where ``x`` is a scenario that has been converted into a list of
        parameters (e.g. with ``scenario.flatten``), and ``x'`` is the list
        of parameters after the encoded constaints have been satisfied.
    with_self (bool, default=True): if True, shortness is also self-consistent
    tol (float, default=0.0): maximum acceptable optimizer termination
        for ``sum(infeasibility)``.

Returns:
    None

Notes:
    - both ``scenario.positions`` and ``scenario.values`` may be adjusted.
    - if *with_self* is True, shortness will be measured not only from the
      scenario to the given *data*, but also between scenario datapoints.
"""
    # imposes: is_short(x, x'), is_short(x, z )
    # use additional 'constraints' kwds to impose: y >= m, norm(wi) = 1.0
    pm = impose_feasible(cutoff, data, guess=self.pts, bounds=bounds, \
                         constraints=constraints, with_self=with_self, **kwds)
    self.update( pm.flatten(all=True) )
    return

  def set_valid(self, model, cutoff=0.0, bounds=None, constraints=None, **kwds):
    """impose model validity on a scenario by adjusting positions and values

This function attempts to minimize the graph distance between reality (data),
``y = G(x)``, and an approximating function, ``y' = F(x')``, by perforing an
optimization on ``w,x,y`` over the given *bounds*.

Args:
    model (func): a model ``y' = F(x')`` that approximates reality ``y = G(x)``
    cutoff (float, default=0.0): acceptable model invalidity ``|y - F(x')|``
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``x' = constraints(x)``,
        where ``x`` is a scenario that has been converted into a list of
        parameters (e.g. with ``scenario.flatten``), and ``x'`` is the list
        of parameters after the encoded constaints have been satisfied.
    hausdorff (bool, default=False): hausdorff ``norm``, where if given,
        then ``ytol = |y - F(x')| + |x - x'|/norm``
    xtol (float, default=0.0): maximum acceptable pointwise graphical distance
        between model and reality.
    tol (float, default=0.0): maximum acceptable optimizer termination
        for ``sum(graphical distances)``.

Returns:
    None

Notes:
    *xtol* defines the n-dimensional base of a pilar of height *cutoff*,
    centered at each point. The region inside the pilar defines the space
    where a "valid" model must intersect. If *xtol* is not specified, then
    the base of the pilar will be a dirac at ``x' = x``. This function
    performs an optimization to find a set of points where the model is valid.
    Here, *tol* is used to set the optimization termination for minimizing the
    ``sum(graphical_distances)``, while *cutoff* is used in defining the
    graphical distance between ``x,y`` and ``x',F(x')``.
"""
    # imposes is_feasible(R, Cv), where R = graphical_distance(model, pts)
    # use additional 'constraints' kwds to impose: y >= m, norm(wi) = 1.0
    pm = impose_valid(cutoff, model, guess=self, \
                      bounds=bounds, constraints=constraints, **kwds)
    self.update( pm.flatten(all=True) )
    return
 
  def pof_value(self, f):
    """calculate probability of failure for a given function

Args:
    f (func): a function returning True for 'success' and False for 'failure'

Returns:
    the probabilty of failure, a float in ``[0.0,1.0]``

Notes:
    - the function ``f`` should take a list of ``values`` (for example,
      ``scenario.values``) and return a single value (e.g. 0.0 or False)
"""
    u = 0.0
    set = zip(self.values, self.weights)
    for x in set:
      if f(x[0]) <= 0.0:
        u += x[1]
    return u

  def update(self, params): #XXX: overwritten.  create standalone instead ?
    """update the scenario from a list of parameters

Args:
    params (list(float)): parameters corresponding to N 1D discrete measures

Returns:
    self (scenario): the scenario itself

Notes:
    The dimensions of the scenario will not change upon update, and
    it is assumed *params* either corresponds to the correct number of
    weights and positions for the existing ``scenario``, or *params*
    has additional values which will be saved as the ``scenario.values``.
    It is assumed that ``len(params) >= 2 * sum(scenario.pts)``.

    If ``scenario.pts = (M, N, ...)``, then it is assumed that
    ``params = [wx1, ..., wxM, x1, ..., xM, wy1, ..., wyN, y1, ..., yN, ...]``.
    Thus, *params* should have ``M`` weights and ``M`` corresponding positions,
    followed by ``N`` weights and ``N`` corresponding positions, with this
    pattern followed for each new dimension of the desired scenario.
"""
    pts = self.pts
    _len = 2 * sum(pts)

    if len(params)  >  _len:  # if Y-values are appended to params
      params, values  =  params[:_len], params[_len:]
      self.values = values[:len(self.values)] + self.values[len(values):] 

    pm = unflatten(params, pts)
    zo = pm.count([])
    self[:] = pm[:len(self) - zo] + self[len(pm) - zo:]
    return self

  def load(self, params, pts): #XXX: overwritten.  create standalone instead ?
    """load the scenario from a list of parameters

Args:
    params (list(float)): parameters corresponding to N 1D discrete measures
    pts (tuple(int)): number of point masses in each of the discrete measures

Returns:
    self (scenario): the scenario itself

Notes:
    To append ``len(pts)`` new discrete measures to the scenario,
    it is assumed *params* either corresponds to the correct number of
    weights and positions specified by *pts*, or *params* has additional
    values which will be saved as the ``scenario.values``. It is assumed
    that ``len(params) >= 2 * sum(scenario.pts)``.

    Given the value of ``pts = (M, N, ...)``, it is assumed that
    ``params = [wx1, ..., wxM, x1, ..., xM, wy1, ..., wyN, y1, ..., yN, ...]``.
    Thus, *params* should have ``M`` weights and ``M`` corresponding positions,
    followed by ``N`` weights and ``N`` corresponding positions, with this
    pattern followed for each new dimension of the desired scenario. Any
    remaining parameters will be treated as ``scenario.values``.
"""
    _len = 2 * sum(pts)
    if len(params)  >  _len:  # if Y-values are appended to params
      params, self.values  =  params[:_len], params[_len:]

    self.extend( unflatten(params, pts) )
    return self

  def flatten(self, all=True): #XXX: overwritten.  create standalone instead ?
    """convert a scenario to a single list of parameters

Args:
    all (bool, default=True): if True, append the scenario values

Returns:
    a list of parameters

Notes:
    Given ``scenario.pts = (M, N, ...)``, then the returned list is
    ``params = [wx1, ..., wxM, x1, ..., xM, wy1, ..., wyN, y1, ..., yN, ...]``.
    Thus, *params* will have ``M`` weights and ``M`` corresponding positions,
    followed by ``N`` weights and ``N`` corresponding positions, with this
    pattern followed for each new dimension of the scenario. If *all* is True,
    then the ``scenario.values`` will be appended to the list of parameters.
"""
    params = flatten(self)
    if all: params.extend(self.values) # if Y-values, return those as well
    return params

  # interface
  values = property(__values, __set_values, doc='a list of values corresponding to output data for all point masses in the underlying product measure')
  get_mean_value = mean_value
  pass


#---------------------------------------------
# creators and destructors from parameter list

def _mimic(samples, weights):
  """Generate a product_measure object from a list of N product measure
positions and a list of N weights. The resulting product measure will
mimic the original product measure's statistics, but be larger in size.

For example:
    >>> smp = [[-6,3,6],[-2,4],[1]]
    >>> wts = [[.4,.2,.4],[.5,.5],[1.]]
    >>> c = compose(samples, weights)
    >>> d = _mimic(c.positions, c.weights)
    >>> c[0].center_mass == d[0].center_mass
    True
    >>> c[1].range == d[1].range
    True
    >>> c.npts == d.npts
    False
    >>> c.npts == d[0].npts
    True
"""
  x = list(zip(*samples))               # 'mimic' to a nested list
  w = [weights for i in range(len(x))]  # 'mimic' to a nested list
  return compose(x,w)


def _uniform_weights(samples):
  """generate a nested list of N x 1D weights from a nested list of N x 1D
discrete measure positions, where the weights have norm 1.0 and are uniform.

>>> c.pos
[[1, 2, 3], [4, 5], [6]]
>>> _uniform_weights(c.pos)
[[0.333333333333333, 0.333333333333333, 0.333333333333333], [0.5, 0.5], [1.0]]
"""
  from mystic.math.measures import normalize
  return [normalize([1.]*len(xi), 1.0) for xi in samples]


def _list_of_measures(samples, weights=None):
  """generate a list of N x 1D discrete measures from a nested list of N x 1D
discrete measure positions and a nested list of N x 1D weights.

Note this function does not return a product measure, it returns a list."""
  total = []
  if not weights: weights = _uniform_weights(samples)
  for i in range(len(samples)):
    next = measure()
    for j in range(len(samples[i])):
      next.append(point_mass( samples[i][j], weights[i][j] ))
    total.append(next)
  return total


def compose(samples, weights=None):
  """Generate a product_measure object from a nested list of N x 1D
discrete measure positions and a nested list of N x 1D weights. If weights
are not provided, a uniform distribution with norm = 1.0 will be used."""
  if not weights: weights = _uniform_weights(samples)
  total = _list_of_measures(samples, weights)
  c = product_measure(total)
  return c


def decompose(c):
  """Decomposes a product_measure object into a nested list of
N x 1D discrete measure positions and a nested list of N x 1D weights."""
  from mystic.math.measures import _nested_split
  w, x = _nested_split(flatten(c), c.pts)
  return x, w


#def expand(data, npts):
#  """Generate a scenario object from a dataset. The scenario will have
#uniformly distributed weights and have dimensions given by pts."""
#  positions,values = data.fetch()
#  from mystic.math.measures import _unpack
#  pm = compose( _unpack(positions, npts) )
#  return scenario(pm, values[:pm.npts])


def unflatten(params, npts):
  """Map a list of random variables to N x 1D discrete measures
in a product_measure object."""
  from mystic.math.measures import _nested_split
  w, x = _nested_split(params, npts)
  return compose(x, w)


from itertools import chain #XXX: faster, but sloppy to have as importable
def flatten(c):
  """Flattens a product_measure object into a list."""
  rv = [(i.weights,i.positions) for i in c]
  # now flatten list of lists into just a list
  return list(chain(*chain(*rv))) # faster than mystic.tools.flatten


##### bounds-conserving-mean: borrowed from seismic/seismic.py #####
def bounded_mean(mean_x, samples, xmin, xmax, wts=None):
  from mystic.math.measures import impose_mean, impose_spread
  from mystic.math.measures import spread, mean
  from numpy import asarray
  a = impose_mean(mean_x, samples, wts)
  if min(a) < xmin:   # maintain the bound
    #print("violate lo(a)")
    s = spread(a) - 2*(xmin - min(a)) #XXX: needs compensation (as below) ?
    a = impose_mean(mean_x, impose_spread(s, samples, wts), wts)
  if max(a) > xmax:   # maintain the bound
    #print("violate hi(a)")
    s = spread(a) + 2*(xmax - max(a)) #XXX: needs compensation (as below) ?
    a = impose_mean(mean_x, impose_spread(s, samples, wts), wts)
  return asarray(a)
#####################################################################


#--------------------------------------------------
# constraints solvers and factories for feasibility

# used in self-consistent constraints function c(x) for
#   is_short(x, x') and is_short(x, z)
def norm_wts_constraintsFactory(pts):
  """factory for a constraints function that:
  - normalizes weights
"""
 #from measure import scenario
  def constrain(rv):
    "constrain:  sum(wi)_{k} = 1 for each k in K"
    pm = scenario()
    pm.load(rv, pts)      # here rv is param: w,x,y
    #impose: sum(wi)_{k} = 1 for each k in K
    norm = 1.0
    for i in range(len(pm)):
      w = pm[i].weights
      w[-1] = norm - sum(w[:-1])
      pm[i].weights = w
    rv = pm.flatten(all=True)
    return rv
  return constrain

# used in self-consistent constraints function c(x) for
#   is_short(x, x'), is_short(x, z), and y >= m
def mean_y_norm_wts_constraintsFactory(target, pts):
  """factory for a constraints function that:
  - imposes a mean on scenario values
  - normalizes weights
"""
 #from measure import scenario
  from mystic.math.measures import mean, impose_mean
 #target[0] is target mean
 #target[1] is acceptable deviation
  def constrain(rv):
    "constrain:  y >= m  and  sum(wi)_{k} = 1 for each k in K"
    pm = scenario()
    pm.load(rv, pts)      # here rv is param: w,x,y
    #impose: sum(wi)_{k} = 1 for each k in K
    norm = 1.0
    for i in range(len(pm)):
      w = pm[i].weights
      w[-1] = norm - sum(w[:-1])
      pm[i].weights = w
    #impose: y >= m 
    values, weights = pm.values, pm.weights
    y = float(mean(values, weights))
    if not (y >= float(target[0])):
      pm.values = impose_mean(target[0]+target[1], values, weights)
    rv = pm.flatten(all=True) 
    return rv
  return constrain

def impose_feasible(cutoff, data, guess=None, **kwds):
  """impose shortness on a given scenario

This function attempts to minimize the infeasibility between observed *data*
and a scenario of synthetic data by perforing an optimization on ``w,x,y``
over the given *bounds*.

Args:
    cutoff (float): maximum acceptable deviation from shortness
    data (mystic.math.discrete.scenario): a dataset of observed points
    guess (mystic.math.discrete.scenario, default=None): the synthetic points
    tol (float, default=0.0): maximum acceptable optimizer termination
        for ``sum(infeasibility)``.
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``x' = constraints(x)``,
        where ``x`` is a scenario that has been converted into a list of
        parameters (e.g. with ``scenario.flatten``), and ``x'`` is the list
        of parameters after the encoded constaints have been satisfied.

Returns:
    a scenario with desired shortness

Notes:
    Here, *tol* is used to set the optimization termination for minimizing the
    ``sum(infeasibility)``, while *cutoff* is used in defining the deviation
    from shortness for observed ``x,y`` and synthetic ``x',y'``.

    *guess* can be either a scenario providing initial guess at feasibility,
    or a tuple of the dimensions of the desired scenario, where initial
    values will be chosen at random.
"""
  from numpy import sum, asarray
  from mystic.math.legacydata import dataset
  from mystic.math.distance import lipschitz_distance, infeasibility, _npts
  if guess is None:
    message = "Requires a guess scenario, or a tuple of scenario dimensions."
    raise TypeError(message)
  # get initial guess
  if hasattr(guess, 'pts'): # guess is a scenario
    pts = guess.pts    # number of x
    guess = guess.flatten(all=True)
  else:
    pts = guess        # guess is given as a tuple of 'pts'
    guess = None
  npts = _npts(pts)    # number of Y
  long_form = len(pts) - list(pts).count(2) # can use '2^K compressed format'

  # prepare bounds for solver
  bounds = kwds.pop('bounds', None)
  # if bounds are not set, use the default optimizer bounds
  if bounds is None:
    lower_bounds = []; upper_bounds = []
    for n in pts:  # bounds for n*x in each dimension  (x2 due to weights)
      lower_bounds += [None]*n * 2
      upper_bounds += [None]*n * 2
    # also need bounds for npts*y values
    lower_bounds += [None]*npts
    upper_bounds += [None]*npts
    bounds = lower_bounds, upper_bounds
  bounds = asarray(bounds).T

  # plug in the 'constraints' function:  param' = constraints(param)
  # constraints should impose_mean(y,w), and possibly sum(weights)
  constraints = kwds.pop('constraints', None) # default is no constraints
  if not constraints:  # if None (default), there are no constraints
    constraints = lambda x: x

  _self = kwds.pop('with_self', True) # default includes self in shortness
  if _self is not False: _self = True
  # tolerance for optimization on sum(y)
  tol = kwds.pop('tol', 0.0) # default
  npop = kwds.pop('npop', 20) #XXX: tune npop?
  maxiter = kwds.pop('maxiter', 1000) #XXX: tune maxiter?

  # if no guess was made, then use bounds constraints
  if guess is None:
    if npop:
      guess = bounds
    else:  # fmin_powell needs a list params (not bounds)
      guess = [(a + b)/2. for (a,b) in bounds]

  # construct cost function to reduce sum(lipschitz_distance)
  def cost(rv):
    """compute cost from a 1-d array of model parameters,
    where:  cost = | sum(lipschitz_distance) | """
    _data = dataset()
    _pm = scenario()
    _pm.load(rv, pts)      # here rv is param: w,x,y
    if not long_form:
      positions = _pm.select(*range(npts))
    else: positions = _pm.positions
    _data.load( data.coords, data.values )                   # LOAD static
    if _self:
      _data.load( positions, _pm.values )                    # LOAD dynamic
    _data.lipschitz = data.lipschitz                         # LOAD L
    Rv = lipschitz_distance(_data.lipschitz, _pm, _data, tol=cutoff, **kwds)
    v = infeasibility(Rv, cutoff)
    return abs(sum(v))

  # construct and configure optimizer
  debug = False  #!!!
  maxfun = 1e+6
  crossover = 0.9; percent_change = 0.9
  ftol = abs(tol); gtol = None

  if debug:
    print("lower bounds: %s" % bounds.T[0])
    print("upper bounds: %s" % bounds.T[1])
  # print("initial value: %s" % guess)
  # use optimization to get feasible points
  from mystic.solvers import diffev2, fmin_powell
  from mystic.monitors import Monitor, VerboseMonitor
  from mystic.strategy import Best1Bin, Best1Exp
  evalmon = Monitor();  stepmon = Monitor(); strategy = Best1Exp
  if debug: stepmon = VerboseMonitor(10)  #!!!
  if npop: # use VTR
    results = diffev2(cost, guess, npop, ftol=ftol, gtol=gtol, bounds=bounds,\
                      maxiter=maxiter, maxfun=maxfun, constraints=constraints,\
                      cross=crossover, scale=percent_change, strategy=strategy,\
                      evalmon=evalmon, itermon=stepmon,\
                      full_output=1, disp=0, handler=False)
  else: # use VTR
    results = fmin_powell(cost, guess, ftol=ftol, gtol=gtol, bounds=bounds,\
                      maxiter=maxiter, maxfun=maxfun, constraints=constraints,\
                      evalmon=evalmon, itermon=stepmon,\
                      full_output=1, disp=0, handler=False)
  # repack the results
  pm = scenario()
  pm.load(results[0], pts)            # params: w,x,y
 #if debug: print("final cost: %s" % results[1])
  if debug and results[2] >= maxiter: # iterations
    print("Warning: constraints solver terminated at maximum iterations")
 #func_evals = results[3]           # evaluation
  return pm


def impose_valid(cutoff, model, guess=None, **kwds):
  """impose model validity on a given scenario

This function attempts to minimize the graph distance between reality (data),
``y = G(x)``, and an approximating function, ``y' = F(x')``, by perforing an
optimization on ``w,x,y`` over the given *bounds*.

Args:
    cutoff (float): maximum acceptable model invalidity ``|y - F(x')|``.
    model (func): the model function, ``y' = F(x')``.
    guess (scenario, default=None): a scenario, defines ``y = G(x)``.
    hausdorff (bool, default=False): hausdorff ``norm``, where if given,
        then ``ytol = |y - F(x')| + |x - x'|/norm``
    xtol (float, default=0.0): maximum acceptable pointwise graphical distance
        between model and reality.
    tol (float, default=0.0): maximum acceptable optimizer termination
        for ``sum(graphical distances)``.
    bounds (tuple, default=None): ``(all lower bounds, all upper bounds)``
    constraints (func, default=None): a function ``x' = constraints(x)``,
        where ``x`` is a scenario that has been converted into a list of
        parameters (e.g. with ``scenario.flatten``), and ``x'`` is the list
        of parameters after the encoded constaints have been satisfied.

Returns:
    a scenario with the desired model validity

Notes:
    *xtol* defines the n-dimensional base of a pilar of height *cutoff*,
    centered at each point. The region inside the pilar defines the space
    where a "valid" model must intersect. If *xtol* is not specified, then
    the base of the pilar will be a dirac at ``x' = x``. This function
    performs an optimization to find a set of points where the model is valid.
    Here, *tol* is used to set the optimization termination for minimizing the
    ``sum(graphical_distances)``, while *cutoff* is used in defining the
    graphical distance between ``x,y`` and ``x',F(x')``.

    *guess* can be either a scenario providing initial guess at validity,
    or a tuple of the dimensions of the desired scenario, where initial
    values will be chosen at random.
"""
  #FIXME: there are a lot of undocumented kwds (see below)
  from numpy import sum as _sum, asarray
  from mystic.math.distance import graphical_distance, infeasibility, _npts
  if guess is None:
    message = "Requires a guess scenario, or a tuple of scenario dimensions."
    raise TypeError(message)
  # get initial guess
  if hasattr(guess, 'pts'): # guess is a scenario
    pts = guess.pts    # number of x
    guess = guess.flatten(all=True)
  else:
    pts = guess        # guess is given as a tuple of 'pts'
    guess = None
  npts = _npts(pts)    # number of Y

  # prepare bounds for solver
  bounds = kwds.pop('bounds', None)
  # if bounds are not set, use the default optimizer bounds
  if bounds is None:
    lower_bounds = []; upper_bounds = []
    for n in pts:  # bounds for n*x in each dimension  (x2 due to weights)
      lower_bounds += [None]*n * 2
      upper_bounds += [None]*n * 2
    # also need bounds for npts*y values
    lower_bounds += [None]*npts
    upper_bounds += [None]*npts
    bounds = lower_bounds, upper_bounds
  bounds = asarray(bounds).T

  # plug in the 'constraints' function:  param' = constraints(param)
  constraints = kwds.pop('constraints', None) # default is no constraints
  if not constraints:  # if None (default), there are no constraints
    constraints = lambda x: x

  # 'wiggle room' tolerances
  ipop = kwds.pop('ipop', 10) #XXX: tune ipop (inner optimization)?
  imax = kwds.pop('imax', 10) #XXX: tune imax (inner optimization)?
  # tolerance for optimization on sum(y)
  tol = kwds.pop('tol', 0.0) # default
  npop = kwds.pop('npop', 20) #XXX: tune npop (outer optimization)?
  maxiter = kwds.pop('maxiter', 1000) #XXX: tune maxiter (outer optimization)?

  # if no guess was made, then use bounds constraints
  if guess is None:
    if npop:
      guess = bounds
    else:  # fmin_powell needs a list params (not bounds)
      guess = [(a + b)/2. for (a,b) in bounds]

  # construct cost function to reduce sum(infeasibility)
  def cost(rv):
    """compute cost from a 1-d array of model parameters,
    where: cost = | sum( infeasibility ) | """
    # converting rv to scenario
    points = scenario()
    points.load(rv, pts)
    # calculate infeasibility
    Rv = graphical_distance(model, points, ytol=cutoff, ipop=ipop, \
                                                        imax=imax, **kwds)
    v = infeasibility(Rv, cutoff)
    # converting v to E
    return _sum(v) #XXX: abs ?

  # construct and configure optimizer
  debug = False  #!!!
  maxfun = 1e+6
  crossover = 0.9; percent_change = 0.8
  ftol = abs(tol); gtol = None #XXX: optimally, should be VTRCOG...

  if debug:
    print("lower bounds: %s" % bounds.T[0])
    print("upper bounds: %s" % bounds.T[1])
  # print("initial value: %s" % guess)
  # use optimization to get model-valid points
  from mystic.solvers import diffev2, fmin_powell
  from mystic.monitors import Monitor, VerboseMonitor
  from mystic.strategy import Best1Bin, Best1Exp
  evalmon = Monitor();  stepmon = Monitor(); strategy = Best1Exp
  if debug: stepmon = VerboseMonitor(2)  #!!!
  if npop: # use VTR
    results = diffev2(cost, guess, npop, ftol=ftol, gtol=gtol, bounds=bounds,\
                      maxiter=maxiter, maxfun=maxfun, constraints=constraints,\
                      cross=crossover, scale=percent_change, strategy=strategy,\
                      evalmon=evalmon, itermon=stepmon,\
                      full_output=1, disp=0, handler=False)
  else: # use VTR
    results = fmin_powell(cost, guess, ftol=ftol, gtol=gtol, bounds=bounds,\
                      maxiter=maxiter, maxfun=maxfun, constraints=constraints,\
                      evalmon=evalmon, itermon=stepmon,\
                      full_output=1, disp=0, handler=False)
  # repack the results
  pm = scenario()
  pm.load(results[0], pts)            # params: w,x,y
 #if debug: print("final cost: %s" % results[1])
  if debug and results[2] >= maxiter: # iterations
    print("Warning: constraints solver terminated at maximum iterations")
 #func_evals = results[3]           # evaluation
  return pm


# backward compatibility
point = point_mass
dirac_measure = measure


if __name__ == '__main__':
  from mystic.math.distance import *
  model = lambda x:sum(x)
  a = [0,1,9,8, 1,0,4,6, 1,0,1,2, 0,1,2,3,4,5,6,7]
  feasability = 0.0; deviation = 0.01
  validity = 5.0; wiggle = 1.0
  y_mean = 5.0; y_buffer = 0.0
  L = [.75,.5,.25]
  bc = [(0,7,2),(3,0,2),(2,0,3),(1,0,3),(2,4,2)]
  bv = [5,3,1,4,8]
  pts = (2,2,2)
  from mystic.math.legacydata import dataset
  data = dataset()
  data.load(bc, bv)
  data.lipschitz = L
  pm = scenario()
  pm.load(a, pts)
  pc = pm.positions
  pv = pm.values
  #---
  _data = dataset()
  _data.load(bc, bv)
  _data.load(pc, pv)
  _data.lipschitz = data.lipschitz
  from numpy import sum
  ans = sum(lipschitz_distance(L, pm, _data))
  print("original: %s @ %s\n" % (ans, a))
 #print("pm: %s" % pm)
 #print("data: %s" % data)
  #---
  lb = [0,.5,-100,-100,  0,.5,-100,-100,  0,.5,-100,-100,   0,0,0,0,0,0,0,0]
  ub = [.5,1, 100, 100,  .5,1, 100, 100,  .5,1, 100, 100,   9,9,9,9,9,9,9,9]
  bounds = (lb,ub)

  _constrain = mean_y_norm_wts_constraintsFactory((y_mean,y_buffer), pts)
  results = impose_feasible(feasability, data, guess=pts, tol=deviation, \
                            bounds=bounds, constraints=_constrain)
  from mystic.math.measures import mean
  print("solved: %s" % results.flatten(all=True))
  print("mean(y): %s >= %s" % (mean(results.values, results.weights), y_mean))
  print("sum(wi): %s == 1.0" % [sum(w) for w in results.wts])

  print("\n---------------------------------------------------\n")

  bc = bc[:-2]
  ids = ['1','2','3']
  t = dataset()
  t.load(bc, list(map(model, bc)), ids)
  t.update(t.coords, list(map(model, t.coords)))
# r = dataset()
# r.load(t.coords, t.values)
# L = [0.1, 0.0, 0.0]
  print("%s" % t)
  print("L: %s" % L)
  print("shortness:")
  print(lipschitz_distance(L, t, t, tol=0.0))
  
  print("\n---------------------------------------------------\n")

  print("Y: %s" % str(results.values))
  print("sum(wi): %s == 1.0" % [sum(w) for w in results.wts])


# EOF
