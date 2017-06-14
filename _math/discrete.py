#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Classes for discrete measure data objects.
Includes point_mass, measure, product_measure, and scenario classes.
"""
from __future__ import absolute_import

# Adapted from seesaw2d.py in branches/UQ/math/examples2/ 
# For usage example, see seesaw2d_inf_example.py .

from mystic.math.measures import impose_mean, impose_expectation
from mystic.math.measures import impose_spread, impose_variance
from mystic.math.measures import impose_weight_norm

class point_mass(object):
  """ a point_mass object with weight and position

 queries:
  p.weight   --  returns weight
  p.position  --  returns position
  p.rms  --  returns the square root of sum of squared position

 settings:
  p.weight = w1  --  set the weight
  p.position = x1  --  set the position
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
  rms = property(__rms)

  pass

class measure(list):  #FIXME: meant to only accept point_masses...
  """ a 1-d collection of point_masses forming a 'discrete_measure'
  s = measure([point_mass1, point_mass2, ..., point_massN])  
    where a point_mass has weight and position

 queries:
  s.weights   --  returns list of weights
  s.positions  --  returns list of positions
  s.npts  --  returns the number of point_masses
  s.mass  --  calculates sum of weights
  s.center_mass  --  calculates sum of weights*positions
  s.range  --  calculates |max - min| for positions
  s.var  --  calculates mean( |positions - mean(positions)|**2 )

 settings:
  s.weights = [w1, w2, ..., wn]  --  set the weights
  s.positions = [x1, x2, ..., xn]  --  set the positions
  s.normalize()  --  normalize the weights to 1.0
  s.center_mass(R)  --  set the center of mass
  s.range(R)  --  set the range
  s.var(R)  --  set the variance

 methods:
  s.support()  -- get the positions that have corresponding non-zero weights
  s.support_index()  --  get the indices of positions that have support
  s.maximum(f)  --  calculate the maximum for a given function
  s.minimum(f)  --  calculate the minimum for a given function
  s.ess_maximum(f)  --  calculate the maximum for support of a given function
  s.ess_minimum(f)  --  calculate the minimum for support of a given function
  s.expect(f)  --  calculate the expectation
  s.set_expect((center,delta), f)  --  impose expectation by adjusting positions

 notes:
  - constraints should impose that sum(weights) should be 1.0
  - assumes that s.n = len(s.positions) == len(s.weights)
"""

  def support_index(self, tol=0):
    """get the indices of the positions which have non-zero weight

Inputs:
    tol -- weight tolerance, where any weight <= tol is considered zero
"""
    from .measures import support_index
    return support_index(self.weights, tol)

  def support(self, tol=0):
    """get the positions which have non-zero weight

Inputs:
    tol -- weight tolerance, where any weight <= tol is considered zero
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
    """normalize the weights"""
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

Inputs:
    f -- a function that takes a list and returns a number
"""
    from .measures import maximum
    return maximum(f, self.positions)

  def ess_maximum(self, f, tol=0.):
    """calculate the maximum for the support of a given function

Inputs:
    f -- a function that takes a list and returns a number
    tol -- weight tolerance, where any weight <= tol is considered zero
"""
    from .measures import ess_maximum
    return ess_maximum(f, self.positions, self.weights, tol)

  def minimum(self, f):
    """calculate the minimum for a given function

Inputs:
    f -- a function that takes a list and returns a number
"""
    from .measures import minimum
    return minimum(f, self.positions)

  def ess_minimum(self, f, tol=0.):
    """calculate the minimum for the support of a given function

Inputs:
    f -- a function that takes a list and returns a number
    tol -- weight tolerance, where any weight <= tol is considered zero
"""
    from .measures import ess_minimum
    return ess_minimum(f, self.positions, self.weights, tol)

  def expect(self, f):
    """calculate the expectation for a given function

Inputs:
    f -- a function that takes a list and returns a number
""" #XXX: maybe more natural if f takes a positional value x, not a list x ?
    from mystic.math.measures import expectation
    positions = [(i,) for i in self.positions]
    return expectation(f, positions, self.weights)

  def set_expect(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose a expectation on a dirac measure

Inputs:
    expected -- tuple of expectation m and acceptable deviation D
    f -- a function that takes a list and returns a number
    bounds -- tuple of lists of bounds  (lower_bounds, upper_bounds)
    constraints -- a function that takes a product_measure  c' = constraints(c)
""" #XXX: maybe more natural if f takes a positional value x, not a list x ?
    #XXX: maybe also natural c' = constraints(c) where c is a measure ?
    #FIXME: undocumented npop, maxiter, maxfun
    m,D = expected
    if constraints:  # then need to adjust interface for 'impose_expectation'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    positions = impose_expectation((m,D), f, [self.npts], bounds, \
                                   self.weights, constraints=cnstr, **kwds)
    from numpy import array
    self.positions = list(array(positions)[:,0])
   #from numpy import squeeze
   #self.positions = list(squeeze(positions))
    return

  # interface
  weights = property(__weights, __set_weights)
  positions = property(__positions, __set_positions)
  ###XXX: why not use 'points' also/instead?
  npts = property(__n )
  mass = property(__mass )
  range = property(__range, __set_range)
  center_mass = property(__mean, __set_mean)
  var = property(__variance, __set_variance)

  # backward compatibility
  coords = positions
  get_expect = expect
  mean = center_mass
  pass

class product_measure(list):  #FIXME: meant to only accept sets...
  """ a N-d measure-theoretic product of discrete measures
  c = product_measure([measure1, measure2, ..., measureN])  
    where all measures are orthogonal

 queries:
  c.npts  --  returns total number of point_masses
  c.weights   --  returns list of weights
  c.positions  --  returns list of position tuples
  c.mass  --  returns list of weight norms
  c.pts  --  returns number of point_masses for each discrete measure
  c.wts  --  returns list of weights for each discrete measure
  c.pos  --  returns list of positions for each discrete measure

 settings:
  c.positions = [(x1,y1,z1),...]  --  set positions (tuples in product measure)

 methods:
  c.pof(f)  --  calculate the probability of failure
  c.sampled_pof(f, npts) -- calculate the pof using sampled point_masses
  c.expect(f)  --  calculate the expectation
  c.set_expect((center,delta), f)  --  impose expectation by adjusting positions
  c.flatten()  --  convert measure to a flat list of parameters
  c.load(params, pts)  --  'fill' the measure from a flat list of parameters
  c.update(params) -- 'update' the measure from a flat list of parameters

 notes:
  - constraints impose expect (center - delta) <= E <= (center + delta)
  - constraints impose sum(weights) == 1.0 for each set
  - assumes that c.npts = len(c.positions) == len(c.weights)
  - weight wxi should be same for each (yj,zk) at xi; similarly for wyi & wzi
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
    from .measures import support_index
    return support_index(self.weights, tol)

  def support(self, tol=0): #XXX: better if generated positions only when needed
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
    return max([i.maximum(f) for i in self])

  def minimum(self, f): #XXX: return min of all or return all min?
    return min([i.minimum(f) for i in self])

  def ess_maximum(self, f, tol=0.): #XXX: return max of all or return all max?
    return max([i.ess_maximum(f, tol) for i in self])

  def ess_minimum(self, f, tol=0.): #XXX: return min of all or return all min?
    return min([i.ess_minimum(f, tol) for i in self])

  def expect(self, f):
    """calculate the expectation for a given function

Inputs:
    f -- a function that takes a list and returns a number
"""
    from mystic.math.measures import expectation
    return expectation(f, self.positions, self.weights)

  def set_expect(self, expected, f, bounds=None, constraints=None, **kwds):
    """impose a expectation on a product measure

Inputs:
    expected -- tuple of expectation m and acceptable deviation D
    f -- a function that takes a list and returns a number
    bounds -- tuple of lists of bounds  (lower_bounds, upper_bounds)
    constraints -- a function that takes a product_measure  c' = constraints(c)
"""
    #FIXME: undocumented npop, maxiter, maxfun
    #self.__center = m
    #self.__delta = D
    m,D = expected
    if constraints:  # then need to adjust interface for 'impose_expectation'
      def cnstr(x, w):
        c = compose(x,w)
        c = constraints(c)
        return decompose(c)[0]
    else: cnstr = constraints  # 'should' be None
    self.positions = impose_expectation((m,D), f, self.pts, bounds, \
                                        self.weights, constraints=cnstr, **kwds)
    return

  def pof(self, f):
    """calculate probability of failure over a given function, f,
where f takes a list of (product_measure) positions and returns a single value

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
"""
    u = 0.0
    set = zip(self.positions, self.weights)
    for x in set:
      if f(x[0]) <= 0.0:
        u += x[1]
    return u
  # for i in range(self.npts):
  #   #if f(self.positions[i]) > 0.0:  #NOTE: f(x) > 0.0 yields prob of success
  #   if f(self.positions[i]) <= 0.0:  #NOTE: f(x) <= 0.0 yields prob of failure
  #     u += self.weights[i]
  # return u  #XXX: does this need to be normalized?

  def sampled_pof(self, f, npts=10000):
    """calculate probability of failure over a given function, f,
where f takes a list of (product_measure) positions and returns a single value

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
    npts -- number of point_masses sampled from the underlying discrete measures
"""
    from mystic.math.samples import _pof_given_samples
    pts = self.sampled_support(npts)
    return _pof_given_samples(f, pts)

  def sampled_support(self, npts=10000): ##XXX: was 'def support'
    """randomly select support points from the underlying discrete measures

Inputs:
    npts -- number of points sampled from the underlying discrete measures

Returns:
    pts -- a nested list of len(prod_measure) lists, each of len(npts)
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

The dimensions of the product measure will not change"""
    pts = self.pts
    _len = 2 * sum(pts)

    if len(params)  >  _len:  # if Y-values are appended to params
      params, values  =  params[:_len], params[_len:]

    pm = unflatten(params, pts)
    zo = pm.count([])
    self[:] = pm[:len(self) - zo] + self[len(pm) - zo:]
    return

  def load(self, params, pts):
    """load a list of parameters corresponding to N x 1D discrete measures

Inputs:
    params -- a list of parameters (see 'notes')
    pts -- number of point_masses in each of the underlying discrete measures

Notes:
    To append len(pts) new discrete measures to product measure c, where
    pts = (M, N, ...)
    params = [wt_x1, ..., wt_xM, \
                 x1, ..., xM,    \
              wt_y1, ..., wt_yN, \
                 y1, ..., yN,    \
                     ...]
    Thus, the provided list is M weights and the corresponding M positions,
    followed by N weights and the corresponding N positions, with this
    pattern followed for each new dimension desired for the product measure.
"""
    _len = 2 * sum(pts)
    if len(params)  >  _len:  # if Y-values are appended to params
      params, values  =  params[:_len], params[_len:]

    self.extend( unflatten(params, pts) )
    return

  def flatten(self):
    """flatten the product_measure into a list of parameters

Returns:
    params -- a list of parameters (see 'notes')

Notes:
    For a product measure c where c.pts = (M, N, ...), then
    params = [wt_x1, ..., wt_xM, \
                 x1, ..., xM,    \
              wt_y1, ..., wt_yN, \
                 y1, ..., yN,    \
                     ...]
    Thus, the returned list is M weights and the corresponding M positions,
    followed by N weights and the corresponding N positions, with this
    pattern followed for each dimension of the product measure.
"""
    params = flatten(self)
    return params

  #XXX: name stinks... better as "non_redundant"? ...is really a helper
  def differs_by_one(self, ith, all=True, index=True):
    """get the product measure coordinates where the associated binary
string differs by exactly one index

  Inputs:
    ith   = the target index
    all   = if False, return only the results for indices < i
    index = if True, return the index of the results (not results themselves)
"""
    from mystic.math.compressed import index2binary, differs_by_one
    b = index2binary(list(range(self.npts)), self.npts)
    return differs_by_one(ith, b, all, index) 

  def select(self, *index, **kwds):
    """generator for product measure positions due to selected position indices
 (NOTE: only works for product measures of dimension 2^K)

  >>> r
  [[9, 8], [1, 3], [4, 2]]
  >>> r.select(*range(r.npts))
  [(9, 1, 4), (8, 1, 4), (9, 3, 4), (8, 3, 4), (9, 1, 2), (8, 1, 2), (9, 3, 2), (8, 3, 2)]
  >>>
  >>> _pack(r)
  [(9, 1, 4), (8, 1, 4), (9, 3, 4), (8, 3, 4), (9, 1, 2), (8, 1, 2), (9, 3, 2), (8, 3, 2)]
"""
    from mystic.math.compressed import index2binary, binary2coords
    v = index2binary(list(index), self.npts)
    return binary2coords(v, self.pos, **kwds)
    #XXX: '_pack' requires resorting ([::-1]) so that indexing is wrong.
    #     Better if modify mystic's pack to match sorting of binary strings ?

 #__center = None
 #__delta = None

  # interface
  npts = property(__n )
  weights = property(__weights )
  positions = property(__positions, __set_positions )
  center_mass = property(__mean, __set_mean)
 #center = property(__get_center ) #FIXME: remove c.center and c.delta... or
 #delta = property(__get_delta )   #       replace with c._params (e.g. (m,D))
 #expect = property(__expect, __set_expect )
  mass = property(__mass )
  pts = property(__pts )
  wts = property(__wts )
  pos = property(__pos )

  # backward compatibility
  coords = positions
  get_expect = expect
  pass


class scenario(product_measure):  #FIXME: meant to only accept sets...
  """ a N-d product measure (collection of dirac measures) with values
  s = scenario(product_measure, [value1, value2, ..., valueN])  
    where each point_mass in the product measure is paried with a value
    (essentially, a dataset in product_measure representation)

 queries:
  s.npts  --  returns total number of point_masse
  s.weights   --  returns list of weights
  s.positions  --  returns list of position tuples
  s.values  --  returns list of values
  s.mass  --  returns list of weight norms
  s.pts  --  returns number of point_masses for each discrete measure
  s.wts  --  returns list of weights for each discrete measure
  s.pos  --  returns list of positions for each discrete measure

 settings:
  s.positions = [(x1,y1,z1),...]  --  set positions (tuples in product measure)
  s.values = [v1,v2,v3,...]  --  set the values (correspond to position tuples)

 methods:
  s.pof(f)  --  calculate the probability of failure
  s.pof_value(f)  --  calculate the probability of failure using the values
  s.sampled_pof(f, npts) -- calculate the pof using sampled points
  s.expect(f)  --  calculate the expectation
  s.set_expect((center,delta), f)  --  impose expectation by adjusting positions
  s.mean_value()  --  calculate the mean values for a scenario
  s.set_mean_value(m)  --  impose mean value by adjusting values
  s.set_feasible(data)  --  impose shortness by adjusting positions and values
  s.short_wrt_data(data) -- check for shortness with respect to data
  s.short_wrt_self(L) -- check for shortness with respect to self
  s.set_valid(model) -- impose validity by adjusting positions and values
  s.valid_wrt_model(model) -- check for validity with respect to the model
  s.flatten()  --  convert measure to a flat list of parameters
  s.load(params, pts)  --  'fill' the measure from a flat list of parameters
  s.update(params) -- 'update' the measure from a flat list of parameters

 notes:
  - constraints impose expect (center - delta) <= E <= (center + delta)
  - constraints impose sum(weights) == 1.0 for each set
  - assumes that s.npts = len(s.positions) == len(s.weights)
  - weight wxi should be same for each (yj,zk) at xi; similarly for wyi & wzi
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
    """calculate the mean of the associated values for a scenario"""
    from mystic.math.measures import mean
    return mean(self.values, self.weights)

  def set_mean_value(self, m):  # set mean of y's
    """set the mean for the associated values of a scenario"""
    from mystic.math.measures import impose_mean
    self.values = impose_mean(m, self.values, self.weights)
    return

  def valid_wrt_model(self, model, blamelist=False, pairs=True, \
                                   all=False, raw=False, **kwds):
    """check for scenario validity with respect to the model

Inputs:
    model -- the model function, y' = F(x')
    blamelist -- if True, report which points are infeasible
    pairs -- if True, report indices of infeasible points
    all -- if True, report results for each point (opposed to all points)
    raw -- if True, report numerical results (opposed to boolean results)

Additional Inputs:
    ytol -- maximum acceptable difference |y - F(x')|; a single value
    xtol -- maximum acceptable difference |x - x'|; an iterable or single value
    cutoff -- zero out distances less than cutoff; typically: ytol, 0.0, or None
    hausdorff -- norm; where if given, ytol = |y - F(x')| + |x - x'|/norm

Notes:
    xtol defines the n-dimensional base of a pilar of height ytol, centered at
    each point. The region inside the pilar defines the space where a "valid"
    model must intersect. If xtol is not specified, then the base of the pilar
    will be a dirac at x' = x. This function performs an optimization for each
    x to find an appropriate x'. While cutoff and ytol are very tightly related,
    they play a distinct role; ytol is used to set the optimization termination
    for an acceptable |y - F(x')|, while cutoff is applied post-optimization.
    If we are using the hausdorff norm, then ytol will set the optimization
    termination for an acceptable |y - F(x')| + |x - x'|/norm, where the x
    values are normalized by norm = hausdorff.
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

Inputs:
    L -- the lipschitz constant
    blamelist -- if True, report which points are infeasible
    pairs -- if True, report indices of infeasible points
    all -- if True, report results for each point (opposed to all points)
    raw -- if True, report numerical results (opposed to boolean results)

Additional Inputs:
    tol -- maximum acceptable deviation from shortness
    cutoff -- zero out distances less than cutoff; typically: tol, 0.0, or None

Notes:
    Each point x,y can be thought to have an associated double-cone with slope
    equal to the lipschitz constant. Shortness with respect to another point is
    defined by the first point not being inside the cone of the second. We can
    allow for some error in shortness, a short tolerance 'tol', for which the
    point x,y is some acceptable y-distance inside the cone. While very tightly
    related, cutoff and tol play distinct roles; tol is subtracted from
    calculation of the lipschitz_distance, while cutoff zeros out the value
    of any element less than the cutoff.
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

Inputs:
    data -- a collection of data points
    L -- the lipschitz constant, if different from that provided with data
    blamelist -- if True, report which points are infeasible
    pairs -- if True, report indices of infeasible points
    all -- if True, report results for each point (opposed to all points)
    raw -- if True, report numerical results (opposed to boolean results)

Additional Inputs:
    tol -- maximum acceptable deviation from shortness
    cutoff -- zero out distances less than cutoff; typically cutoff = tol or 0.0

Notes:
    Each point x,y can be thought to have an associated double-cone with slope
    equal to the lipschitz constant. Shortness with respect to another point is
    defined by the first point not being inside the cone of the second. We can
    allow for some error in shortness, a short tolerance 'tol', for which the
    point x,y is some acceptable y-distance inside the cone. While very tightly
    related, cutoff and tol play distinct roles; tol is subtracted from
    calculation of the lipschitz_distance, while cutoff zeros out the value
    of any element less than the cutoff.
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
    """impose shortness on a scenario with respect to given data points

Inputs:
    data -- a collection of data points
    cutoff -- acceptable deviation from shortness

Additional Inputs:
    with_self -- if True, shortness will also be imposed with respect to self
    tol -- acceptable optimizer termination before sum(infeasibility) = 0.
    bounds -- a tuple of sample bounds:   bounds = (lower_bounds, upper_bounds)
    constraints -- a function that takes a flat list parameters
        x' = constraints(x)
"""
    # imposes: is_short(x, x'), is_short(x, z )
    # use additional 'constraints' kwds to impose: y >= m, norm(wi) = 1.0
    pm = impose_feasible(cutoff, data, guess=self.pts, bounds=bounds, \
                         constraints=constraints, with_self=with_self, **kwds)
    self.update( pm.flatten(all=True) )
    return

  def set_valid(self, model, cutoff=0.0, bounds=None, constraints=None, **kwds):
    """impose validity on a scenario with respect to given data points

Inputs:
    model -- the model function, y' = F(x'), that approximates reality, y = G(x)
    cutoff -- acceptable model invalidity |y - F(x')|

Additional Inputs:
    hausdorff -- norm; where if given, ytol = |y - F(x')| + |x - x'|/norm
    xtol -- acceptable pointwise graphical distance of model from reality
    tol -- acceptable optimizer termination before sum(infeasibility) = 0.
    bounds -- a tuple of sample bounds:   bounds = (lower_bounds, upper_bounds)
    constraints -- a function that takes a flat list parameters
        x' = constraints(x)

Notes:
    xtol defines the n-dimensional base of a pilar of height cutoff, centered at
    each point. The region inside the pilar defines the space where a "valid"
    model must intersect. If xtol is not specified, then the base of the pilar
    will be a dirac at x' = x. This function performs an optimization to find
    a set of points where the model is valid. Here, tol is used to set the
    optimization termination for the sum(graphical_distances), while cutoff is
    used in defining the graphical_distance between x,y and x',F(x').
"""
    # imposes is_feasible(R, Cv), where R = graphical_distance(model, pts)
    # use additional 'constraints' kwds to impose: y >= m, norm(wi) = 1.0
    pm = impose_valid(cutoff, model, guess=self, \
                      bounds=bounds, constraints=constraints, **kwds)
    self.update( pm.flatten(all=True) )
    return
 
  def pof_value(self, f):
    """calculate probability of failure over a given function, f,
where f takes a list of (scenario) values and returns a single value

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
"""
    u = 0.0
    set = zip(self.values, self.weights)
    for x in set:
      if f(x[0]) <= 0.0:
        u += x[1]
    return u

  def update(self, params): #XXX: overwritten.  create standalone instead ?
    """update the scenario from a list of parameters

The dimensions of the scenario will not change"""
    pts = self.pts
    _len = 2 * sum(pts)

    if len(params)  >  _len:  # if Y-values are appended to params
      params, values  =  params[:_len], params[_len:]
      self.values = values[:len(self.values)] + self.values[len(values):] 

    pm = unflatten(params, pts)
    zo = pm.count([])
    self[:] = pm[:len(self) - zo] + self[len(pm) - zo:]
    return

  def load(self, params, pts): #XXX: overwritten.  create standalone instead ?
    """load a list of parameters corresponding to N x 1D discrete measures

Inputs:
    params -- a list of parameters (see 'notes')
    pts -- number of points in each of the underlying discrete measures

Notes:
    To append len(pts) new discrete measures to scenario c, where
    pts = (M, N, ...)
    params = [wt_x1, ..., wt_xM, \
                 x1, ..., xM,    \
              wt_y1, ..., wt_yN, \
                 y1, ..., yN,    \
                     ...]
    Thus, the provided list is M weights and the corresponding M positions,
    followed by N weights and the corresponding N positions, with this
    pattern followed for each new dimension desired for the scenario.
"""
    _len = 2 * sum(pts)
    if len(params)  >  _len:  # if Y-values are appended to params
      params, self.values  =  params[:_len], params[_len:]

    self.extend( unflatten(params, pts) )
    return

  def flatten(self, all=True): #XXX: overwritten.  create standalone instead ?
    """flatten the scenario into a list of parameters

Returns:
    params -- a list of parameters (see 'notes')

Notes:
    For a scenario c where c.pts = (M, N, ...), then
    params = [wt_x1, ..., wt_xM, \
                 x1, ..., xM,    \
              wt_y1, ..., wt_yN, \
                 y1, ..., yN,    \
                     ...]
    Thus, the returned list is M weights and the corresponding M positions,
    followed by N weights and the corresponding N positions, with this
    pattern followed for each dimension of the scenario.
"""
    params = flatten(self)
    if all: params.extend(self.values) # if Y-values, return those as well
    return params

  # interface
  values = property(__values, __set_values )
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
  """impose shortness on a given list of parameters w,x,y.

Optimization on w,x,y over the given bounds seeks sum(infeasibility) = 0.
  (this function is not ???-preserving)

Inputs:
    cutoff -- maximum acceptable deviation from shortness
    data -- a dataset of observed points (these points are 'static')
    guess -- the scenario providing an initial guess at feasibility,
        or a tuple of dimensions of the target scenario

Additional Inputs:
    tol -- acceptable optimizer termination before sum(infeasibility) = 0.
    bounds -- a tuple of sample bounds:   bounds = (lower_bounds, upper_bounds)
    constraints -- a function that takes a flat list parameters
        x' = constraints(x)

Outputs:
    pm -- a scenario with desired shortness
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
  """impose model validity on a given list of parameters w,x,y

Optimization on w,x,y over the given bounds seeks sum(infeasibility) = 0.
  (this function is not ???-preserving)

Inputs:
    cutoff -- maximum acceptable model invalidity |y - F(x')|; a single value
    model -- the model function, y' = F(x'), that approximates reality, y = G(x)
    guess -- the scenario providing an initial guess at validity,
        or a tuple of dimensions of the target scenario

Additional Inputs:
    hausdorff -- norm; where if given, ytol = |y - F(x')| + |x - x'|/norm
    xtol -- acceptable pointwise graphical distance of model from reality
    tol -- acceptable optimizer termination before sum(infeasibility) = 0.
    bounds -- a tuple of sample bounds:   bounds = (lower_bounds, upper_bounds)
    constraints -- a function that takes a flat list parameters
        x' = constraints(x)

Outputs:
    pm -- a scenario with desired model validity

Notes:
    xtol defines the n-dimensional base of a pilar of height cutoff, centered at
    each point. The region inside the pilar defines the space where a "valid"
    model must intersect. If xtol is not specified, then the base of the pilar
    will be a dirac at x' = x. This function performs an optimization to find
    a set of points where the model is valid. Here, tol is used to set the
    optimization termination for the sum(graphical_distances), while cutoff is
    used in defining the graphical_distance between x,y and x',F(x').
"""
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
