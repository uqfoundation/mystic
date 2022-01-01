#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# Adapted from seesaw2d.py in branches/UQ/math/examples2/ 
# For usage example, see seesaw2d_inf_example.py .
"""
TESTS for Dirac measure data objects.
Includes point_mass, measure, and product_measure classes.
"""

from mystic.math.discrete import point_mass as point
from mystic.math.discrete import measure as set
from mystic.math.discrete import product_measure as collection
from mystic.math.samples import random_samples
from mystic.math.grid import samplepts

from mystic.math import almostEqual
from mystic.math.measures import *
from mystic.math.measures import _pack, _unpack

disp = False

def test_calculate_methods(npts=2):
  upper_bounds = [1.0]
  lower_bounds = [0.0]

  # -------------------------------------
  # generate initial coordinates, weights 
  # -------------------------------------
  # get a random distribution of points
  if disp: print("generate random points and weights")
  coordinates = samplepts(lower_bounds, upper_bounds, npts)
  D0 = D = [i[0] for i in coordinates]
  if disp: print("positions: %s" % D)

  # calculate sample range
  R0 = R = spread(D)
  if disp: print("range: %s" % R)

  # select weights randomly in [0,1], then normalize so sum(weights) = 1
  wts = random_samples([0],[1], npts)[0]
  weights = normalize(wts, 0.0, zsum=True)
  if disp: print("weights (when normalized to 0.0): %s" % weights)
  assert almostEqual(sum(weights), 0.0, tol=1e-15)
  weights = normalize(wts, 1.0)
  assert almostEqual(sum(weights), 1.0, tol=1e-15)
  if disp: print("weights (when normalized to 1.0): %s" % weights)
  w = norm(weights)
  if disp: print("norm: %s" % w)
  assert almostEqual(w, sum(weights)/npts)

  # calculate sample mean
  m0 = m = mean(D,weights)
  if disp: print("mean: %s" % m)
  if disp: print("")

  # -------------------------------------
  # modify coordinates, maintaining mean & range 
  # -------------------------------------
  # get new random distribution
  if disp: print("modify positions, maintaining mean and range")
  coordinates = samplepts(lower_bounds, upper_bounds, npts)
  D = [i[0] for i in coordinates]

  # impose a range and mean on the points
  D = impose_spread(R, D, weights)
  D = impose_mean(m, D, weights)

  # print results
  if disp: print("positions: %s" % D)
  R = spread(D)
  if disp: print("range: %s" % R)
  assert almostEqual(R, R0)
  m = mean(D, weights)
  if disp: print("mean: %s" % m)
  assert almostEqual(m, m0)
  if disp: print("")

  # -------------------------------------
  # modify weights, maintaining mean & norm
  # -------------------------------------
  # select weights randomly in [0,1]
  if disp: print("modify weights, maintaining mean and range")
  wts = random_samples([0],[1], npts)[0]

  # print intermediate results
 #print("weights: %s" % wts)
 #sm = mean(D, wts)
 #print("tmp mean: %s" % sm)
 #print("")

  # impose mean and weight norm on the points
  D = impose_mean(m, D, wts)
  DD, weights = impose_weight_norm(D, wts)

  # print results
  if disp: print("weights: %s" % weights)
  w = norm(weights)
  if disp: print("norm: %s" % w)
  assert almostEqual(w, sum(weights)/npts)
  if disp: print("positions: %s" % DD)
  R = spread(DD)
  if disp: print("range: %s" % R)
  assert almostEqual(R, R0)
  sm = mean(DD, weights)
  if disp: print("mean: %s" % sm)
  assert almostEqual(sm, m0)
  sv = variance(DD, weights)
  if disp: print("var: %s" % sv)
  assert not almostEqual(sv, R)
  assert almostEqual(sv, 0.0, tol=.3)

  # -------------------------------------
  # modify variance, maintaining mean
  # -------------------------------------
  if disp: print("\nmodify variance, maintaining mean")
  DD = impose_variance(R, DD, weights)
  sm = mean(DD, weights)
  if disp: print("mean: %s" % sm)
  assert almostEqual(sm, m0)
  sv = variance(DD, weights)
  if disp: print("var: %s" % sv)
  assert almostEqual(sv, R)
  

def test_set_behavior():
  from mystic.math import almostEqual
  from numpy import inf

  # check basic behavior for set of two points
  s = set([point(1.0,1.0), point(3.0,2.0)]) #XXX: s + [pt, pt] => broken
  assert almostEqual(s.mean, 2.33333333)
  assert almostEqual(s.range, 2.0)
  assert almostEqual(s.mass, 3.0)

  # basic behavior for an admissable set
  s.normalize()
  s.mean = 1.0
  s.range = 1.0
  assert almostEqual(s.mean, 1.0)
  assert almostEqual(s.range, 1.0)
  assert almostEqual(s.mass, 1.0)

  # add and remove points
  # test special cases: SUM(weights)=0, RANGE(samples)=0, SUM(samples)=0
  s.append(point(1.0,-1.0))
  assert s.mean == -inf
  assert almostEqual(s.range, 1.0)
  assert almostEqual(s.mass, 0.0)

  '''
  _ave = s.mean
  s.mean = 1.0
  assert str(s.mean) == 'nan'
  assert str(s.range) == 'nan'
  assert almostEqual(s.mass, 0.0)
  s.normalize()
  assert str(s.mass) == 'nan'
  s.mean = _ave
  '''

  s.pop()
  s[0] = point(1.0,1.0)
  s[1] = point(-1.0,1.0)
  assert almostEqual(s.mean, 0.0)
  assert almostEqual(s.range, 2.0)
  assert almostEqual(s.mass, 2.0)
  s.normalize()
  s.mean = 1.0
  assert almostEqual(s.mean, 1.0)
  assert almostEqual(s.range, 2.0)
  assert almostEqual(s.mass, 1.0)

  s[0] = point(1.0,1.0)
  s[1] = point(1.0,1.0)
  assert almostEqual(s.mean, 1.0)
  assert almostEqual(s.range, 0.0)
  assert almostEqual(s.mass, 2.0)

  s.range = 1.0
  assert str(s.mean) == 'nan'
  assert str(s.range) == 'nan'
  assert almostEqual(s.mass, 2.0)
  return


def test_pack_unpack():
  x = [[1,2,3],[4,5],[6,7,8,9]]
  n = [len(i) for i in x]
  assert x == _unpack(_pack(x),n)
  return

def test_collection_behavior():
  from mystic.math import almostEqual
  from numpy import inf
  def f(x): return sum(x)  # a test function for expectation value

  # build three sets (x,y,z)
  sx = set([point(1.0,1.0), point(2.0,1.0), point(3.0,2.0)])
  sy = set([point(0.0,1.0), point(3.0,2.0)])
  sz = set([point(1.0,1.0), point(2.0,3.0)])
  #NOTE: for marc_surr(x,y,z), we must have x > 0.0
  sx.normalize()
  sy.normalize()
  sz.normalize()
  assert sx.mass == sy.mass == sz.mass == 1.0

  # build a collection
  c = collection([sx,sy,sz])
  xpos = c[0].positions
  ypos = c[1].positions
  zpos = c[2].positions
  xwts = c[0].weights
  ywts = c[1].weights
  zwts = c[2].weights
  if disp:
    print("x_positions: %s" % xpos)
    print("y_positions: %s" % ypos)
    print("z_positions: %s" % zpos)
    print("x_weights: %s" % xwts)
    print("y_weights: %s" % ywts)
    print("z_weights: %s" % zwts)
  assert xpos == sx.positions
  assert ypos == sy.positions
  assert zpos == sz.positions
  assert xwts == sx.weights
  assert ywts == sy.weights
  assert zwts == sz.weights
  tol = .2
  supp = c.support(tol)
  positions = c.positions
  weights = c.weights
  assert supp == [p for (p,w) in zip(positions,weights) if w > tol]
  if disp:
    print("support points:\n %s" % supp)
    print("npts: %s (i.e. %s)" % (c.npts, c.pts))
    print("weights: %s" % weights)
    print("positions: %s" % positions)
  assert c.npts == sx.npts * sy.npts * sz.npts
  assert len(weights) == len(positions) == c.npts
  assert sx.positions in c.pos
  assert sy.positions in c.pos
  assert sz.positions in c.pos
  assert sx.weights in c.wts
  assert sy.weights in c.wts
  assert sz.weights in c.wts

  c_exp = c.expect(f)
  if disp:
    print("mass: %s" % c.mass)
    print("expect: %s" % c_exp)
  assert c.mass == [sx.mass, sy.mass, sz.mass]
  assert c_exp == expectation(f, c.positions, c.weights)

 #print("center: %s" % c.center)
 #print("delta: %s" % c.delta)

  # change the positions in the collection
  points = [ list(i) for i in positions[::3] ]
  for i in range(len(points)):
    points[i][0] = 0.5
  positions[::3] = points
  c.positions = positions

  _xpos = c[0].positions
  _ypos = c[1].positions
  _zpos = c[2].positions
  _cexp = c.expect(f)
  if disp:
    print("x_positions: %s" % _xpos)
    print("y_positions: %s" % _ypos)
    print("z_positions: %s" % _zpos)
    print("expect: %s" % _cexp)
  assert _xpos == [0.5] + xpos[1:]
  assert _ypos == ypos
  assert _zpos == zpos
  assert _cexp < c_exp # due to _xpos[0] is 0.5 and less than 1.0

  _mean = 85.0
  _range = 0.25

  c.set_expect(_mean, f, npop=40, maxiter=200, tol=_range)
  _exp = c.expect(f)
  if disp:
    print("mean: %s" % _mean)
    print("range: %s" % _range)
    print("expect: %s" % _exp)
  assert almostEqual(_mean, _exp, tol=_mean*0.01)

  # a test function for probability of failure
  def g(x):
    if f(x) <= 0.0: return False
    return True
  pof = c.pof(g)
  spof = c.sampled_pof(g, npts=10000)
  if disp:
    print("pof: %s" % pof)
    print("sampled_pof: %s" % spof)
  assert almostEqual(pof, spof, tol=0.02)
  return

def test_flatten_unflatten():
  # build three sets (x,y,z)
  sx = set([point(1.0,1.0), point(2.0,1.0), point(3.0,2.0)])
  sy = set([point(0.0,1.0), point(3.0,2.0)])
  sz = set([point(1.0,1.0), point(2.0,3.0)])

  # build a collection
  c = collection([sx,sy,sz])

  # flatten and unflatten
  from mystic.math.discrete import flatten, unflatten
  d = unflatten(flatten(c), c.pts)

  # check if the same
  assert c.npts == d.npts
  assert c.weights == d.weights
  assert c.positions == d.positions

  # flatten() and load(...)
  e = collection().load(c.flatten(), c.pts)

  # check if the same
  assert c.npts == e.npts
  assert c.weights == e.weights
  assert c.positions == e.positions

  # decompose and compose
  from mystic.math.discrete import decompose, compose
  b = compose(*decompose(c))

  # check if the same
  assert c.npts == b.npts
  assert c.weights == b.weights
  assert c.positions == b.positions
  return


def test_min_max():
  # build a collection
  c = collection().load([.1, .4, 0, .2, .3, 1, 2, 6, 3, 5], (5,))

  # build a function
  f = lambda x: sum((i*i - 2*i) for i in x)

  # check min, max, etc
  assert c.minimum(f) == -1 == minimum(f, c.positions)
  assert c.maximum(f) == 24 == maximum(f, c.positions)
  assert c.ptp(f) == 25 == ptp(f, c.positions)
  assert c.ess_minimum(f) == -1 == ess_minimum(f, c.positions, c.weights)
  assert c.ess_maximum(f) == 15 == ess_maximum(f, c.positions, c.weights)
  assert c.ess_ptp(f) == 16 == ess_ptp(f, c.positions, c.weights)
  assert c.sampled_minimum(f, 100) == -1
  assert c.sampled_maximum(f, 100) == 15
  assert c.sampled_ptp(f, 100) == 16
  return


if __name__ == '__main__':
  test_calculate_methods(npts=2)
  test_set_behavior()
  test_pack_unpack()
  test_collection_behavior()
  test_flatten_unflatten()
  test_min_max()


# EOF
