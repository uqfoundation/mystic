#! /usr/bin/env python
"""
TESTS for Dirac measure data objects.
Includes point, dirac_measure, and product_measure classes.
"""
# Adapted from seesaw2d.py in branches/UQ/math/examples2/ 
# For usage example, see seesaw2d_inf_example.py .

from mystic.math.dirac_measure import point
from mystic.math.dirac_measure import dirac_measure as set
from mystic.math.dirac_measure import product_measure as collection
from mystic.math.samples import random_samples
from mystic.math.grid import samplepts

from mystic.math.measures import *
from mystic.math.measures import _pack, _unpack

def test_calculate_methods(npts=2):
  upper_bounds = [1.0]
  lower_bounds = [0.0]

  # -------------------------------------
  # generate initial coordinates, weights 
  # -------------------------------------
  # get a random distribution of points
  print "generate random points and weights"
  coordinates = samplepts(lower_bounds, upper_bounds, npts)
  D = [i[0] for i in coordinates]
  print "coords: %s" % D

  # calculate sample range
  R = spread(D)
  print "range: %s" % R

  # select weights randomly in [0,1], then normalize so sum(weights) = 1
  wts = random_samples([0],[1], npts)[0]
  weights = normalize(wts, 0.0, zsum=True)
  print "weights (when normalized to 0.0): %s" % weights
  weights = normalize(wts)
  print "weights (when normalized to 1.0): %s" % weights
  w = norm(weights)
  print "norm: %s" % w

  # calculate sample mean
  m = mean(D,weights)
  print "mean: %s" % m
  print ""

  # -------------------------------------
  # modify coordinates, maintaining mean & range 
  # -------------------------------------
  # get new random distribution
  print "modify coords, maintaining mean and range"
  coordinates = samplepts(lower_bounds, upper_bounds, npts)
  D = [i[0] for i in coordinates]

  # impose a range and mean on the points
  D = impose_spread(R, D, weights)
  D = impose_mean(m, D, weights)

  # print results
  print "coords: %s" % D
  R = spread(D)
  print "range: %s" % R
  m = mean(D, weights)
  print "mean: %s" % m
  print ""

  # -------------------------------------
  # modify weights, maintaining mean & norm
  # -------------------------------------
  # select weights randomly in [0,1]
  print "modify weights, maintaining mean and range"
  wts = random_samples([0],[1], npts)[0]

  # print intermediate results
 #print "weights: %s" % wts
 #sm = mean(D, wts)
 #print "tmp mean: %s" % sm
 #print ""

  # impose mean and weight norm on the points
  D = impose_mean(m, D, wts)
  DD, weights = impose_weight_norm(D, wts)

  # print results
  print "weights: %s" % weights
  w = norm(weights)
  print "norm: %s" % w
  print "coords: %s" % DD
  R = spread(DD)
  print "range: %s" % R
  sm = mean(DD, weights)
  print "mean: %s" % sm
  sv = variance(DD, weights)
  print "var: %s" % sv

  # -------------------------------------
  # modify variance, maintaining mean
  # -------------------------------------
  print "\nmodify variance, maintaining mean"
  DD = impose_variance(R, DD, weights)
  sm = mean(DD, weights)
  print "mean: %s" % sm
  sv = variance(DD, weights)
  print "var: %s" % sv
  

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
  s.mean = 1.0
  print s.mean  # nan
  print s.range # nan
  print s.mass  # 0.0
  s.normalize()
  print s.mass  # nan
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
  '''
  s.range = 1.0
  print s.mean  # nan
  print s.range # nan
  print s.mass  # 2.0
  s.normalize()
  print s.mass  # 1.0
  '''

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

  # build a collection
  c = collection([sx,sy,sz])
  print "x_coords: %s" % c[0].coords
  print "y_coords: %s" % c[1].coords
  print "z_coords: %s" % c[2].coords
  print "x_weights: %s" % c[0].weights
  print "y_weights: %s" % c[1].weights
  print "z_weights: %s" % c[2].weights
  print "randomly selected support:\n %s" % c.support(10)

  print "npts: %s (i.e. %s)" % (c.npts, c.pts)
  print "weights: %s" % c.weights
  coords = c.coords
  print "coords: %s" % coords

  print "mass: %s" % c.mass
  print "expect: %s" % c.get_expect(f)

 #print "center: %s" % c.center
 #print "delta: %s" % c.delta

  # change the coords in the collection
  coords[::3]
  points = [ list(i) for i in coords[::3] ]
  for i in range(len(points)):
    points[i][0] = 0.5

  coords[::3] = points
  c.coords = coords
  print "x_coords: %s" % c[0].coords
  print "y_coords: %s" % c[1].coords
  print "z_coords: %s" % c[2].coords
  print "expect: %s" % c.get_expect(f)

  _mean = 85.0
  _range = 0.25

  c.set_expect((_mean,_range), f)
  print "mean: %s" % _mean
  print "range: %s" % _range
  print "expect: %s" % c.get_expect(f)

  # a test function for probability of failure
  def g(x):
    if f(x) <= 0.0: return False
    return True
  print "pof: %s" % c.pof(g)
  print "sampled_pof: %s" % c.sampled_pof(g, npts=10000)
  return

def test_flatten_unflatten():
  # build three sets (x,y,z)
  sx = set([point(1.0,1.0), point(2.0,1.0), point(3.0,2.0)])
  sy = set([point(0.0,1.0), point(3.0,2.0)])
  sz = set([point(1.0,1.0), point(2.0,3.0)])

  # build a collection
  c = collection([sx,sy,sz])

  # flatten and unflatten
  from mystic.math.dirac_measure import flatten, unflatten
  d = unflatten(flatten(c), c.pts)

  # check if the same
  assert c.npts == d.npts
  assert c.weights == d.weights
  assert c.coords == d.coords

  # flatten() and load(...)
  e = collection()
  e.load(c.flatten(), c.pts)

  # check if the same
  assert c.npts == e.npts
  assert c.weights == e.weights
  assert c.coords == e.coords

  # decompose and compose
  from mystic.math.dirac_measure import decompose, compose
  b = compose(*decompose(c))

  # check if the same
  assert c.npts == b.npts
  assert c.weights == b.weights
  assert c.coords == b.coords
  return


if __name__ == '__main__':
  #test_calculate_methods(npts=2)
  #test_set_behavior()
  #test_pack_unpack()
  test_collection_behavior()
  #test_flatten_unflatten()
  pass


# EOF
