#! /usr/bin/env python
"""
test imposing the expectation for a function f by optimization
"""
debug = False

from math import pi, cos, tanh
import random

def ballistic_limit(h,a):
  """calculate ballistic limit

  Inputs:
    - h = thickness
    - a = obliquity

  Outputs:
    - v_bl = velocity (ballistic limit)
"""
 #assumes h,a have been scaled:
 #h = x[0] * 25.4 * 1e-3
 #a = x[1] * pi/180.0
  Ho = 0.5794
  s = 1.4004
  n = 0.4482
  return Ho * ( h / cos(a)**n )**s

def marc_surr(x):
  """calculate perforation area using a tanh-based model surrogate

  Inputs:
    - x = [thickness, obliquity, speed]

  Outputs:
    - A = perforation area
"""
# h = thickness = [60,105]
# a = obliquity = [0,30]
# v = speed = [2.1,2.8]
  h = x[0] * 25.4 * 1e-3
  a = x[1] * pi/180.0
  v = x[2]

  K = 10.3963
  p = 0.4757
  u = 1.0275
  m = 0.4682
  Dp = 1.778

  # compare to ballistic limit
  v_bl = ballistic_limit(h,a)
  if v < v_bl:
    return 0.0

  return K * (h/Dp)**p * (cos(a))**u * (tanh((v/v_bl)-1))**m


if __name__ == '__main__':
  G = marc_surr  #XXX: uses the above-provided test function
  function_name = G.__name__

  _mean = 06.0   #NOTE: SET THE mean HERE!
  _range = 00.5  #NOTE: SET THE range HERE!
  nx = 3  #NOTE: SET THE NUMBER OF 'h' POINTS HERE!
  ny = 3  #NOTE: SET THE NUMBER OF 'a' POINTS HERE!
  nz = 3  #NOTE: SET THE NUMBER OF 'v' POINTS HERE!

  h_lower = [60.0];  a_lower = [0.0];  v_lower = [2.1]
  h_upper = [105.0]; a_upper = [30.0]; v_upper = [2.8]

  lower_bounds = (nx * h_lower) + (ny * a_lower) + (nz * v_lower)
  upper_bounds = (nx * h_upper) + (ny * a_upper) + (nz * v_upper)
  bounds = (lower_bounds,upper_bounds)

  print " model: f(x) = %s(x)" % function_name
  print " mean: %s" % _mean
  print " range: %s" % _range
  print "..............\n"

  if debug:
    param_string = "["
    for i in range(nx):
      param_string += "'x%s', " % str(i+1)
    for i in range(ny):
      param_string += "'y%s', " % str(i+1)
    for i in range(nz):
      param_string += "'z%s', " % str(i+1)
    param_string = param_string[:-2] + "]"

    print " parameters: %s" % param_string
    print " lower bounds: %s" % lower_bounds
    print " upper bounds: %s" % upper_bounds
  # print " ..."

  wx = [1.0 / float(nx)] * nx
  wy = [1.0 / float(ny)] * ny
  wz = [1.0 / float(nz)] * nz

  from mystic.math.measures import _pack
  wts = _pack([wx,wy,wz])
  weights = [i[0]*i[1]*i[2] for i in wts]

  from mystic.math.measures import expectation, impose_expectation
  samples = impose_expectation((_mean,_range), G, (nx,ny,nz), bounds, weights)

  if debug:
    from numpy import array
    # rv = [xi]*nx + [yi]*ny + [zi]*nz
    print "solved: [x]\n%s" % array(samples[:nx])
    print "solved: [y]\n%s" % array(samples[nx:nx+ny])
    print "solved: [z]\n%s" % array(samples[nx+ny:])
    #print "solved: %s" % samples

  Ex = expectation(G, samples, weights)
  print "expect: %s" % Ex
  print "cost = (E[G] - m)^2: %s" % (Ex - _mean)**2

# EOF
