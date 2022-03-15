#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2015 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

"""Original matlab code:

function A=marc_surr(x)
h=x(1)*25.4*10^(-3);
a=x(2)*pi/180;
v=x(3);
Ho=0.5794;
s=1.4004;
n=0.4482;
K=10.3963;
p=0.4757;
u=1.0275;
m=0.4682;
Dp=1.778;

v_bl=Ho*(h/(cos(a))^(n))^s;

if v<v_bl
  A=0
else
  A=K*((h/Dp)^p)*((cos(a))^u)*(tanh((v/v_bl)-1))^m;
end
"""

### NOTES ###
# h = thickness = [60,105]
# a = obliquity = [0,30]
# v = speed = [2.1,2.8]

# explore the cuboid (h,a,v), with
# subdivisions at h=100, a=20, v=2.2
# due to ballistic limit: v(h=100,a=20) = 2.22,
# perforation in this region should be zero.
# NOTE: 'failure' is A < = t
# 
# Calculate for each of the 8 subcuboids:
#  * probability mass, i.e. the product of the normalized side-lengths,
#    since we're taking h, a and v to be uniformly distributed in their
#    intervals
#  * McDiarmid diameter of the perforation area A when restricted to that
#    cuboid or subcuboid
#  * the mean value of the perforation area A on each (sub)cuboid

from math import pi, cos, tanh

def ballistic_limit(h,a):
  """calculate ballistic limit

  Inputs:
    - h = thickness in (unknown) units
    - a = obliquity in (unknown) units

  Outputs:
    - v_bl = velocity (ballistic limit) in (unknown) units
"""
 #h = x[0] * 25.4 * 1e-3
 #a = x[1] * pi/180.0
  Ho = 0.5794
  s = 1.4004
  n = 0.4482
  return Ho * ( h / cos(a)**n )**s


def marc_surr(x):
  """calculate perforation area using a tanh-based model surrogate

  Inputs:
    - x = [thickness, obliquity, speed] in (unknown) units

  Outputs:
    - A = performation area in (unknown) units
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
    return 0

  return K * (h/Dp)**p * (cos(a))**u * (tanh((v/v_bl)-1))**m

# EOF
