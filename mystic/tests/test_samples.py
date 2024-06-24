#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

try:
  from pathos.pools import ProcessPool
  p = ProcessPool()
  pmap = p.map
except ImportError:
  pmap = None

import mystic as my
import mystic.math.integrate as mi
import mystic.math.samples as ms
import mystic.models as mm
def inside(x):
  import mystic.models as mm
  return mm.sphere(x) < 1
def positive(x):
  import mystic.models as mm
  return mm.sphere(x) >= 0
lb,ub = [0,0,0],[1,1,1]
kwd = dict(tol=None, rel=.05)
_kwd = dict(tol=.05, rel=None)

# lb,ub must be iterable
miss,hits = ms.sample(inside, lb, ub)
zero,npts = ms.sample(inside, lb[:1], ub[:1]) # 1/3 bound --> 1/3 result
assert zero == 0
assert miss > zero
assert npts == sum((miss,hits))
if pmap:
  n,y = ms.sample(inside, lb, ub, map=pmap)
  assert my.math.approx_equal(n, miss, **kwd)
  assert my.math.approx_equal(y, hits, **kwd)

ans = ms.sampled_mean(mm.sphere, lb, ub)
assert my.math.approx_equal(mi.integrated_mean(mm.sphere, lb, ub), ans, **kwd)
assert my.math.approx_equal(ms.sampled_mean(mm.sphere, lb[:1], ub[:1]) * 3, ans, **kwd)
if pmap:
  a = ms.sampled_mean(mm.sphere, lb, ub, map=pmap)
  assert my.math.approx_equal(a, ans, **kwd)

ans = ms.sampled_variance(mm.sphere, lb, ub)
assert my.math.approx_equal(mi.integrated_variance(mm.sphere, lb, ub), ans, **kwd)
assert my.math.approx_equal(ms.sampled_variance(mm.sphere, lb[:1], ub[:1]) * 3, ans, **kwd)
if pmap:
  a = ms.sampled_variance(mm.sphere, lb, ub, map=pmap)
  assert my.math.approx_equal(a, ans, **kwd)

ans = ms.sampled_minimum(mm.sphere, lb, ub, npts=50000)
assert my.math.approx_equal(mm.sphere(my.solvers.fmin(mm.sphere, x0=[.5,.5,.5], bounds=list(zip(lb, ub)), disp=False)), ans, **_kwd)
assert my.math.approx_equal(ms.sampled_minimum(mm.sphere, lb[:1], ub[:1]) * 3, ans, **_kwd)
if pmap:
  a = ms.sampled_minimum(mm.sphere, lb, ub, npts=50000, map=pmap)
  assert my.math.approx_equal(a, ans, **_kwd)

ans = ms.sampled_maximum(mm.sphere, lb[:2], ub[:2], npts=50000)
assert my.math.approx_equal(ms.sampled_maximum(mm.sphere, lb[:1], ub[:1]) * 2, ans, **_kwd)
if pmap:
  a = ms.sampled_maximum(mm.sphere, lb[:2], ub[:2], npts=50000, map=pmap)
  assert my.math.approx_equal(a, ans, **_kwd)

ans = ms.sampled_ptp(mm.sphere, lb[:2], ub[:2], npts=50000)
assert my.math.approx_equal(ms.sampled_ptp(mm.sphere, lb[:1], ub[:1]) * 2, ans, **_kwd)
if pmap:
  a = ms.sampled_ptp(mm.sphere, lb[:2], ub[:2], npts=50000, map=pmap)
  assert my.math.approx_equal(a, ans, **_kwd)

ans = ms.sampled_pof(positive, lb, ub)
assert my.math.approx_equal(ms.sampled_pof(positive, lb[:1], ub[:1]), ans, **kwd)
if pmap:
  a = ms.sampled_pof(positive, lb, ub, map=pmap)
  assert my.math.approx_equal(a, ans, **kwd)

ans = ms.sampled_pof(inside, lb, ub)
assert my.math.approx_equal(0.475, ans, **kwd)
assert my.math.approx_equal(0.000, ms.sampled_pof(inside, lb[:1], ub[:1]), **kwd)
if pmap:
  a = ms.sampled_pof(inside, lb, ub, map=pmap)
  assert my.math.approx_equal(a, ans, **kwd)

pts = ms._random_samples(lb, ub, 10000)
_pts = ms._random_samples(lb[:1], ub[:1], 10000)

assert 0 < ms._minimum_given_samples(mm.sphere, pts) < 0.1
assert 0 < ms._minimum_given_samples(mm.sphere, _pts) < 0.0001
if pmap:
  assert my.math.approx_equal(ms._minimum_given_samples(mm.sphere, pts), ms._minimum_given_samples(mm.sphere, pts, map=pmap), **kwd)

ans = ms.sampled_mean(mm.sphere, lb, ub)
assert my.math.approx_equal(ms._expectation_given_samples(mm.sphere, pts), ans, **kwd)
assert my.math.approx_equal(ms._expectation_given_samples(mm.sphere, _pts) * 3, ans, **kwd)
if pmap:
  assert my.math.approx_equal(ms._expectation_given_samples(mm.sphere, pts), ms._expectation_given_samples(mm.sphere, pts, map=pmap), **kwd)

ans = ms.sampled_variance(mm.sphere, lb, ub)
assert my.math.approx_equal(ms._variance_given_samples(mm.sphere, pts), ans, **kwd)
assert my.math.approx_equal(ms._variance_given_samples(mm.sphere, _pts) * 3, ans, **kwd)
if pmap:
  assert my.math.approx_equal(ms._variance_given_samples(mm.sphere, pts), ms._variance_given_samples(mm.sphere, pts, map=pmap), **kwd)

assert 2.0 < ms._maximum_given_samples(mm.sphere, pts) < 3.5
assert 0.9 < ms._maximum_given_samples(mm.sphere, _pts) < 1.1
if pmap:
  assert my.math.approx_equal(ms._maximum_given_samples(mm.sphere, pts), ms._maximum_given_samples(mm.sphere, pts, map=pmap), **kwd)

assert ms._maximum_given_samples(mm.sphere, pts) - ms._minimum_given_samples(mm.sphere, pts) == ms._ptp_given_samples(mm.sphere, pts)
assert ms._maximum_given_samples(mm.sphere, _pts) - ms._minimum_given_samples(mm.sphere, _pts) == ms._ptp_given_samples(mm.sphere, _pts)
if pmap:
  assert my.math.approx_equal(ms._ptp_given_samples(mm.sphere, pts), ms._ptp_given_samples(mm.sphere, pts, map=pmap), **kwd)

assert 0.47 <= ms._pof_given_samples(inside, pts) < 0.48
assert 0.00 <= ms._pof_given_samples(inside, _pts) < 0.01
if pmap:
  assert my.math.approx_equal(ms._pof_given_samples(positive, pts), ms._pof_given_samples(positive, pts, map=pmap), **kwd)

assert my.math.approx_equal(ms.sampled_prob(pts, lb, ub), 1, **kwd)
assert my.math.approx_equal(ms.sampled_prob(_pts, lb[:1], ub[:1]), 1, **kwd)
assert my.math.approx_equal(ms.sampled_prob(pts, [.25,.25,.25], ub), 0.42, **kwd)
assert my.math.approx_equal(ms.sampled_pts(pts, [.25,.25,.25], ub), 4200, **kwd)
if pmap:
  assert my.math.approx_equal(ms.sampled_prob(pts, [.25,.25,.25], ub, map=pmap), 0.42, **kwd)
  assert my.math.approx_equal(ms.sampled_pts(pts, [.25,.25,.25], ub, map=pmap), 4200, **kwd)

  # shutdown
  p.close()
  p.join()

