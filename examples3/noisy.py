#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
functions to add noise to function inputs or outputs
"""

import numpy as np

def _noisy(x, dist=None, op=None):
  """make x noisy, by sampling from dist; returns op(x, dist.rvs(x.shape))

  x: parameters (list or array)
  dist: mystic.math.Distribution object
  op: numpy operator (e.g. np.add or np.prod)

  For example:
    >>> from numpy.random import normal
    >>> import mystic
    >>> x = [1,2,3]
    >>> rng = mystic.random_state('numpy.random', new=True, seed=123)
    >>> dist = mystic.math.Distribution(normal, 0, .1, rng=rng)
    >>> noisy(x, dist)
    [0.8914369396699439, 2.0997345446583586, 3.0282978498051993]
  """
  if op is None: op = np.add
  _type = type(x) if type(x) is not np.ndarray else None
  x = np.asarray(x)
  if dist is None:
    from mystic.math import Distribution
    dist = Distribution()
  dx = dist.rvs(x.shape)
  return op(x,dx) if _type is None else _type(op(x,dx))


def noisy(x, mu=0, sigma=1, seed='!'):
  """make x noisy, by adding noise from a normal distribution

  x: parameters (list or array)
  mu: distribution mean value
  sigma: distribution standard deviation
  seed: random seed [default: '!', do not reseed the RNG]
  """ # new: if True, generate a new RNG object [default: use the global RNG]
  from numpy.random import normal
  import mystic
  rng = mystic.random_state('numpy.random', new=True, seed=seed)
  dist = mystic.math.Distribution(normal, mu, sigma, rng=rng)
  return _noisy(x, dist=dist)

