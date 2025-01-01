#!/usr/bin/env python
# 
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2024-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
noisy emulator:
* input parameters (a_alpha, c_alpha, a_steel)
  with solved values at:
    [2.9306538, 4.6817646, 3.6026807]
  measured error of:
    [1e-4, 1e-4, 1e-4]
  bounds at:
    [(0.95 * i, 1.05 * i) for i in solved]
  yielding a minimum at:
    a_beta = 3.233392
  with a measured error of:
    a_beta_error = 1e-4
  and gaussian noise on the input parameters of:
    sigma = 0.001
'''
from itertools import repeat
from mystic.monitors import LoggingMonitor
from emulators import cost3, x3, bounds3
from noisy import noisy

try: # parallel maps
    from pathos.maps import Map
    from pathos.pools import ProcessPool, ThreadPool, SerialPool
    cmap = Map(SerialPool) #ProcessPool
except ImportError:
    cmap = map

# use a logging monitor to track cost function evaluations
monitor3 = LoggingMonitor(1, filename='truth.txt', label='truth')

# generate an objective with gaussian noise on the inputs
noisycost3 = lambda x: cost3(noisy(x, sigma=.001))

# generate a tracked cost function
def trackcost3(x):
  fx = noisycost3(x)
  monitor3(x, fx)
  return fx

# average the cost over the number of cost evaluations per objective call
avecost3 = lambda x, n=1: sum(cmap(trackcost3, repeat(x, n)))/n

# set the number of cost evaluations per objective call
ave10cost3 = lambda x: avecost3(x, n=10)
