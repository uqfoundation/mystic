#!/usr/bin/env python
# 
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2022-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
mystic.models where the cost and solved inputs have been shifted
to emulate results from a more expensive model

base models:
* rosen: n-D function with minimum at x_i = 1, with i in [1,n]
* sphere: n-D function with minimum at x_i = 0, with i in [1,n]

emulator #1:
* input parameters (a_alpha, c_alpha, a_steel, a_beta)
  with solved values at:
    [2.9306538, 4.6817646, 3.6026807, 3.233392]
  measured error of:
    [1e-4, 1e-4, 1e-4, 1e-4]
  bounds at:
    [(0.95 * i, 1.05 * i) for i in solved]
  yielding a minimum at:
    wR = 0.13658616

emulator #2:
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

emulator #3:
* input (dist_spec, center_x, center_y, detc_2theta, detc_phiDA, detc_omegaDN):
  with solved values at:
    [65.0458, 2.540605, -0.28430015, -27.21818, -0.44651195, 1.1400392]
  measured error of:
    [0.02124893, 0.028312204, 0.10003452, 0.011049773, 1.0831603, 2.0910518]
  bounds at:
    [(10,1000),(-1,10000),(-10000,10000),(-180,180),(-180,180),(-10,10)]
  yielding a minimum at:
    wR = 0.13658616
'''
import mystic.models as mm
import numpy as np

# 'solved' inputs (to use as 'xo')
x6 = [65.0458, 2.540605, -0.28430015, -27.21818, -0.44651195, 1.1400392]
x4 = [2.9306538, 4.6817646, 3.6026807, 3.233392]
x3 = x4[:-1]

# 'solved' minimum cost (to use as 'yo')
wR = 0.13658616
a_beta = x4[-1]

# error and bounds information
error6 = [0.02124893,0.028312204,0.10003452,0.011049773,1.0831603,2.0910518]
error4 = [1e-4, 1e-4, 1e-4, 1e-4]
error3 = error4[:-1]
bounds6 = [(10,1000),(-1,10000),(-10000,10000),(-180,180),(-180,180),(-10,10)]
bounds4 = [tuple(sorted((0.95 * i, 1.05 * i))) for i in x4]
bounds3 = bounds4[:-1]
a_beta_error = error4[-1]


# to use as objective, either use ExtraArgs,
# or fix the args: cost = lambda x: rosen(x, xo=x4, yo=wR)
def rosen(x, xo=None, yo=None):
    "rosen shifted to have minimum cost at yo and solved x at xo"
    xo = x if xo is None else (np.ones(len(x)) + x - xo)
    return mm.rosen(xo) + (0 if yo is None else yo)

def sphere(x, xo=None, yo=None):
    "sphere shifted to have minimum cost at yo and solved x at xo"
    xo = x if xo is None else (np.asarray(x) - xo)
    return mm.sphere(xo) + (0 if yo is None else yo)

# emulator instances
cost3 = lambda x: rosen(x, xo=x3, yo=a_beta)
cost4 = lambda x: rosen(x, xo=x4, yo=wR)
cost6 = lambda x: sphere(x, xo=x6, yo=wR)
