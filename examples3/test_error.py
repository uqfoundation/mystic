#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
calculate error for actively learned/interpolated models

Test function is y = F(x), where:
  y0 = x0 + x1 * | x2 * x3**2 - (x4 / x1)**2 |**.5
  y1 = x0 - x1 * | x2 * x3**2 + (x4 / x1)**2 |**.5
  y2 = x0 - | x1 * x2 * x3 - x4 |

toy = lambda x: F(x)[0]
golden = lambda x: toy(x + .001) - .001
truth = lambda x: G(toy(G(x + .001, .01)) -.001, .01)
G(mu, sigma) is a Gaussian with mean = mu and std = sigma

1) Sample 10 pts in [0,10] with truth.
   Find graphical distance between truth and sampled pts.
2) Sample golden with 4 solvers, then interpolate to produce a surrogate.
   Find pointwise distance (golden(x) - surrogate(x))**2.
3) Sample golden with 4 more solvers, then interpolate an updated surrogate.
   Find pointwise distance (golden(x) - surrogate(x))**2.
4) Train a MLP Regressor on the sampled data.
   Find pointwise distance (golden(x) - surrogate(x))**2.

Creates 'golden' and 'truth' databases of stored evaluations.
'''
from ouq_models import *


if __name__ == '__main__':

    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None

    # build a model representing 'truth' (one deterministic, and one not)
    truth = dict(model=toy, nx=nx, ny=ny, mu=.001, zmu=-.001)#, uid=True)
    golden = NoisyModel('golden', cached=True, sigma=0, zsigma=0, **truth)
    truth = NoisyModel('truth', sigma=.01, zsigma=.01, **truth)

    # generate data (DB) of sampled 'truth'
    deterministic = False
    Gx = golden if deterministic else truth
    bounds = [(0,10)]*nx
    data = Gx.sample(bounds, pts=10) #FIXME: activate cache w/o calling sample?
    print("size of data: %s" % len(data.coords))
    #print(len(data.values))

    # get graphical distance (for 'truth')
    error = Gx.distance(data, axis=None)
    import numpy as np
    print('total error: %s' % np.sum(error))
    print('max error: %s' % np.max(error, axis=-1))

    # calculate model error for 'golden'
    data = golden.sample(bounds, pts=-4)
    #print('truth: %s' % str(golden([1,2,3,4,5])))
    estimate = dict(nx=nx, ny=ny, data=golden, noise=0, smooth=0)
    surrogate = InterpModel('surrogate', method='thin_plate', **estimate)
    print('estimate: %s' % str(surrogate([1,2,3,4,5])))
    print('truth: %s' % str(golden([1,2,3,4,5])))
    error = dict(model=golden, surrogate=surrogate)
    misfit = ErrorModel('misfit', **error)
    print('error: %s' % str(misfit([1,2,3,4,5])))

    #'''
    # sample more data, refit, and recalculate error
    print('resampling and refitting surrogate')
    data = golden.sample(bounds, pts=-4)
    surrogate.fit()
    print('estimate: %s' % str(surrogate([1,2,3,4,5])))
    print('error: %s' % str(misfit([1,2,3,4,5])))
    #'''

    #'''
    # fit a learned model using the sampled data, and calculate error
    print('fitting an estimator with machine learning')
    estimate = dict(nx=nx, ny=ny, data=golden)
    mlarg = dict(hidden_layer_sizes=(100,75,50,25),  max_iter=1000, n_iter_no_change=5, solver='lbfgs')
    import sklearn.neural_network as nn
    estimator = nn.MLPRegressor(**mlarg)
    learned = LearnedModel('learned', estimator=estimator, **estimate)
    print('estimate: %s' % str(learned([1,2,3,4,5])))
    mlerror = dict(model=golden, surrogate=learned)
    error = ErrorModel('error', **mlerror)
    print('error: %s' % str(error([1,2,3,4,5])))
    #'''

