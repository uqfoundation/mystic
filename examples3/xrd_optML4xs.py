#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# adapted from XRD example at: https://spotlight.readthedocs.io/
"""
optimization of 4-input cost function using online learning of a surrogate
"""
import os
from mystic.math.legacydata import dataset, datapoint
from mystic.samplers import LatticeSampler
from ouq_models import WrapModel, InterpModel
from emulators import cost4 as cost, x4 as target, bounds4 as bounds
from mystic.cache.archive import file_archive, read as get_db

# remove any prior cached evaluations of truth (i.e. an 'expensive' model)
if os.path.exists("truth"):
    import shutil
    shutil.rmtree("truth")

# remove any prior cached evaluations of surrogate
if os.path.exists("surrogate"):
    import shutil
    shutil.rmtree("surrogate")

try: # parallel maps
    from pathos.maps import Map
    from pathos.pools import ProcessPool, ThreadPool, SerialPool
    pmap = Map(SerialPool) #ProcessPool
except ImportError:
    pmap = None

# generate a training dataset by sampling truth
#archive = get_db('truth.db', type=file_archive)
truth = WrapModel('truth', cost, nx=4, ny=None, cached=False)#archive)
data = truth.sample(bounds, pts=[2, 1, 1, 1], pmap=map)#, archive=archive)

# shutdown mapper
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()

# create an inexpensive surrogate for truth
surrogate = InterpModel("surrogate", nx=4, ny=None, data=truth, smooth=0.0,
                        noise=0.0, method="thin_plate", extrap=False)

# iterate until mean error (of candidate minima) < 1e-3
N = 4
import numpy as np
error = np.array([float('inf')])
while error[:N].mean() > 1e-3:

    # fit a new surrogate to data in truth database
    get_db('surrogate').clear()
    surrogate.fit(data=data)

    # find the first-order critical points of the surrogate
    surr = surrogate.sample(bounds, pts='.%s' % N)

    # evaluate truth at the same input as the surrogate critical points
    ytrue = list(map(truth, surr.coords))

    # compute absolute error between truth and surrogate at candidate extrema
    error = abs(np.array(ytrue) - surr.values)
    print("error@mins: %s" % error[:N])
    print("error@maxs: %s" % error[-N:])
    print("stop: %s; ave: %s; max: %s" % (error[:N].mean(), error.mean(), error.max()))

    # add most recent candidate extrema to truth database
    data.load(surr.coords, ytrue)

# get minimum of last batch of results
idx = np.argmin(ytrue)
xnew = surr.coords[idx]
ynew = ytrue[idx]

# print the best parameters
print(f"Best solution is {xnew} with Rwp {ynew}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, xnew)]
print(f"Ratios of best to reference solution: {ratios}")
