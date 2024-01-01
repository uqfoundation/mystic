#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# adapted from XRD example at: https://spotlight.readthedocs.io/
"""
optimization of 4-input cost function using online learning of a surrogate
"""
import os
from mystic.samplers import SparsitySampler
from mystic.monitors import LoggingMonitor
from mystic.solvers import PowellDirectionalSolver
from mystic.termination import NormalizedChangeOverGeneration as NCOG
from ouq_models import WrapModel, InterpModel
from emulators import cost4 as cost, x4 as target, bounds4 as bounds
#from mystic.cache.archive import file_archive, read as get_db

# remove any prior cached evaluations of truth (i.e. an 'expensive' model)
if os.path.exists("truth"):
    import shutil
    shutil.rmtree("truth")

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

# iterate until error (of candidate minimum) < 1e-3
N = 4
import mystic._counter as it
counter = it.Counter()
tracker = LoggingMonitor(1, filename='error.txt', label='error')
import numpy as np
error = np.array([float('inf')]); idx = 0
#while error[:N].mean() > 1e-3:
while error[idx] > 1e-3:

    # fit the surrogate to data in truth database
    surrogate.fit(data=data)

    # find the first-order critical points of the surrogate
    s = SparsitySampler(bounds, lambda x: surrogate(x, axis=None), npts=N,
                        maxiter=8000, maxfun=1e6, id=counter.count(N),
                        stepmon=LoggingMonitor(1, label='output'),
                        solver=PowellDirectionalSolver,
                        termination=NCOG(1e-6, 10))
    s.sample_until(terminated=all)
    xdata = [list(i) for i in s._sampler._all_bestSolution]
    ysurr = s._sampler._all_bestEnergy

    # evaluate truth at the same input as the surrogate critical points
    ytrue = list(map(truth, xdata))

    # compute absolute error between truth and surrogate at candidate extrema
    idx = np.argmin(ytrue)
    error = abs(np.array(ytrue) - ysurr)
    print("truth: %s @ %s" % (ytrue[idx], xdata[idx]))
    print("candidate: %s; error: %s" % (ysurr[idx], error[idx]))
    print("error ave: %s; error max: %s" % (error.mean(), error.max()))
    print("evaluations of truth: %s" % len(data))
    tracker(xdata[idx], error[idx])

    # add most recent candidate extrema to truth database
    data.load(xdata, ytrue)

# get minimum of last batch of results
xnew = xdata[idx]
ynew = ytrue[idx]

# print the best parameters
print(f"Best solution is {xnew} with Rwp {ynew}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, xnew)]
print(f"Ratios of best to reference solution: {ratios}")
