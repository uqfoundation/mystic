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
from mystic.samplers import LatticeSampler
from mystic.monitors import LoggingMonitor
from ouq_models import WrapModel, InterpModel
from emulators import cost4 as cost, x4 as target, bounds4 as bounds
#from mystic.cache.archive import file_archive, read as get_db

# prepare truth (i.e. an 'expensive' model)
nx = 4; ny = None
#archive = get_db('truth.db', type=file_archive)
truth = WrapModel('truth', cost, nx=nx, ny=ny, cached=False)#archive)

# remove any prior cached evaluations of truth
import shutil
if os.path.exists("truth"): shutil.rmtree("truth")
if os.path.exists("error.txt"): os.remove("error.txt")
if os.path.exists("log.txt"): os.remove("log.txt")

try: # parallel maps
    from pathos.maps import Map
    from pathos.pools import ProcessPool, ThreadPool, SerialPool
    pmap = Map(SerialPool) #ProcessPool
except ImportError:
    pmap = None

# generate a training dataset by sampling truth
data = truth.sample(bounds, pts=[2, 1, 1, 1], pmap=map)#, archive=archive)

# shutdown mapper
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()

# create an inexpensive surrogate for truth
surrogate = InterpModel("surrogate", nx=nx, ny=ny, data=truth, smooth=0.0,
                        noise=0.0, method="thin_plate", extrap=False)

# iterate until error (of candidate minimum) < 1e-3
N = 4
import numpy as np
tracker = LoggingMonitor(1, filename='error.txt', label='error')
from mystic.abstract_solver import AbstractSolver
from mystic.termination import VTR
loop = AbstractSolver(nx)
loop.SetTermination(VTR(1e-3)) #XXX: VTRCOG, TimeLimits, etc?
loop.SetEvaluationLimits(maxiter=500)
loop.SetGenerationMonitor(tracker)
while not loop.Terminated():

    # fit the surrogate to data in truth database
    surrogate.fit(data=data)

    # find the first-order critical points of the surrogate
    surr_ = lambda x: surrogate(x, axis=None)
    s = LatticeSampler(bounds, surr_, npts=N)
    s.sample_until(terminated=all)
    xdata = [list(i) for i in s._sampler._all_bestSolution]
    ysurr = s._sampler._all_bestEnergy

    _surr = lambda x: -surrogate(x, axis=None)
    s_ = LatticeSampler(bounds, _surr, npts=N)
    s_.sample_until(terminated=all)
    xdata = xdata + [list(i) for i in s_._sampler._all_bestSolution]
    ysurr = ysurr + [-i for i in s_._sampler._all_bestEnergy]

    # evaluate truth at the same input as the surrogate critical points
    ytrue = list(map(truth, xdata))
    # add most recent candidate extrema to truth database
    data.load(xdata, ytrue)

    # compute absolute error between truth and surrogate at candidate extrema
    idx = np.argmin(ytrue)
    error = abs(np.array(ytrue) - ysurr)
    #print("error@mins: %s" % error[:N])
    #print("error@maxs: %s" % error[-N:])
    print("ave: %s; max: %s" % (error.mean(), error.max()))
    print("ave mins: %s; at min: %s" % (error[:N].mean(), error[idx]))
    print("evaluations of truth: %s" % len(data))

    # save to tracker if less than current best
    ysave = error # track error when learning surrogate
    if len(tracker) and tracker.y[-1] < ysave[idx]:
        tracker(*tracker[-1])
    else: tracker(xdata[idx], ysave[idx])

# get the results at the best parameters from the truth database
xbest = tracker[-1][0]
#ybest = archive[tuple(xbest)]
ybest = data[data.coords.index(xbest)].value

# print the best parameters
print(f"Best solution is {xbest} with Rwp {ybest}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, xbest)]
print(f"Ratios of best to reference solution: {ratios}")
