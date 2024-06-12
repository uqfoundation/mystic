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
from mystic.solvers import diffev2
from mystic.monitors import LoggingMonitor
from mystic.math.legacydata import datapoint
from mystic.cache.archive import file_archive, read as get_db
from ouq_models import WrapModel, InterpModel
from emulators import cost4 as cost, x4 as target, bounds4 as bounds

# remove any prior cached evaluations of truth (i.e. an 'expensive' model)
if os.path.exists("truth.db"):
    os.remove("truth.db")

try: # parallel maps
    from pathos.maps import Map
    from pathos.pools import ProcessPool, ThreadPool, SerialPool
    pmap = Map(SerialPool) #ProcessPool
except ImportError:
    pmap = None

# generate a dataset by sampling truth
archive = get_db('truth.db', type=file_archive)
truth = WrapModel("truth", cost, nx=4, ny=None, cached=archive)
data = truth.sample(bounds, pts=[2, 1, 1, 1], map=pmap)

# shutdown mapper
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()

# create an inexpensive surrogate for truth
surrogate = InterpModel("surrogate", nx=4, ny=None, data=truth, smooth=0.0,
                        noise=0.0, method="thin_plate", extrap=False)

# iterate until error (of candidate minimum) < 1e-3
import mystic._counter as it
counter = it.Counter()
tracker = LoggingMonitor(1, filename='error.txt', label='error')
from mystic.abstract_solver import AbstractSolver
from mystic.termination import VTR
loop = AbstractSolver(4) # nx
loop.SetTermination(VTR(1e-3)) #XXX: VTRCOG, TimeLimits, etc?
loop.SetEvaluationLimits(maxiter=500)
loop.SetGenerationMonitor(tracker)
while not loop.Terminated():

    # fit the surrogate to data in truth database
    surrogate.fit(data=data)

    # find the minimum of the surrogate
    results = diffev2(lambda x: surrogate(x), bounds, npop=20, gtol=500,
                      ftol=1e-6, maxiter=8000, maxfun=1e6, map=None,
                      itermon=LoggingMonitor(1, label='output'), disp=False,
                      bounds=bounds, full_output=True, id=counter.count())

    # evaluate truth at the same input as the surrogate minimum
    xnew = results[0].tolist()
    ynew = truth(xnew)

    # compute absolute error between truth and surrogate at candidate minimum
    ysur = results[1]
    error = abs(ynew - ysur)
    print("truth: %s @ %s" % (ynew, xnew))
    print("candidate: %s; error: %s" % (ysur, error))
    print("evaluations of truth: %s" % len(data))

    # save to tracker if less than current best
    if len(tracker) and tracker.y[-1] < error:
        tracker(*tracker[-1])
    else: tracker(xnew, error)

    # add most recent candidate minimum evaluated with truth to database
    pt = datapoint(xnew, value=ynew)
    data.append(pt)

# get the results at the best parameters from the truth database
xbest = tracker[-1][0]
ybest = archive[tuple(xbest)]
    
# print the best parameters
print(f"Best solution is {xbest} with Rwp {ybest}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, xbest)]
print(f"Ratios of best to reference solution: {ratios}")
