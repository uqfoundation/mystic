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
from mystic.solvers import diffev2
from mystic.math.legacydata import dataset, datapoint
from mystic.cache.archive import file_archive, read as get_db
from ouq_models import WrapModel, InterpModel
from emulators import cost4 as cost, x4 as target, bounds4 as bounds

# remove any prior cached evaluations of truth (i.e. an 'expensive' model)
if os.path.exists("truth.db"):
    os.remove("truth.db")

# remove any prior cached evaluations of surrogate
if os.path.exists("surrogate.db"):
    os.remove("surrogate.db")

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
N = 4
import numpy as np
error = float("inf")
sign = 1.0
while error > 1e-3:

    # fit a new surrogate to data in truth database
    candidates = get_db('surrogate.db', type=file_archive)
    candidates.clear()
    surrogate.fit(data=data)

    # find the minimum of the surrogate
    surr = surrogate.sample(bounds, pts='-.%s' % N, archive=candidates)
    idx = np.argmin(surr.values)

    # evaluate truth at the same input as the surrogate minimum
    xnew = surr.coords[idx]
    ynew = truth(xnew)

    # compute absolute error between truth and surrogate at candidate minimum
    ysur = surr.values[idx]
    error = abs(ynew - ysur)
    print("truth: %s @ %s" % (ynew, xnew))
    print("candidate: %s; error: %s" % (ysur, error))
    print("evaluations of truth: %s" % len(data))

    # add most recent candidate mimumim evaluated with truth to database
    pt = datapoint(xnew, value=ynew)
    data.append(pt)

# print the best parameters
print(f"Best solution is {xnew} with Rwp {ynew}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, xnew)]
print(f"Ratios of best to reference solution: {ratios}")
