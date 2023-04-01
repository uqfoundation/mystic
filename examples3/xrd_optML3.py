#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# adapted from XRD example at: https://spotlight.readthedocs.io/
"""
optimization of 3-input cost function using online learning of a surrogate
"""
import os
import shutil
from mystic.solvers import diffev2
from mystic.math.legacydata import dataset, datapoint
from ouq_models import WrapModel, InterpModel
from emulators import cost3 as cost, x3 as target, bounds3 as bounds

# remove any prior cached results
if os.path.exists("surrogate"):
    shutil.rmtree("surrogate")

try: # parallel maps
    from pathos.maps import Map
    from pathos.pools import ProcessPool, ThreadPool, SerialPool
    pmap = Map(SerialPool) #ProcessPool
except ImportError:
    pmap = None

# generate a sampled dataset for the model
truth = WrapModel("surrogate", cost, nx=3, ny=None, cached=False)
data = truth.sample(bounds, pts=[2, 1, 1], map=pmap)

# shutdown mapper
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()

# create surrogate model
surrogate = InterpModel("surrogate", nx=3, ny=None, data=truth, smooth=0.0,
                        noise=0.0, method="thin_plate", extrap=False)

# iterate until error < 1e-3
error = float("inf")
sign = 1.0
while error > 1e-3:

    # fit surrogate to data in database
    surrogate.fit(data=data)

    # find minimum/maximum of surrogate
    results = diffev2(lambda x: sign * surrogate(x), bounds, npop=20,
                      bounds=bounds, gtol=500, full_output=True)

    # get minimum/maximum of actual expensive model
    xnew = results[0].tolist()
    ynew = truth(xnew)

    # compute error which is actual model value - surrogate model value
    ysur = results[1]
    error = abs(ynew - ysur)

    # print statements
    print("truth", xnew, ynew)
    print("surrogate", xnew, ysur)
    print("error", ynew - ysur, error)
    print("data", len(data))

    # add latest point evaluated with actual expensive model to database
    pt = datapoint(xnew, value=ynew)
    data.append(pt)

# print the best parameters
print(f"Best solution is {xnew} with beta {ynew}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, xnew)]
print(f"Ratios of best to reference solution: {ratios}")
