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
from mystic.solvers import diffev2
from mystic.math.legacydata import dataset, datapoint
from mystic.cache.archive import file_archive, read as get_db
from ouq_models import WrapModel, LearnedModel
from emulators import cost3 as cost, x3 as target, bounds3 as bounds

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
truth = WrapModel("truth", cost, nx=3, ny=None, cached=archive)
data = truth.sample(bounds, pts=[2, 1, 1], map=pmap)

# shutdown mapper
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()

# create an inexpensive surrogate for truth
args = dict(n_estimators=100, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0)
#args['max_features'] = None
#args['max_leaf_nodes'] = None
#args['max_samples'] = None
args['warm_start'] = True
args['bootstrap'] = False
args['oob_score'] = False
args['criterion'] = 'squared_error'
#from sklearn.ensemble import RandomForestRegressor as DTRegressor
from sklearn.ensemble import ExtraTreesRegressor as DTRegressor
from ml import Estimator, MLData, improve_score
kwds = dict(estimator=DTRegressor(**args), transform=None)
# iteratively improve estimator
dtr = Estimator(**kwds)
best = improve_score(dtr, MLData(data.coords, data.coords, data.values, data.values), tries=10, verbose=True)
mlkw = dict(estimator=best.estimator, transform=best.transform)
surrogate = LearnedModel('surrogate', nx=3, ny=None, data=truth, **mlkw)

# iterate until error (of candidate minimum) < 1e-3
error = float("inf")
sign = 1.0
while error > 1e-3:

    # fit the surrogate to data in truth database
    surrogate.fit(data=data)

    # find the minimum of the surrogate
    results = diffev2(lambda x: sign * surrogate(x), bounds, npop=20,
                      bounds=bounds, gtol=500, full_output=True)

    # evaluate truth at the same input as the surrogate minimum
    xnew = results[0].tolist()
    ynew = truth(xnew)

    # compute absolute error between truth and surrogate at candidate minimum
    ysur = results[1]
    error = abs(ynew - ysur)
    print("truth: %s @ %s" % (ynew, xnew))
    print("candidate: %s; error: %s" % (ysur, error))
    print("evaluations of truth: %s" % len(data))

    # add most recent candidate mimumim evaluated with truth to database
    pt = datapoint(xnew, value=ynew)
    data.append(pt)

# print the best parameters
print(f"Best solution is {xnew} with beta {ynew}")
print(f"Reference solution: {target}")
ratios = [x / y for x, y in zip(target, xnew)]
print(f"Ratios of best to reference solution: {ratios}")
