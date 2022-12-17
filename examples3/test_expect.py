#!/usr/bin/env python
# 
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from misc import *
from ouq import ExpectedValue
from mystic.bounds import MeasureBounds
from ouq_models import WrapModel
from toys import function5x3, function5x1, function5
from noisy import noisy as noise
import numpy as np

# build a 5x3, 5x1, and 5x0 model
cost3 = lambda rv: np.log(np.abs(function5x3(rv))).tolist()
cost1 = lambda rv: np.log(np.abs(function5x1(rv))).tolist()
cost0 = lambda rv: np.log(np.abs(function5(rv))).tolist()
noisyin3 = lambda rv: cost3(noise(rv, sigma=.1))
noisyin0 = lambda rv: cost0(noise(rv, sigma=.1))
noisyout0 = lambda rv: (noise((cost0(rv),), sigma=.1)[0])

# build a model representing 'truth' F(x)
cost = cost3
nx = 5; ny = 3
N = 500 # number of samples in the input distribution
Ni = N # number of samples per sampling iteration
Ns = None #500 # number of samples of F(x) in the objective
nargs = dict(nx=nx, ny=ny, rnd=(True if Ns else False))
model = WrapModel('model', cost, **nargs)

# set the bounds
bnd = MeasureBounds(xlb, xub, n=npts, wlb=wlb, wub=wub)

try: # parallel maps
    from pathos.maps import Map
    from pathos.pools import ProcessPool, ThreadPool, _ThreadPool
    pmap = Map(ProcessPool) if Ns else Map() # for sampling
    param['map'] = Map()#Map(ThreadPool) # for objective
    if ny: param['axmap'] = Map(_ThreadPool, join=True) # for multi-axis
    smap = Map() # for expected #XXX: multi-axis parallel db issues?
except ImportError:
    pmap = smap = None

import datetime
kwds.update(dict(npts=N, ipts=Ni, smap=smap, verbose=True))
axis = None

# where x,F(x) has uncertainty, calculate:
# expected value
print('\n%s'% datetime.datetime.now())
#param['id'] = 0
b = ExpectedValue(model, bnd, constraint=ccons, cvalid=is_cons, samples=Ns, map=pmap)
b.expected(archive='ave.db', axis=axis, **kwds, **param)
print("expected value per axis:")
for ax,solver in b._expect.items():
    print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

# shutdown
print('\n%s'% datetime.datetime.now())
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()
pmap = param['map']
if pmap is not None:
    pmap.close(); pmap.join(); pmap.clear()
if ny:
    pmap = param['axmap']
    if pmap is not None:
        pmap.close(); pmap.join(); pmap.clear()
if smap is not None:
    smap.close(); smap.join(); smap.clear()
