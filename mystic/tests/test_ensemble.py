import numpy as np
from mystic.solvers import *
from mystic.models import rosen
from mystic.monitors import Monitor, VerboseMonitor
from mystic.tools import random_seed
ndim = 3
npts = 3
ini = 2

def get_xy(solver):
    mon = init_data()
    solver.SetEvaluationMonitor(mon)
    solver.SetStrictRanges(min=[0]*ndim, max=[10]*ndim)
    solver.SetObjective(rosen)
    solver.Step()
    solver.Solution()
    x = [getattr(i, '_evalmon', Monitor())._x[ini:][0] for i in solver._allSolvers]
    y = [getattr(i, '_evalmon', Monitor())._y[ini:][0] for i in solver._allSolvers]
    return x,y

def init_data():
    # add some intital data
    random_seed(123)
    mon = Monitor()
    mon._x = (np.random.random(size=(ini,ndim))*10).tolist()
    mon._y = (list(map(rosen, mon._x)) + np.random.normal(0,.05, size=ini)).tolist()
    return mon

solvers = ['ResidualSolver', 'SparsitySolver', 'BuckshotSolver', 'LatticeSolver']
for solver in solvers:
    samp = {solver: (npts,)}
    solver = eval(solver)(ndim, *samp[solver])
    x,y = get_xy(solver)
    solver = MixedSolver(ndim, samp)
    xm,ym = get_xy(solver)
    assert x == xm
    assert y == ym
