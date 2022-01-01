#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
an example of using an interpolator within a surface object
""" #XXX: use interpolator, plotter, and sampler (instead of Surface)?

from surface import Surface
import time


if __name__ == '__main__':
    start = time.time()
    """
    from mystic.models import griewangk
    from mystic.termination import NormalizedChangeOverGeneration as NCOG

    stop = NCOG(1e-4)
    bounds = 2*[(-9.5,9.5)]

    self = Surface(griewangk, maxpts=1000)

    # self.doit(bounds, stop)
    step=100; scale=False; shift=False; density=9; kwds={}

    if not self.sampler.traj: self.sampler.UseTrajectories()
    # get trajectories
    self.Sample(bounds, stop)
    # get interpolated function
    self.Interpolate(**kwds)
    # check extrema  #XXX: put _min,_max in Interpolate? (downsampled)
    f = lambda x,z: (z,self.surrogate(*x))
    print("min: {}; min@f: {}".format(*f(*self._min())))
    print("max: {}; max@f: {}".format(*f(*self._max())))
    # plot surface
    self.Plot(step=step, scale=scale, shift=shift, density=density)
    """

    # parallel configuration
    try:
        from pathos.helpers import freeze_support, shutdown
        freeze_support()
        from pathos.pools import ProcessPool as Pool
       #from pathos.pools import ThreadPool as Pool
       #from pathos.pools import ParallelPool as Pool
    except ImportError:
        from mystic.pools import SerialPool as Pool
        shutdown = lambda x=None:None

    _map = Pool().map

    # tools
    from mystic.termination import VTR, ChangeOverGeneration as COG
    from mystic.termination import NormalizedChangeOverGeneration as NCOG
    from mystic.monitors import LoggingMonitor, VerboseMonitor, Monitor
    from klepto.archives import dir_archive

    stop = NCOG(1e-4)
    disp = True # print optimization summary
    monitor = True # use LoggingMonitor (uses much less memory)
    archive = False # save an archive
    all = True # use EvalMonitor (instead of StepMonitor only)

    traj = not monitor # save all trajectories internally, if no logs

    # cost function
    from mystic.models import griewangk as model
    ndim = 2  # model dimensionality
    bounds = ndim * [(-9.5,9.5)] # griewangk

    # the ensemble solvers
    from mystic.solvers import BuckshotSolver, LatticeSolver, SparsitySolver
    # the local solvers
    from mystic.solvers import PowellDirectionalSolver

    sprayer = BuckshotSolver
    seeker = PowellDirectionalSolver
    npts = 25    # number of solvers
    retry = 1    # max consectutive iteration retries without a cache 'miss'
    repeat = 0   # number of times to repeat the search
    tol = 8      # rounding precision
    mem = 1      # cache rounding precision
    size = 0     # max in-memory cache size

    #CUTE: 'configure' monitor and archive if they are desired
    if monitor:
        monitor = LoggingMonitor(1) # montor for all runs
        costmon = LoggingMonitor(1, filename='inv.txt') #XXX: log.txt?
    else:
        monitor = costmon = None
    if archive: #python2.5
        name = getattr(model,'__name__','model')
        ar_name = '__%s_%sD_cache__' % (name,ndim)
        archive = dir_archive(ar_name, serialized=True, cached=False)
        ar_name = '__%s_%sD_invcache__' % (name,ndim)
        ivcache = dir_archive(ar_name, serialized=True, cached=False)
    else:
        archive = ivcache = None

    from mystic.search import Searcher #XXX: init w/ archive, then UseArchive?
    expts,evals = (None,archive) if all else (archive, None)
    #expts,evals = (archive, None) #XXX: don't override the sample archive
    sampler = Searcher(npts, retry, tol, mem, size, _map, evals, expts, sprayer, seeker, repeat=repeat)
    sampler.Verbose(disp)
    sampler.UseTrajectories(traj)

    ### doit ###
    maxpts = 1000. #10000.
    surface = Surface(model, sampler, maxpts=maxpts, dim=ndim)
    surface.UseMonitor(monitor, costmon)
    surface.UseArchive(archive, ivcache)

    density = 9
    shift = 0
    scale = 0
    step = 200
    args = {
   #'smooth': 0,
    'method': 'thin_plate',
    'extrap': False,
    'arrays': False,
    }
    #surface.doit(bounds, stop, step=step)
   #'multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate'

    # impose data filter (so X >= 0 and Y >= 0) 
    from mystic.filters import generate_mask, generate_filter
    from mystic.constraints import impose_bounds
    _bounds = impose_bounds((0.0, None))(lambda x:x)
    filter = generate_filter(generate_mask(_bounds, _bounds))
    # from numpy import round as _round
    #############

    # get trajectories
    surface.Sample(bounds, stop, all=all, filter=filter)#, constraints=_round)
    print("TOOK: %s" % (time.time() - start))
#   exit()
    # get interpolated function
    surface.Interpolate(**args)
    # check extrema  #XXX: put _min,_max in Interpolate? (downsampled)
    f = lambda x,z: (z,surface.surrogate(*x))
    print("min: {}; min@f: {}".format(*f(*surface._min())))
    print("max: {}; max@f: {}".format(*f(*surface._max())))
    # shutdown worker pool
    shutdown()
#   print("TOOK: %s" % (time.time() - start))

    # plot interpolated surface
    axes = (0,1)
    vals = () # use remaining minima as the fixed values
    surface.Plot(step=step, scale=scale, shift=shift, density=density, axes=axes, vals=vals)

    """
    try:
        from klepto.archives import file_archive
        archive = file_archive('models.pkl', serialized=True, cached=False)
        archive[model.im_class.__name__.lower()] = surface.surrogate
    except Exception:
        print("serialization failed")
    """
    # some testing of interpolated model
    #import numpy as np
    #actual = np.asarray(surface.z)           # downsample?
    #interp = surface.surrogate(*surface.x.T) # downsample? #NOTE: SLOW
    #print("sum diff squares") #NOTE: is *worse* than with test_searcher.py
    #print("actual and interp: %s" % np.sum((actual - interp)**2))

# EOF
