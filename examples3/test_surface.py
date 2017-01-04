#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
an example of using an interpolator within a surface object
"""

from surface import Surface_Rbf as Surface
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
    print "min: {}; min@f: {}".format(*f(*self._min()))
    print "max: {}; max@f: {}".format(*f(*self._max()))
    # plot surface
    self.Plot(step, scale, shift, density)
    """

    # parallel configuration
    try:
        from pathos.helpers import freeze_support
        freeze_support()
        from pathos.pools import ProcessPool as Pool
       #from pathos.pools import ThreadPool as Pool
       #from pathos.pools import ParallelPool as Pool
    except ImportError:
        from mystic.pools import SerialPool as Pool

    _map = Pool().map

    # tools
    from mystic.termination import VTR, ChangeOverGeneration as COG
    from mystic.termination import NormalizedChangeOverGeneration as NCOG
    from mystic.monitors import LoggingMonitor, VerboseMonitor, Monitor
    from klepto.archives import dir_archive

    stop = NCOG(1e-4)
    disp = False # print optimization summary
    stepmon = False # use LoggingMonitor
    archive = False # save an archive
    traj = not stepmon # save all trajectories internally, if no logs

    # cost function
    from mystic.models import griewangk as model
    ndim = 2  # model dimensionality
    bounds = ndim * [(-9.5,9.5)] # griewangk

    # the ensemble solvers
    from mystic.solvers import BuckshotSolver, LatticeSolver
    # the local solvers
    from mystic.solvers import PowellDirectionalSolver

    sprayer = BuckshotSolver
    seeker = PowellDirectionalSolver
    npts = 25 # number of solvers
    retry = 1 # max consectutive iteration retries without a cache 'miss'
    tol = 8   # rounding precision
    mem = 1   # cache rounding precision

    #CUTE: 'configure' monitor and archive if they are desired
    if stepmon:
        stepmon = LoggingMonitor(1) # montor for all runs
        itermon = LoggingMonitor(1, filename='inv.txt') #XXX: log.txt?
    else:
        stepmon = itermon = None
    if archive:
        ar_name = '__%s_%sD_cache__' % (model.im_class.__name__,ndim)
        archive = dir_archive(ar_name, serialized=True, cached=False)
        ar_name = '__%s_%sD_invcache__' % (model.im_class.__name__,ndim)
        ivcache = dir_archive(ar_name, serialized=True, cached=False)
    else:
        archive = ivcache = None

    from mystic.search import Searcher #XXX: init w/ archive, then UseArchive?
    sampler = Searcher(npts, retry, tol, mem, _map, archive, sprayer, seeker)
    sampler.Verbose(disp)
    sampler.UseTrajectories(traj)

    ### doit ###
    maxpts = 1000. #10000.
    surface = Surface(model, sampler, maxpts=maxpts, dim=ndim)
    surface.UseMonitor(stepmon, itermon)
    surface.UseArchive(archive, ivcache)

    density = 9
    shift = 0
    scale = 0
    step = 200
    args = {
    'smooth': 0,
    'function': 'thin_plate',
    }
    #surface.doit(bounds, stop, step=step)
   #'multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate'
    #############

    # get trajectories
    surface.Sample(bounds, stop)
    print "TOOK: %s" % (time.time() - start)
#   exit()
    # get interpolated function
    surface.Interpolate(**args)
    # check extrema  #XXX: put _min,_max in Interpolate? (downsampled)
    f = lambda x,z: (z,surface.surrogate(*x))
    print "min: {}; min@f: {}".format(*f(*surface._min()))
    print "max: {}; max@f: {}".format(*f(*surface._max()))

#   print "TOOK: %s" % (time.time() - start)

    # plot surface
    axes = (0,1)
    vals = () # use remaining minima as the fixed values
    surface.Plot(step, scale, shift, density, axes, vals)


    """
    try:
        from klepto.archives import file_archive
        archive = file_archive('models.pkl', serialized=True, cached=False)
        archive[model.im_class.__name__.lower()] = surface.surrogate
    except Exception:
        print "serialization failed"
    """

    # some testing of interpolated model
    import numpy as np
    actual = np.asarray(surface.z)           # downsample?
    interp = surface.surrogate(*surface.x.T) # downsample?
    print "sum diff squares"
    print "actual and interp: %s" % np.sum((actual - interp)**2)


# EOF
