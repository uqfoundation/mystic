#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
example of using a global searcher to find all extrema (and rough interpolation)
"""

from mystic.search import Searcher
#import time


if __name__ == '__main__':
#   start = time.time()
    # if available, use a multiprocessing worker pool
    try:
        from pathos.helpers import freeze_support, shutdown
        freeze_support()
        from pathos.pools import ProcessPool as Pool
    except ImportError:
        from mystic.pools import SerialPool as Pool
        shutdown = lambda x=None:None

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
    npts = 25    # number of solvers
    _map = Pool().map
    retry = 2    # max consectutive iteration retries without a cache 'miss'
    repeat = 2   # number of times to repeat the search
    tol = 8      # rounding precision
    mem = 1      # cache rounding precision
    size = None  # max in-memory cache size

    #CUTE: 'configure' monitor and archive if they are desired
    if stepmon: stepmon = LoggingMonitor(1) # montor for all runs
    else: stepmon = None
    if archive: #python2.5
        name = getattr(model,'__name__','model')
        ar_name = '__%s_%sD_cache__' % (name,ndim)
        archive = dir_archive(ar_name, serialized=True, cached=False)
    else: archive = None

    # configure a Searcher to use a "trajectory cache"
    searcher = Searcher(npts, retry, tol, mem, size, _map, None, archive, sprayer, seeker, repeat=repeat)
    searcher.Verbose(disp)
    searcher.UseTrajectories(traj)

    searcher.Reset(archive, inv=False)
    searcher.Search(model, bounds, stop=stop, monitor=stepmon)
    searcher._summarize()

    ##### extract results #####
    xyz = searcher.Samples()

    ##### invert the model, and get the maxima #####
    imodel = lambda *args, **kwds: -model(*args, **kwds)

    #CUTE: 'configure' monitor and archive if they are desired
    if stepmon not in (None, False):
        itermon = LoggingMonitor(1, filename='inv.txt') #XXX: log.txt?
    else: itermon = None
    if archive not in (None, False): #python2.5
        name = getattr(model,'__name__','model')
        ar_name = '__%s_%sD_invcache__' % (name,ndim)
        archive = dir_archive(ar_name, serialized=True, cached=False)
    else: archive = None

    searcher.Reset(archive, inv=True)
    searcher.Search(imodel, bounds, stop=stop, monitor=itermon)
    searcher._summarize()

    ##### extract results #####
    import numpy as np
    xyz = np.hstack((xyz, searcher.Samples()))
    # shutdown worker pool
    shutdown()
#   print("TOOK: %s" % (time.time() - start))

    ########## interpolate ##########
    from mystic.math.interpolate import Rbf

    #############
    shift = 0
    scale = 0
    N = 1000.
    M = 200
    args = {
    'smooth': 0,
    'function': 'thin_plate',
    }
    #############

    # get params (x) and cost (z)
    x, z = xyz.T[:,:-1], xyz.T[:,-1]

    #HACK: remove any duplicate points by adding noise
    _x = x + np.random.normal(scale=1e-8, size=x.shape)

    if len(z) > N:
        N = max(int(round(len(z)/float(N))),1)
        print("for speed, sampling {} down to {}".format(len(z),len(z)/N))
        x, _x, z = x[::N], _x[::N], z[::N]

    f = Rbf(*np.vstack((_x.T, z)), **args)
    f.__doc__ = model.__doc__
    # convert to 'model' format (i.e. takes a parameter vector)
    _model = lambda x: f(*x).tolist()
    _model.__doc__ = f.__doc__

    mz = np.argmin(z)
    print("min: {}; min@f: {}".format(z[mz], f(*x[mz])))
    mz = np.argmax(z)
    print("max: {}; max@f: {}".format(z[mz], f(*x[mz])))

#   print("TOOK: %s" % (time.time() - start))

    # plot
    #############
    # specify 2-of-N dim (with bounds) and (N-2) with values
    axes = (0,1)  # axes to plot               (specified by user)
    vals = ()     # values for remaining param (specified by user)
    #############

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    figure = plt.figure()
    kwds = {'projection':'3d'}
    ax = figure.axes[0] if figure.axes else plt.axes(projection='3d')
    ax.autoscale(tight=True)

    # build a list of fixed values (default zero) and override with user input
    ix = [i for i in range(len(x.T)) if i not in axes]
   #fix = np.zeros(len(ix))
    fix = enumerate(x[np.argmin(z)])
    fix = np.array(tuple(j for (i,j) in fix if i not in axes))
    fix[:len(vals)] = vals

    # build a grid of points, one for each param, and apply fixed values
    grid = np.ones((len(x.T),M,M))
    grid[ix] = fix[:,None,None]
    del ix, fix

    # build the sub-surface of surrogate(x) to display, and apply to the grid
    axes = list(axes)
    xy = x.T[axes]
    S = complex('{}j'.format(M))
    grid[axes] = np.mgrid[xy[0].min():xy[0].max():S, xy[1].min():xy[1].max():S]
    del xy

    # evaluate the surrogate on the sub-surface
    z_ = f(*grid)

    # scaling used by model plotter
    if scale:
        if shift:
            z_ = np.asarray(z_)+shift
            z = np.asarray(z)+shift
        z_ = np.log(4*np.asarray(z_)*scale+1)+2
        z = np.log(4*np.asarray(z)*scale+1)+2

    # plot the surface and points
    density = 9
    d = max(11 - density, 1)
    x_ = grid[axes[0]]
    y_ = grid[axes[-1]]
    ax.plot_wireframe(x_, y_, z_, rstride=d, cstride=d)
    #ax.plot_surface(x_, y_, z_, rstride=d, cstride=d, cmap=cm.jet, linewidth=0, antialiased=False)
    x_ = x.T[axes[0]]
    y_ = x.T[axes[-1]]
    ax.plot(x_, y_, z, 'ko', linewidth=2, markersize=4)

    plt.show()
#   figure.savefig('griewangk.png')

    """
    try:
        from klepto.archives import file_archive
        archive = file_archive('models.pkl', serialized=True, cached=False)
        archive[model.im_class.__name__.lower()] = f
    except Exception:
        print("serialization failed")
    """

    # some testing of interpolated model
    data = np.asarray(z)
#   actual = [model([xi,yi]) for (xi,yi) in zip(x,y)]
    interp = f(*x.T)
    print("sum diff squares")
#   print("data and actual: %s" % np.sum((data - actual)**2))
#   print("data and interp: %s" % np.sum((data - interp)**2))
#   print("actual and interp: %s" % np.sum((actual - interp)**2))
    print("actual and interp: %s" % np.sum((data - interp)**2))

# EOF
