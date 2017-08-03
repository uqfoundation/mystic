#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
an interpolator
"""

class Surface(object):
   #surface has:
   #    args - interpolation configuration (smooth, function, ...)
   #    sampler - a search algorithm
   #    maxpts - a maximum number of sampling points
   #    noise - a noise coefficient
   #    dim - dimensionality of the model
   #    function - target function [F(*x)]
   #    objective - target function [F(x)]
   #    surrogate - interpolated function [F(*x)]
   #    model - interpolated function [F(x)]
   #    x,y,z - sampled points(*)
   #
   #surface (or sampler) has:
   #    _minmon,_maxmon - step monitor(*)
   #    _minarch,_maxarch - sampled point archives(*)
   #
   #surface can:
   #    Sample - populate sampled point archive with solver trajectories
   #    Interpolate - build interpolated function from sampled points
   #    Plot - plot sampled points and interpolated surface
   #
   #surface (or sampler) can:
   #    UseMonitor - track trajectories with a monitor(s)
   #    UseArchive - track sampled points in an archive(s)
   #    _noise - remove duplicate sampled points (x) by adding noise to x
   #    _downsample - skip sampled points at a regular interval (for speed)
   #    _max - fetch (x,y,z,model(x,y)) for maximal z of sampled points
   #    _min - fetch (x,y,z,model(x,y)) for minimal z of sampled points

    def __init__(self, objective, sampler=None, **kwds):
        # sampler configuration
        from mystic.search import Searcher
        self.sampler = Searcher() if sampler is None else sampler
        self.maxpts = kwds.pop('maxpts', None)  # N = 1000
        self.noise = kwds.pop('noise', 1e-8)
        # monitor, archive, and trajectories
        self._minmon = self._maxmon = None  #XXX: better default?
        self._minarch = self._maxarch = None  #XXX: better default?
        self.x = None  # params (x)
        self.z = None  # cost   (objective(x))
        # point generator(s) and interpolated model(s) #XXX: better names?
        self.dim = kwds.pop('dim', None)  #XXX: should be (or set) len(x)
        self.objective = objective # original F(x)
        self.surrogate = None # interpolated F(*x)
        # interpolator configuration
        self.args = {}#dict(smooth=0, function='thin_plate')
        self.args.update(kwds)
        return

  # XXX: useful?
  # def _invert_model:
  #     takes model and returns inverted_model (maxmodel or invmodel?)
  # def _invert_trajectories:
  #     takes (xyz) trajectories and returns inverted trajectories (xy-z)

    def UseMonitor(self, min=None, max=None):
        from mystic.monitors import Monitor
        if type(min) is bool: self._minmon = Monitor() if min else None
        elif min is not None: self._minmon = min
        if type(max) is bool: self._maxmon = Monitor() if max else None
        elif max is not None: self._maxmon = max
        return


    def UseArchive(self, min=None, max=None):
        from klepto.archives import dict_archive as d
        if type(min) is bool: self._minarch = d(cached=False) if min else None
        elif min is not None: self._minarch = min
        if type(max) is bool: self._maxarch = d(cached=False) if max else None
        elif max is not None: self._maxarch = max
        return

    """
    def doit(self, bounds, stop, step=200, scale=False, shift=False,
             density=9, axes=(), vals=(), maxpts=maxpts, **kwds):
        if not self.sampler.traj: self.sampler.UseTrajectories()
        # get trajectories
        self.Sample(bounds, stop)
        # get interpolated function
        self.Interpolate(**kwds)
        # check extrema  #XXX: put _min,_max in Interpolate? (downsampled)
        f = lambda x,z: (z,surface.surrogate(*x))
        print("min: {}; min@f: {}".format(*f(*surface._min())))
        print("max: {}; max@f: {}".format(*f(*urfacef._max())))
        # plot surface
        self.Plot(step, scale, shift, density, axes, vals, maxpts)
        return
    """

    def Sample(self, bounds, stop, clear=False, verbose=False):
        #XXX: does the strategy of finding min/max always apply?
        import numpy as np

        # get model (for minima)
        model = self.objective
        self.dim = len(bounds)

        ### get mins ###
        stepmon = self._minmon
        archive = None if clear else self._minarch
        inverse = False

        self.sampler.Reset(archive, inv=inverse) # reset the sampler
        self.sampler.Search(model, bounds, stop=stop, monitor=stepmon)
        if verbose: self.sampler._summarize()
        # read trajectories from log (or monitor)
        xyz = self.sampler.Samples()
        if clear: self.sampler.Reset()  # reset the sampler
        ### end mins ###

        # invert model (for maxima)
        imodel = lambda *args, **kwds: -model(*args, **kwds)

        ### get maxs ###
        stepmon = self._maxmon
        archive = None if clear else self._maxarch
        inverse = True

        self.sampler.Reset(archive, inv=inverse) # reset the sampler
        self.sampler.Search(imodel, bounds, stop=stop, monitor=stepmon)
        if verbose: self.sampler._summarize()
        xyz = np.hstack((xyz, self.sampler.Samples()))
        if clear: self.sampler.Reset()  # reset the sampler
        ### end maxs ###

        # split into params and cost
        self.x = xyz.T[:,:-1]
        self.z = xyz.T[:,-1]
        return self.x, self.z


    def _noise(self, scale=None, x=None):
        #HACK: remove any duplicate points by adding noise
        import numpy as np
        if x is None: x = self.x
        if scale is None: scale = self.noise
        if not scale: return x
        return x + np.random.normal(scale=scale, size=x.shape)


    def _downsample(self, maxpts=None, x=None, z=None):
        if maxpts is None: maxpts = self.maxpts
        if x is None: x = self.x
        if z is None: z = self.z
        if len(x) != len(z):
            raise ValueError("the input array lengths must match exactly")
        if maxpts is not None and len(z) > maxpts:
            N = max(int(round(len(z)/float(maxpts))),1)
        #   print("for speed, sampling {} down to {}".format(len(z),len(z)/N))
        #   ax.plot(x[:,0], x[:,1], z, 'ko', linewidth=2, markersize=4)
            x = x[::N]
            z = z[::N]
        #   plt.show()
        #   exit()
        return x, z

    def _interpolate(self, x, z, **kwds):
        import numpy as np
        from scipy.interpolate import Rbf as interpolate
        return interpolate(*np.vstack((x.T, z)), **kwds)


    def Interpolate(self, **kwds): #XXX: better take a strategy?
        maxpts = kwds.pop('maxpts', self.maxpts)
        noise = kwds.pop('noise', self.noise)
        args = self.args.copy()
        args.update(kwds)
        x, z = self._downsample(maxpts)
        #NOTE: really only need to add noise when have duplicate x,y coords
        x = self._noise(noise, x)
        # build the surrogate
        self.surrogate = self._interpolate(x, z, **args)
        self.surrogate.__doc__ = self.function.__doc__
        return self.surrogate


    def _max(self):
        import numpy as np
        x = self.x
        z = self.z
        mz = np.argmax(z)
        return x[mz],z[mz]

    def _min(self):
        import numpy as np
        x = self.x
        z = self.z
        mz = np.argmin(z)
        return x[mz],z[mz]


    def Plot(self, step=200, scale=False, shift=False, \
             density=9, axes=(), vals=(), maxpts=None):
        # plot interpolated surface
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np

        figure = plt.figure()
        kwds = {'projection':'3d'}
        ax = figure.gca(**kwds)
        ax.autoscale(tight=True)

        if maxpts is None: maxpts = self.maxpts
        x, z = self._downsample(maxpts)

        # get two axes to plot, and indices of the remaining axes
        axes = axes[:2]  #XXX: error if wrong size?
        ix = [i for i in range(len(x.T)) if i not in axes]
        n = 2-len(axes)
        axes, ix = list(axes)+ix[:n], ix[n:]

        # build list of fixed values (default mins), override with user input
       #fix = np.zeros(len(ix))
        fix = enumerate(self._min()[0])
        fix = np.array(tuple(j for (i,j) in fix if i not in axes))
        fix[:len(vals)] = vals

        # build grid of points, one for each param, apply fixed values
        grid = np.ones((len(x.T),step,step))
        grid[ix] = fix[:,None,None]
        del ix, fix

        # build sub-surface of surrogate(x) to display, apply to the grid
        xy = x.T[axes]
        M = complex('{}j'.format(step))
        grid[axes] = np.mgrid[xy[0].min():xy[0].max():M, 
                              xy[1].min():xy[1].max():M]
        del xy

        # evaluate the surrogate on the sub-surface
        z_ = self.surrogate(*grid)
	# scaling used by model plotter
        if scale:
            if shift:
                z_ = np.asarray(z_)+shift
            z_ = np.log(4*np.asarray(z_)*scale+1)+2

        # plot surface
        d = max(11 - density, 1)
        x_ = grid[axes[0]]
        y_ = grid[axes[-1]]
        ax.plot_wireframe(x_, y_, z_, rstride=d, cstride=d)
        #ax.plot_surface(x_, y_, z_, rstride=d, cstride=d, cmap=cm.jet, linewidth=0, antialiased=False)

        # use the sampled values
        z_ = z
	# scaling used by model plotter
        if scale:
            if shift:
                z_ = np.asarray(z_)+shift
            z_ = np.log(4*np.asarray(z_)*scale+1)+2

        # plot data points
        x_ = x.T[axes[0]]
        y_ = x.T[axes[-1]]
        ax.plot(x_, y_, z_, 'ko', linewidth=2, markersize=4)
        plt.show()  #XXX: show or don't show?... or return?
#       figure.savefig('griewangk.png')


    def __set_function(self, function): #XXX: deal w/ selector (2D)? ExtraArgs?
        # convert to 'model' format (i.e. takes a parameter vector)
        def objective(x, *args, **kwds):
            return function(*(tuple(x)+args), **kwds).tolist()
        self.objective = objective
        self.objective.__doc__ = function.__doc__
        return

    def __function(self): #XXX: deal w/ selector (2D)? ExtraArgs?
        # convert model to 'args' format (i.e. takes positional args)
        def function(*args, **kwds):
            len = self.dim # or kwds.pop('len', None)
            if len is None: return self.objective(args, **kwds)
            return self.objective(args[:len], *args[len:], **kwds)
        function.__doc__ = self.objective.__doc__
        return function

    def __model(self): #XXX: deal w/ selector (2D)? ExtraArgs?
        # convert to 'model' format (i.e. takes a parameter vector)
        if self.surrogate is None: return None
        def objective(x, *args, **kwds):
            return self.surrogate(*(tuple(x)+args), **kwds).tolist()
        objective.__doc__ = self.objective.__doc__
        return objective


    # interface
    function = property(__function, __set_function )
    model = property(__model )


class Surface_Rbf(Surface):
    pass

class Surface_Linear(Surface):
    def _interpolate(self, x, z, **kwds):
        from scipy.interpolate import LinearNDInterpolator as interpolate
        return interpolate(x, z, **kwds)

class Surface_Nearest(Surface):
    def _interpolate(self, x, z, **kwds):
        from scipy.interpolate import NearestNDInterpolator as interpolate
        return interpolate(x, z, **kwds)

class Surface_Clough(Surface):
    def _interpolate(self, x, z, **kwds):
        from scipy.interpolate import CloughTocher2DInterpolator as interpolate
        return interpolate(x, z, **kwds)


# EOF
