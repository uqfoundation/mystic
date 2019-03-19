#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2019 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
an interpolator
  - initalize with objective f(x) (and 'Sampler' object)
  - can attach a monitor and/or archiver
  - can sample points (using the Sampler)
  - can downsample and/or add noise
  - interpolates with "interp.interp"
  - converts f(*x) <-> f(x)
  - plot data and interpolated surface
"""

class Surface(object): #FIXME: should be subclass of Interpolator (?)
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
   #    _max - fetch (x,y,z,model(x,y)) for maximal z of sampled points
   #    _min - fetch (x,y,z,model(x,y)) for minimal z of sampled points

    def __init__(self, objective, sampler=None, **kwds):
        """response surface interpolator, where data is sampled from objective

        Input:
          objective: function of the form z=f(x)
          sampler: mystic.search.Searcher instance

        Additional Inputs:
          maxpts: int, maximum number of points to use from (x,z)
          noise: float, amplitude of gaussian noise to remove duplicate x
          method: string for kind of interpolator
          dim: number of parameters in the input for the objective function

        NOTE:
          if scipy is not installed, will use np.interp for 1D (non-rbf),
          or mystic's rbf otherwise. default method is 'nearest' for
          1D and 'linear' otherwise. method can be one of ('rbf','linear',
          'nearest','cubic','inverse','gaussian','quintic','thin_plate').
        """
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
        """track parameter trajectories with a monitor(s)

        Input:
          min: monitor instance to track minima; if True, use a new Monitor
          max: monitor instance to track maxima; if True, use a new Monitor

        Output:
          None
        """
        from mystic.monitors import Monitor
        if type(min) is bool: self._minmon = Monitor() if min else None
        elif min is not None: self._minmon = min
        if type(max) is bool: self._maxmon = Monitor() if max else None
        elif max is not None: self._maxmon = max
        return


    def UseArchive(self, min=None, max=None):
        """track sampled points in an archive(s)

        Input:
          min: archive instance to store minima; if True, use a new archive
          max: archive instance to store maxima; if True, use a new archive

        Output:
          None
        """
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
        self.Plot(step=step, scale=scale, shift=shift, density=density, axes=axes, vals=vals, maxpts=maxpts)
        return
    """

    def Sample(self, bounds, stop, clear=False, verbose=False, all=False):
        """sample data (x,z) using objective function z=f(x)

        Input:
          bounds: tuple of floats (min,max), bounds on the search region
          stop: termination condition
          clear: if True, clear the archive of stored points
          verbose: if True, print a summary of search/sampling results
          all: if True, use solver EvalMonitor, else use StepMonitor

        Output:
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,)
        """
        #XXX: does the strategy of finding min/max always apply?
        import numpy as np

        # get model (for minima)
        model = self.objective
        self.dim = len(bounds)

        ### get mins ###
        monitor = self._minmon
        archive = None if clear else self._minarch
        inverse = False

        self.sampler.Reset(archive, inv=inverse) # reset the sampler
        if all:
            self.sampler.Search(model, bounds, stop=stop, evalmon=monitor)
        else:
            self.sampler.Search(model, bounds, stop=stop, monitor=monitor)
        if verbose: self.sampler._summarize()
        # read trajectories from log (or monitor)
        xyz = self.sampler.Samples(all=all)
        if clear: self.sampler.Reset()  # reset the sampler
        ### end mins ###

        # invert model (for maxima)
        imodel = lambda *args, **kwds: -model(*args, **kwds)

        ### get maxs ###
        monitor = self._maxmon
        archive = None if clear else self._maxarch
        inverse = True

        self.sampler.Reset(archive, inv=inverse) # reset the sampler
        if all:
            self.sampler.Search(imodel, bounds, stop=stop, evalmon=monitor)
        else:
            self.sampler.Search(imodel, bounds, stop=stop, monitor=monitor)
        if verbose: self.sampler._summarize()
        xyz = np.hstack((xyz, self.sampler.Samples(all=all)))
        if clear: self.sampler.Reset()  # reset the sampler
        ### end maxs ###

        # split into params and cost
        self.x = xyz.T[:,:-1]
        self.z = xyz.T[:,-1]
        return self.x, self.z


    def Interpolate(self, **kwds): #XXX: refactor so use self.interpolator ?
        """interpolate data (x,z) to generate response function z=f(*x)

        Input:
          maxpts: int, maximum number of points to use from (x,z)
          noise: float, amplitude of gaussian noise to remove duplicate x
          method: string for kind of interpolator
          extrap: if True, extrapolate a bounding box (can reduce # of nans)
          arrays: if True, return a numpy array; otherwise don't return arrays

        Output:
          interpolated response function, where z=f(*x.T)

        NOTE:
          if scipy is not installed, will use np.interp for 1D (non-rbf),
          or mystic's rbf otherwise. default method is 'nearest' for
          1D and 'linear' otherwise. method can be one of ('rbf','linear',
          'nearest','cubic','inverse','gaussian','quintic','thin_plate').
        """
        from interpolator import Interpolator
        args = self.args.copy()
        args.update(kwds)
        maxpts, noise = self.maxpts, self.noise
        ii = Interpolator(self.x, self.z, maxpts=maxpts, noise=noise, **args)
        self.surrogate = ii.Interpolate(**args)
        # build the surrogate
        self.surrogate.__doc__ = self.objective.__doc__
        return self.surrogate


    def _max(self): #XXX: remove?
        """get the x[i],z[i] corresponding to the max(z)
        """
        import numpy as np
        mz = np.argmax(self.z)
        return self.x[mz], self.z[mz]

    def _min(self): #XXX: remove?
        """get the x[i],z[i] corresponding to the min(z)
        """
        import numpy as np
        mz = np.argmin(self.z)
        return self.x[mz], self.z[mz]


    def Plot(self, **kwds):
        """produce a scatterplot of (x,z) and the surface z = function(*x.T)

        Input:
          step: int, plot every 'step' points on the grid [default: 200]
          scale: float, scaling factor for the z-axis [default: False]
          shift: float, additive shift for the z-axis [default: False]
          density: int, density of wireframe for the plot surface [default: 9]
          axes: tuple, indicies of the axes to plot [default: ()]
          vals: list of values (one per axis) for unplotted axes [default: ()]
          maxpts: int, maximum number of (x,z) points to use [default: None]
        """
        # get interpolted function
        fx = self.surrogate
        # plot interpolated surface
        from plotter import Plotter
        p = Plotter(self.x, self.z, fx, **kwds)
        p.Plot()
        # if plotter interpolated the function, get the function
        self.surrogate = fx or p.function


    def __set_function(self, function): #XXX: deal w/ selector (2D)? ExtraArgs?
        # convert to 'model' format (i.e. takes a parameter vector)
        from mystic.math.interpolate import _to_objective
        _objective = _to_objective(function)
        def objective(x, *args, **kwds):
            result = _objective(x, *args, **kwds)
            return result.tolist() if hasattr(result, 'tolist') else result
        self.objective = objective
        self.objective.__doc__ = function.__doc__
        return

    def __function(self): #XXX: deal w/ selector (2D)? ExtraArgs? _to_function
        # convert model to 'args' format (i.e. takes positional args)
        from mystic.math.interpolate import _to_function
        function = _to_function(self.objective, ndim=self.dim)
        function.__doc__ = self.objective.__doc__
        return function

    def __model(self): #XXX: deal w/ selector (2D)? ExtraArgs? _to_objective
        # convert to 'model' format (i.e. takes a parameter vector)
        if self.surrogate is None: return None
        from mystic.math.interpolate import _to_objective
        _objective = _to_objective(self.surrogate)
        def objective(x, *args, **kwds):
            result = _objective(x, *args, **kwds)
            return result.tolist() if hasattr(result, 'tolist') else result
        objective.__doc__ = self.objective.__doc__
        return objective


    # interface
    function = property(__function, __set_function )
    model = property(__model )


# EOF
