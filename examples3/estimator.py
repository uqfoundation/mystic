#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
a function/surface estimator
  - initalize with x (and z)
  - can downsample and/or add noise
  - learns with sklearn interface (internally)
  - converts f(*x) <-> f(x)
  - plot data and learned surface
"""

class Estimator(object):

    def __init__(self, x, z=None, **kwds):
        """estimator for data (x,z)

        Input:
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,) or (npts, N)

        Additional Inputs:
          estimator: a fitted sklearn estimator
          transform: a fitted sklearn transform
          axis: int in [0,N], the index of z to estimate (all, by default)

        NOTE:
          if sklearn is not installed, an error will be thrown.

        NOTE:
          additional keyword arguments are available for use within the
          selected estimator and/or transform. The estimator and transform
          are expected to be sklearn instances where "fit" has already been
          called (e.g. sklearn.linear_model.LinearRegression().fit(x,z)).
          However, if any of the relevant hyperparameters for the estimator
          or the transform are provided, then those parameters will override
          the hyperparameters used in the relevant instance. Hyperparameters
          intended for the tranform should be passed as a dict named 'xfopt',
          while hyperparameters for the estimator can be given directly as
          keyword arguments.
        """
        # basic configuration
        self.maxpts = kwds.pop('maxpts', None)  # N = 1000
        self.noise = kwds.pop('noise', 0.0) #XXX: or 1e-8 ?
        # parameter trajectories (from arrays or monitor)
        self.x = getattr(x, '_x', x)  # params (x)
        self.z = x._y if z is None else z # cost (f(x))
        import numpy as np
        self.x = np.asarray(self.x); self.z = np.asarray(self.z)
        # estimator configuration
        self.args = {}
        self.function = None # learned F(*x)
        self._configure(kwds) # configure learned F(*x)
        return

    def _configure(self, kwds):
        """configure the learning instance

        Inputs:
          kwds: a dict of configuration options

        NOTE:
          All entries in kwds are used to update the estimator configuration,
          with the exception of the following:
            estimator: a fitted sklearn estimator
            transform: a fitted sklearn transform
            axis: int in [0,N], the index of z to estimate (all, by default)
            xfopt: dict of configuration options to update the transform
        """
        learner = self.args.get('learner', None)
        # get new estimator class/instance, if given
        estimator = kwds.pop('estimator', None)
        transform = kwds.pop('transform', None)
        if estimator is None:
            if learner is None:
                import sklearn.linear_model as lm
                estimator = lm.LinearRegression #FIXME: or ANN?
                # import sklearn.neural_network as nn
                # estimator = nn.MLPRegressor
            else: # use a copy of the existing instance
                from sklearn.base import clone
                estimator = clone(learner.estimator)
        if transform is None:
            if learner is None:
                import sklearn.preprocessing as pre
                transform = pre.StandardScaler #FIXME: default to None ?
            else: # use a copy of the existing instance
                from sklearn.base import clone
                transform = clone(learner.transform)
        # get options
        axis = kwds.pop('axis', self.args.get('axis', None))
        opts = {} if hasattr(transform, 'mro') else transform.get_params()
        args = {} if hasattr(estimator, 'mro') else estimator.get_params()
        opts.update(kwds.pop('xfopt', {})) #XXX: assumes no 'estimator' change
        args.update(kwds)
        # build learner
        if hasattr(transform, 'mro'):
            transform = transform(**opts)
        else:
            transform.set_params(**opts)
        if hasattr(estimator, 'mro'):
            estimator = estimator(**args)
        else:
            estimator.set_params(**args)
        from ml import Estimator as Learner
        learner = self.args['learner'] = Learner(estimator, transform)
        self.args['axis'] = axis
        return learner

    def _noise(self, scale=None, x=None):
        """inject gaussian noise into x to remove duplicate points

        Input:
          scale: amplitude of gaussian noise
          x: an array of shape (npts, dim) or (npts,)

        Output:
          array x, with added noise
        """
        import numpy as np
        if x is None: x = self.x
        if scale is None: scale = self.noise
        if not scale: return x
        return x + np.random.normal(scale=scale, size=x.shape)

    def _downsample(self, maxpts=None, x=None, z=None):
        """downsample (x,z) to at most maxpts

        Input:
          maxpts: int, maximum number of points to use from (x,z)
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,) or (npts, N)

        Output:
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,) or (npts, N)
        """
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

    def _train(self, x, z, **kwds):
        """learn data (x,z) to generate response function z=f(*x)

        Input:
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,) or (npts, N)

        Additional Inputs:
          estimator: a fitted sklearn estimator
          transform: a fitted sklearn transform
          axis: int in [0,N], the index of z to estimate (all, by default)

        Output:
          learned response function, where z=f(*x.T)

        NOTE:
          if sklearn is not installed, an error will be thrown.

        NOTE:
          additional keyword arguments are available for use within the
          selected estimator and/or transform. The estimator and transform
          are expected to be sklearn instances where "fit" has already been
          called (e.g. sklearn.linear_model.LinearRegression().fit(x,z)).
          However, if any of the relevant hyperparameters for the estimator
          or the transform are provided, then those parameters will override
          the hyperparameters used in the relevant instance. Hyperparameters
          intended for the tranform should be passed as a dict named 'xfopt',
          while hyperparameters for the estimator can be given directly as
          keyword arguments.
        """
        import numpy as np
        #import warnings
        #from sklearn.exceptions import ConvergenceWarning
        #warnings.simplefilter('ignore', ConvergenceWarning)
        #warnings.simplefilter('ignore', RuntimeWarning)
        _map = kwds.pop('map', map)
        learner = self._configure(kwds)
        axis = self.args.get('axis', None)
        # apply kwds to instantiate transform and estimator
        if axis is None:
            if len(z) and hasattr(z[0], '__len__'):
                #zt = np.array if arrays else list
                zt = list #XXX: is this ever desired to be an array?
                # iterate over each axis, build a 'combined' learner
                def function(*args, **kwds): #XXX: z = array(f(*x.T)).T
                    axis = kwds.get('axis', None)
                    fs = function.__axis__
                    if axis is None:
                        if hasattr(args[0], '__len__'):
                            return tuple(zt(fi(*args)) for fi in fs)
                        return tuple(fi(*args) for fi in fs)
                    return fs[axis](*args)
                def learn_ax(i):
                    import numpy as np
                    from sklearn.base import clone
                    estimator = clone(learner.estimator)
                    transform = clone(learner.transform)
                    from mystic.math.interpolate import _getaxis
                    from ml import Estimator as Learner
                    func = Learner(estimator, transform)
                    with np.warnings.catch_warnings(): #FIXME: enable warn=True
                        np.warnings.filterwarnings('ignore')
                        func = func.train(x, _getaxis(z, i))
                    return func
                function.__axis__ = list(_map(learn_ax, range(len(z[0]))))
                return function
        else:
            from mystic.math.interpolate import _getaxis
            z = _getaxis(z, axis)
        with np.warnings.catch_warnings(): #FIXME: enable warn=True
            np.warnings.filterwarnings('ignore')
            function = learner.train(x, z)
        function.__axis__ = axis
        return function

    def Train(self, **kwds): #XXX: better take a strategy?
        """train data (x,z) to generate response function z=f(*x)

        Input:
          maxpts: int, maximum number of points to use from (x,z)
          noise: float, amplitude of gaussian noise to remove duplicate x

        Additional Input:
          estimator: a fitted sklearn estimator
          transform: a fitted sklearn transform
          axis: int in [0,N], the index of z to estimate (all, by default)

        Output:
          learned response function, where z=f(*x.T)

        NOTE:
          if sklearn is not installed, an error will be thrown.

        NOTE:
          additional keyword arguments are available for use within the
          selected estimator and/or transform. The estimator and transform
          are expected to be sklearn instances where "fit" has already been
          called (e.g. sklearn.linear_model.LinearRegression().fit(x,z)).
          However, if any of the relevant hyperparameters for the estimator
          or the transform are provided, then those parameters will override
          the hyperparameters used in the relevant instance. Hyperparameters
          intended for the tranform should be passed as a dict named 'xfopt',
          while hyperparameters for the estimator can be given directly as
          keyword arguments.
        """
        maxpts = kwds.pop('maxpts', self.maxpts)
        noise = kwds.pop('noise', self.noise)
        x, z = self._downsample(maxpts)
        #NOTE: really only need to add noise when have duplicate x,y coords
        x = self._noise(noise, x)
        # build the surrogate
        self.function = self._train(x, z, **kwds)
        return self.function

    def Test(self, x=None, **kwds):
        """evaluate data x with response function z=f(*x)

        Input:
          x: an array of shape (npts, dim) or (npts,)

        Additional Input:
          arrays: if True, z = f(*x) is a numpy array; otherwise don't use arrays
          axis: int in [0,N], the index of z to estimate (all, by default)

        Output:
          z: an array of shape (npts,) or (npts, N)

        NOTE:
          if sklearn is not installed, an error will be thrown.

        NOTE:
          if x is not provided, test against the x used for training.
        """
        #NOTE: don't add noise or downsample here
        axis = kwds.get('axis', None)
        arrays = kwds.get('arrays', False)
        import numpy as np
        zt = np.array if arrays else list
        if x is None:
            x = self.x
        if axis is None:
            return zt([self.function(*xi) for xi in x])
        if hasattr(self.function, '__axis__') and hasattr(self.function.__axis__, '__len__'):
            z = self.function.__axis__[axis].test(x)
            return z if arrays else z.tolist()
        z = self.function.test(x)
        return z if arrays else z.tolist()

    # def Score(self, **kwds): #FIXME: add this?
    #     pass

    def Plot(self, **kwds):
        """produce a scatterplot of (x,z) and the surface z = function(*x.T)

        Input:
          step: int, plot every 'step' points on the grid [default: 200]
          scale: float, scaling factor for the z-axis [default: False]
          shift: float, additive shift for the z-axis [default: False]
          density: int, density of wireframe for the plot surface [default: 9]
          axes: tuple, indicies of the axes to plot [default: ()]
          axis: int, index of the z-axis to plot, if multi-dim [default: 0]
          vals: list of values (one per axis) for unplotted axes [default: ()]
          maxpts: int, maximum number of (x,z) points to use [default: None]
          kernel: function transforming x to x', where x' = kernel(x)
          vtol: float, maximum distance outside bounds hypercube to plot data

        NOTE: the default axis is 0 unless an estimator axis has been set
        """
        axis = kwds.pop('axis', self.args.get('axis', 0))
        # get learned function
        fx = self.function or self.Train()
        # plot learned surface
        from plotter import Plotter
        p = Plotter(self.x, self.z, fx, axis=axis, **kwds)
        p.Plot()
        # if function was trained, save the function
        self.function = fx

    def __model(self): #XXX: deal w/ selector (2D)? ExtraArgs?
        # convert to 'model' format (i.e. takes a parameter vector)
        if self.function is None: return None
        from mystic.math.interpolate import _to_objective
        _objective = _to_objective(self.function)
        def objective(x, *args, **kwds):
            result = _objective(x, *args, **kwds)
            return result.tolist() if hasattr(result, 'tolist') else result
        objective.__doc__ = _objective.__doc__
        return objective

    # interface
    model = property(__model )


if __name__ == '__main__':

    from ouq_models import *
    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None

    # build a model representing 'truth'
    truth = dict(model=toy, nx=nx, ny=ny, mu=.001, zmu=-.001)
    golden = NoisyModel('golden', cached=True, sigma=0, zsigma=0, **truth)

    # generate some 'truth' data, using solver-directed sampling
    bounds = [(0,10)]*nx
    data = golden.sample(bounds, pts=-4)

    # build an estimator instance 
    import mystic.monitors as mm
    m = mm.Monitor()
    m._x, m._y = data.coords, data.values
    args = dict(hidden_layer_sizes=(100,75,50,25), max_iter=1000, n_iter_no_change=5, solver = 'lbfgs')
    import sklearn.neural_network as nn
    mlp = nn.MLPRegressor(**args)
    e = Estimator(m._x, m._y, estimator=mlp)

    # train on 'truth' data
    print('training...')
    f = e.Train(axis=None)

    print('spot-checking the first point...')
    print('using function: %s' % str(f(*m._x[0])))
    print('actual value: %s' % str(m._y[0]))

    # test on training data
    import numpy as np
    import sklearn.metrics as sm
    x, y = np.array(m._x), np.array(m._y)
    #print('r2 scores: %s' % str(tuple(sm.r2_score(j.test(x), y[:,i]) for i,j in enumerate(f.__axis__))))
    #ypred = np.array([f(*xi) for xi in x])
    ypred = e.Test(x, axis=None, arrays=True)
    if f.__axis__ is None:
        print('r2 score: %s' % str(sm.r2_score(ypred, y)))
    else:
        print('r2 scores: %s' % str(tuple(sm.r2_score(ypred[:,i], y[:,i]) for i,j in enumerate(f.__axis__))))

