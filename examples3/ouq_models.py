#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
model objects (and helper functions) to be used with OUQ classes
'''
#FIXME: hardwired to multivalue function
#FIXME: dict_archive('truth', cached=False) does not cache (is empty)
#FIXME: option to cache w/o lookup (e.g. for model with randomness)
def sample(model, bounds, pts=None, **kwds):
    """sample model within bounds, writing to an archive and returning data

    Inputs:
        model: a cached model function, of form y = model(x, axis=None)
        bounds: list of tuples of (lower,upper), bounds on each 'x'
        pts: int, number of points sampled by the sampler

    Additional Inputs:
        sampler: the mystic.sampler type [default: LatticeSampler]
        solver: the mystic.solver type [default: NelderMeadSimplexSolver]
        dist: a distribution type (or float amplitude) [default: None]
        map: map instance, to search for min/max in parallel [default: None]
        ny: int, number of model outputs, len(y) [default: None]
        axis: int, index of output on which to search [default: None]
        axmap: map instance, to execute each axis in parallel [default: None]
        archive: the archive (str name or archive instance) [default: None]

    Returns:
        the mystic.math.legacydata.dataset of sampled data

    NOTE:
        additional keywords (evalmon, stepmon, maxiter, maxfun,
        saveiter, state, termination, constraints, penalty, reducer)
        are available for use. See mystic.ensemble for more details.

    NOTE:
        dist can be used to add randomness to the sampler, and can
        accept a numpy distribution type such as numpy.random.normal,
        or a mystic distribution type built with mystic.math.Distribution.
        if dist=N, where N is an int or float, use normalized Gaussian noise,
        mystic.math.Distribution(numpy.random.normal, 0, sigma), where
        sigma is N * sum(bound) for each bound in the bounds, and N scales
        the amplitude of the noise (typically, N ~ 0.05).

    NOTE:
        if pts is a string (i.e. pts='4'), use solver-directed sampling.
        initial points are chosen by the sampler, then solvers run
        until converged. a LatticeSampler also accepts a list of pts,
        indicating the number of bins on each axis; again, if pts is a
        string (i.e. pts='[2,2,1]'), then use solver-directed sampling.
        if the string for pts begins with a '.', then only return the
        termini of the solver trajectories (i.e. pts='.4'). if the string
        begins with a '-' (i.e. pts='-4'), then only search for minima,
        or if the string begins with a '+', then only search for maxima.

    NOTE:
        if the model does not have an associated cached archive, sampling
        will create an archive used for the current sampling (and will not
        be attached to the model as a cached archive); a klepto.dir_archive
        will be created using the name of the model, unless an archive is
        otherwise specified using the archive keyword.
    """
    from mystic.samplers import LatticeSampler
    searcher = kwds.pop('sampler', LatticeSampler)
    ax = getattr(model, '__axis__', None)
    axis = None if hasattr(ax, '__len__') else ax # get default for axis
    axis = kwds.pop('axis', axis) # allow override?
    ny = kwds.pop('ny', getattr(model, 'ny', None)) #XXX: best?
    mvl = ny is not None # True if multivalued
    axis = axis if mvl else None #XXX: allow multi-axis search?
    dist = kwds.pop('dist', None)
    from numbers import Integral
    if isinstance(dist, (float, Integral)): # noise N(0,sig); sig = dist*(ub+lb)
        import numpy as np
        from mystic.math import Distribution
        sig = [dist * (ub+lb) for (lb,ub) in bounds] #FIXME: allow None and inf
        dist = Distribution(np.random.normal, 0, sig)
    pmap = kwds.pop('map', None)
    axmap = kwds.pop('axmap', None) #NOTE: was _ThreadPool.map w/ join
    if pmap is None: pmap = map
    if axmap is None: axmap = map
    name = kwds.pop('archive', None) # allow override? (elif below)
    if not hasattr(model, '__cache__') or not hasattr(model, '__inverse__'):
        import mystic.cache as mc
        if name is None: name = getattr(model, '__name__', None)
        model = mc.cached(archive=name, multivalued=mvl)(model)
    elif name is not None:
        model = getattr(model, '__orig__', model)
        import mystic.cache as mc
        model = mc.cached(archive=name, multivalued=mvl)(model)
    cache = model.__cache__
    imodel = model.__inverse__
    from mystic.math.legacydata import dataset
    # specification of first-order and second-order solve, or sampling
    deriv = False
    traj = mins = maxs = True

    if isinstance(pts, str):
        # special case starting values
        if pts[0] == '-': # mins only
            maxs = False
            pts = pts[1:]
        elif pts[0] == '+': # maxs only
            mins = False
            pts = pts[1:]
        if pts[0] == '.': # termini only
            traj = False
            pts = pts[1:]
        # check for second-order search
        if ':' in pts: # include 2nd-order
            pts,deriv = pts.split(':', 1)
            # handle special cases
            if not deriv: deriv = None # 2 * pts
        # handle special cases
        if not pts: pts = None
        pts = [pts] if deriv is False else [pts,deriv]

        # convert to non-string
        for i,s in enumerate(pts):
            if s is None: # non-strings
                continue
            try: # strings of ints
                pts[i] = int(s)
                if pts[i] < 0: # negative solvers should error
                    pts[i] = 'invalid value for number of solvers: %s' % s
                else:
                    pts[i] *= -1 #NOTE: legacy: solver should be negative
            except ValueError:
                from string import ascii_letters # no letters or decimals
                if set(ascii_letters+'.').intersection(s): #XXX: screen more?
                    pts[i]='invalid specification in number of solvers: %s' % s
                    continue
                try:
                    pts[i] = eval(s, {}, {})
                    import numpy as np
                    if np.min(pts[i]) < 0: # must be a sequence or int
                        pts[i] = 'invalid value in number of solvers: %s' % s
                    else:
                        pts[i][0] *= -1 #NOTE: legacy: solver should be negative
                except:
                    pts[i]='invalid specification of number of solvers: %s' % s
    else:
        pts = [pts]

    #NOTE: by here, pts[i] is either an int, list, or None
    for i,pt in enumerate(pts):
        if isinstance(pt, str):
            raise ValueError(pt)
        if hasattr(pt, '__len__'):
            import numpy as np
            neg = np.min(pt) < 0
            pt, _pt = np.prod(pt), [abs(p) for p in pt]
            if neg: pt = -abs(pt)
        else:
            _pt = None
        pts[i] = (pt,_pt)

    #NOTE: by here, pts[i] is a tuple of (int or None, list or None)
    if deriv is False:
        _deriv = False
    else:
        deriv, _deriv = pts[1]
    pts, _pts = pts[0]

    # handle special cases (i.e. the defaults)
    if pts is None: pts = -1
    if deriv is None: deriv = pts*2

    if deriv is not False:
        msg = '1st-order: %s and non-zero 2nd-order: %s' % (-pts, -deriv)
        raise NotImplementedError(msg)

    get_evals = False
    if traj: #HACK: workaround dict_archive doesn't cache
        archive = cache()
        from klepto._archives import dict_archive
        if type(archive) is dict_archive:
            from mystic.monitors import Null, Monitor
            mon = kwds.get('evalmon', None)
            if isinstance(mon, (type(None), Null)):
                kwds['evalmon'] = Monitor()
            get_evals = True

    if pts == 0: # don't sample, just grab the archive
        if not traj:
            dset = dataset() # = []
    elif pts > 0: # sample pts without optimizing
        pts = pts if _pts is None else _pts
        def doit(axis=None):
            _model = lambda x: model(x, axis=axis)
            s = searcher(bounds, _model, npts=pts, dist=dist, **kwds)
            s.sample()
            return s
        def dont(axis=None):
            _model = lambda x: model(x, axis=axis)
            s = searcher(bounds, _model, npts=pts, dist=dist, **kwds)
            s._sampler._penalty = lambda x: [] #XXX: removes the penalty
            s._sampler._bootstrap_objective()
            return s
        if mvl and axis is None:
            if ny and get_evals:
                dset = [dont(axis)] #HACK: dict_archive
                traj = False #XXX: causes to circumvent solver below
            else:
                dset = [doit(axis=0)]
        else:
            dset = [doit(axis)]
        # build a dataset of the samples #FIXME: check format when ny != None
        if not traj:
            _ID_ = dset[0]._sampler.id or 0
            if ny and get_evals and mvl and axis is None: #HACK: dict_archive
                _X_ = dset[0]._sampler._InitialPoints()
                _F_ = dset[0]._sampler._cost[0] or dset[0]._sampler._cost[1]
                dset = dataset().load(
                    [list(i) for i in _X_],
                    [_F_(i) for i in _X_],
                    [i for i,j in enumerate(_X_, _ID_)] #XXX: ? for ordering?
                )
            else:
                _X_ = dset[0]._sampler._all_bestSolution
                _F_ = dset[0]._sampler._all_bestEnergy
                dset = dataset().load(
                    [list(i) for i in _X_],
                    _F_,
                    [i for i,j in enumerate(_X_, _ID_)] #XXX: ? for ordering?
                )
            del _X_,_ID_,_F_
        elif get_evals: #HACK: dict_archive
            _ID_ = dset[0]._sampler.id or 0
            traj = False
            from itertools import chain
            dset = dataset().load(
                chain.from_iterable([i._evalmon.x for i in dset[0]._sampler._allSolvers]),
                chain.from_iterable([i._evalmon.y for i in dset[0]._sampler._allSolvers]),
                [i for i,j in enumerate(dset[0]._sampler._allSolvers, _ID_)] #XXX: ? for ordering?
            )
            del _ID_
    else: # search for minima until terminated
        pts = -pts if _pts is None else _pts
        def lower(axis=None):
            _model = lambda x: model(x, axis=axis)
            s = searcher(bounds, _model, npts=pts, dist=dist, **kwds)
            s.sample_until(terminated=all)
            return s
        def upper(axis=None):
            model_ = lambda x: imodel(x, axis=axis)
            si = searcher(bounds, model_, npts=pts, dist=dist, **kwds)
            si.sample_until(terminated=all)
            return si
        def _apply(f, arg):
            return f(arg)
        # search for mins, maxs, or both
        if mins is False:
            fs = upper,
        elif maxs is False:
            fs = lower,
        else:
            fs = lower, upper
        def doit(axis=None):
            return list(pmap(_apply, fs, [axis]*len(fs)))
        from itertools import chain #NOTE: flattens a list of lists
        if mvl and axis is None:
            if ny: #XXX: is the format correct? (i.e. flattened list)
                dset = list(chain.from_iterable(axmap(doit, range(ny))))
            else: #XXX: default to 0, warn, or error?
                dset = doit(axis=0)
        else:
            dset = doit(axis)
        # build a dataset of the termini #FIXME: check format when ny != None
        if not traj:
            if mins is False:
                dset = dataset().load(
                    chain.from_iterable([[list(i) for i in s._sampler._all_bestSolution] for s in dset]), 
                    chain.from_iterable([[-i for i in s._sampler._all_bestEnergy] for s in dset])
                )
            elif maxs is False:
                dset = dataset().load(
                    chain.from_iterable([[list(i) for i in s._sampler._all_bestSolution] for s in dset]), 
                    chain.from_iterable([s._sampler._all_bestEnergy for s in dset])
                )
            else:
                dset = dataset().load(
                    chain.from_iterable([[list(i) for i in s._sampler._all_bestSolution] for s in dset[0::2]] + [[list(i) for i in s._sampler._all_bestSolution] for s in dset[1::2]]), 
                    chain.from_iterable([s._sampler._all_bestEnergy for s in dset[0::2]] + [[-i for i in s._sampler._all_bestEnergy] for s in dset[1::2]])
                )
        #HACK: dict_archive
        elif get_evals:
            traj = False
            from itertools import chain
            if mins is False:
                dset = dataset().load(
                    chain.from_iterable([list(chain.from_iterable([i._evalmon.x for i in s._sampler._allSolvers])) for s in dset]), 
                    chain.from_iterable([list(chain.from_iterable([[-j for j in i._evalmon.y] for i in s._sampler._allSolvers])) for s in dset])
                )
            elif maxs is False:
                dset = dataset().load(
                    chain.from_iterable([list(chain.from_iterable([i._evalmon.x for i in s._sampler._allSolvers])) for s in dset]), 
                    chain.from_iterable([list(chain.from_iterable([i._evalmon.y for i in s._sampler._allSolvers])) for s in dset])
                )
            else:
                dset = dataset().load(
                    chain.from_iterable([list(chain.from_iterable([i._evalmon.x for i in s._sampler._allSolvers])) for s in dset[0::2]] + [list(chain.from_iterable([i._evalmon.x for i in s._sampler._allSolvers])) for s in dset[1::2]]), 
                    chain.from_iterable([list(chain.from_iterable([i._evalmon.y for i in s._sampler._allSolvers])) for s in dset[0::2]] + [list(chain.from_iterable([[-j for j in i._evalmon.y] for i in s._sampler._allSolvers])) for s in dset[1::2]])
                )
    if not traj: return dset #TODO: check format (vs below) when ny != None
    import dataset as ds
    return ds.from_archive(cache(), axis=None)


def _init_axis(model):
    """ensure axis is a keyword for the model

    Input:
        model: a function of the form y = model(x)

    Returns:
        a function of the form y = model(x, axis=None)
    """
    from klepto import signature
    signature = signature(model)[0]
    if type(model) is OUQModel or 'axis' in signature: #XXX: check len? kwds?
        return model
    # add axis (as second argument) #XXX: better utilize signature?
    def dummy(x, axis=None, **kwds): #XXX: *args?
        "model of form, y = model(x, axis=None)"
        result = model(x, **kwds)
        if axis is None or not hasattr(result, '__len__'):
            return result
        return result[axis]
    # copy attributes
    dummy.__orig__ = model #XXX: better name, __wrap__ ? 
    #dummy.__doc__ = model.__doc__ or "model of form, y = model(x, axis=None)"
    d = model.__dict__.items()
    dummy.__dict__.update((i,j) for (i,j) in d if i not in dummy.__dict__)
    return dummy


class OUQModel(object): #NOTE: effectively, this is WrapModel

    def __init__(self, id=None, **kwds): #XXX: take 'map' now in sample(map)?
        """base class for models to be used with OUQ classes

    Input:
        id: string, unique id for model instance [default: '']

    Additional Input:
        cached: bool, if True, use a mystic.cache [default: False]

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """ #FIXME: allow cached to take archive (instance or possibly name)
        #HACK: ok=True enables __init__ to be called (for super-like usage)
        if not kwds.pop('ok', False) or not hasattr(self, '__name__'):
            msg = 'use a derived class (e.g. WrapModel)'
            raise NotImplementedError(msg)
        self.__init_name()
        #FIXME: make sure model has 'axis' kwd here
        cached = kwds.pop('cached', False)
        if cached is False:
            self.__kwds__['cached'] = False
        else:
            self.__kwds__['cached'] = cached
            self.__init_cache()
        if not hasattr(self, '__func__'):
            self.__init_func()
        return

    def __init_cache(self):
        """ensure model has a mystic.cache"""
        model = self.__model__
        mvl = getattr(self, 'ny', getattr(model, 'ny', getattr(model, '__axis__', None))) is not None # True if multivalued
        archive = self.__kwds__.get('cached', True)
        name = getattr(model, '__name__', None) #XXX: do better?
        if not hasattr(model, '__cache__') or not hasattr(model, '__inverse__'):
            import mystic.cache as mc
            if archive is True: archive = name
            model = mc.cached(archive=archive, multivalued=mvl)(model)
        self.__model__ = model
        if name is not None:
            self.__model__.__name__ = name
        return

    def __init_name(self):
        """update name, potentially with UID"""
        if self.__name__ is None:
            self.__name__ = self.__model__.__name__
        if self.__kwds__.pop('uid', False):
            import numpy as np
            self.__name__ += ('_%s' % np.random.randint(1e16))
        self.__model__.__name__ = self.__name__
        return

    def __init_func(self):
        """add a function interface and axis"""
        from mystic.math.interpolate import _to_function
        self.__func__ = _to_function(self.__model__, ndim=self.nx)
        if hasattr(self.__func__, '__axis__'):
            self.__axis__ = self.__func__.__axis__
        else:
            if self.ny is None: # i.e. multivalued=False
                self.__axis__ = None
            else:
                def build_it(axis):
                    func = self.__func__
                    return (lambda *x, **kwds: func(*x, axis=axis, **kwds))
                self.__axis__ = [build_it(i) for i in range(self.ny)]
            self.__func__.__axis__ = self.__axis__
        return

    def __call__(self, x, axis=None, **kwds): #FIXME: cache as with sample
        """evaluate model at x, for the given axis, where y = model(x)

    Input:
        x: list of input parameters, where len(x) is the number of inputs
        axis: int, index of output to evaluate (all, by default)
        """
        return self.__model__(x, axis=axis, **kwds)

    #XXX: use self.bounds?
    def sample(self, bounds=None, pts=1, **kwds): 
        """sample model within bounds, writing to an archive and returning data

    Inputs:
        bounds: list of tuples of (lower,upper), bounds on each 'x'
        pts: int, number of points sampled by the sampler

    Additional Inputs:
        sampler: the mystic.sampler type [default: LatticeSampler]
        solver: the mystic.solver type [default: NelderMeadSimplexSolver]
        dist: a distribution type (or float amplitude) [default: None]
        map: map instance, to search for min/max in parallel [default: None]
        axis: int, index of output on which to search [default: 0]
        axmap: map instance, to execute each axis in parallel [default: None]
        archive: the archive (str name or archive instance) [default: None]
        multivalued: bool, True if output is multivalued [default: False]

    Returns:
        the mystic.math.legacydata.dataset of sampled data

    NOTE:
        additional keywords (evalmon, stepmon, maxiter, maxfun,
        saveiter, state, termination, constraints, penalty, reducer)
        are available for use. See mystic.ensemble for more details.

    NOTE:
        dist can be used to add randomness to the sampler, and can
        accept a numpy distribution type such as numpy.random.normal,
        or a mystic distribution type built with mystic.math.Distribution.
        if dist=N, where N is an int or float, use normalized Gaussian noise,
        mystic.math.Distribution(numpy.random.normal, 0, sigma), where
        sigma is N * sum(bound) for each bound in the bounds, and N scales
        the amplitude of the noise (typically, N ~ 0.05).

    NOTE:
        if pts is a string (i.e. pts='4'), use solver-directed sampling.
        initial points are chosen by the sampler, then solvers run
        until converged. a LatticeSampler also accepts a list of pts,
        indicating the number of bins on each axis; again, if pts is a
        string (i.e. pts='[2,2,1]'), then use solver-directed sampling.
        if the string for pts begins with a '.', then only return the
        termini of the solver trajectories (i.e. pts='.4'). if the string
        begins with a '-' (i.e. pts='-4'), then only search for minima,
        or if the string begins with a '+', then only search for maxima.

    NOTE:
        if the model does not have an associated cached archive, sampling
        will create an archive used for the current sampling (and will not
        be attached to the model as a cached archive); a klepto.dir_archive
        will be created using the name of the model, unless an archive is
        otherwise specified using the archive keyword.
        """
        model = self.__model__
        ax = getattr(self, '__axis__', getattr(model, '__axis__', None))
        axis = None if hasattr(ax, '__len__') else ax
        axis = kwds.pop('axis', axis) # allow override?
        ny = getattr(self, 'ny', getattr(model, 'ny', None))
        mvl = getattr(self, 'ny', getattr(model, 'ny', ax)) is not None # True if multivalued
        mvl = kwds.pop('multivalued', mvl) # allow override?
        kwds['axis'] = axis if mvl else None #XXX: allow multiaxis search?
        kwds['ny'] = ny if mvl else None
        #FIXME: anything needed to modify/prepare if kwds['archive'] ????
        return sample(model, bounds, pts=pts, **kwds)


    #XXX: np.sum, np.max ?
    def distance(self, data, axis=None, **kwds):
        """get graphical distance between function y=f(x) and a dataset

    Inputs:
      data: a mystic.math.legacydata.dataset of i points, M inputs, N outputs
      hausdorff: if True, use Hausdorff norm

    Additional Inputs:
      method: string for kind of interpolator
      maxpts: int, maximum number of points (x,z) to use from the monitor
      noise: float, amplitude of gaussian noise to remove duplicate x
      extrap: if True, extrapolate a bounding box (can reduce # of nans)
      arrays: if True, return a numpy array; otherwise don't return arrays
      axis: int in [0,N], index of z on which to interpolate (all, by default)

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').

    NOTE:
      data and function may provide tuple-valued or single-valued output.
      Distance will be measured component-wise, resulting in a tuple of
      distances, unless an 'axis' is selected. If an axis is selected,
      then return distance for the selected component (i.e. axis) only.
        """
        import dataset as ds
        return ds.distance(data, self.__func__, axis=axis, **kwds)


class NoisyModel(OUQModel):

    def __init__(self, id=None, model=None, mu=0, sigma=1, seed='!', **kwds):
        """noisy model, with Gaussian noise on inputs and/or outputs

    Input:
        id: string, unique id for model instance [default: 'noisy']
        model: a model function, of form y = model(x, axis=None)
        mu: input distribution mean value [default: 0]
        sigma: input distribution standard deviation [default: 1]
        seed: input random seed [default: '!', do not reseed the RNG]

    Additional Input:
        nx: int, number of model inputs, len(x) [default: None]
        ny: int, number of model outputs, len(y) [default: None]
        uid: bool, if True, append a random integer in [0, 1e16] to __name__
        cached: bool, if True, use a mystic.cache [default: False]
        zmu: output distribution mean value [default: 0]
        zsigma: output distribution standard deviation [default: 0]
        zseed: outut random seed [default: '!', do not reseed the RNG]

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        # get state defined in model
        if model is None:
            msg = 'a callable model, y = model(x), is required'
            raise NotImplementedError(msg)
        model = _init_axis(model) #FIXME: class method?
        uid = kwds.pop('uid', False)
        cached = kwds.pop('cached', False)
        self.nx = kwds.pop('nx', None) #XXX: None or 1?
        self.ny = kwds.pop('ny', None) #XXX: None or 1?
        args = dict(mu=mu, sigma=sigma, seed=seed)
        kwds['mu'] = kwds.pop('zmu', 0)
        kwds['sigma'] = kwds.pop('zsigma', 0)
        kwds['seed'] = kwds.pop('zseed', '!')
        def noisy(x, axis=None):
            """a noisy model, with Gaussian noise on inputs and/or outputs"""
            from noisy import noisy
            return noisy(model(noisy(tuple(x), **args), axis), **kwds)
        self.__model__ = noisy
        self.__name__ = self.__model__.__name__ if id is None else id
        self.__model__.__name__ = self.__name__
        kwd = self.__kwds__ = kwds.copy()
        self.__kwds__.update(args)
        def has_randomness(**kwds):
            if not bool(kwds.get('sigma', 1)) \
               and not bool(kwds.get('zsigma', 0)):
                return False
            from numbers import Integral
            if isinstance(kwds.get('seed', '!'), Integral) \
               and isinstance(kwds.get('zseed', '!'), Integral):
                return False
            return True
        self.rnd = has_randomness(**kwd)
        self.__kwds__['uid'] = uid
        #FIXME: ugly. this *must* be called *after* any subclass's init
        super(self.__class__, self).__init__(id, ok=True, cached=cached, **kwds)
        return

    def __call__(self, x, axis=None):
        """evaluate model at x, for the given axis, where y = model(x)

    Input:
        x: list of input parameters, where len(x) is the number of inputs
        axis: int, index of output to evaluate (all, by default)
        """
        return self.__model__(x, axis=axis)


class WrapModel(OUQModel):

    def __init__(self, id=None, model=None, **kwds):
        """a model object, to be used with OUQ classes

    Input:
        id: string, unique id for model instance [default: model.__name__]
        model: a model function, of form y = model(x, axis=None)

    Additional Input:
        nx: int, number of model inputs, len(x) [default: None]
        ny: int, number of model outputs, len(y) [default: None]
        rnd: bool, if False, treat the model as deterministic [default: True]
        uid: bool, if True, append a random integer in [0, 1e16] to __name__
        cached: bool, if True, use a mystic.cache [default: False]

    NOTE:
        any additional keyword arguments will be passed to 'model'

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        # get state defined in model
        if model is None:
            msg = 'a callable model, y = model(x), is required'
            raise NotImplementedError(msg)
        model = _init_axis(model) #FIXME: class method?
        self.nx = kwds.pop('nx', getattr(model, 'nx', None))
        self.ny = kwds.pop('ny', getattr(model, 'ny', None))
        self.rnd = kwds.pop('rnd', getattr(model, 'rnd', True))
        self.__model__ = model
        self.__name__ = self.__model__.__name__ if id is None else id
        self.__model__.__name__ = self.__name__
        self.__kwds__ = kwds.copy()
        #FIXME: ugly. this *must* be called *after* any subclass's init
        super(self.__class__, self).__init__(id, ok=True, **kwds)
        return

    def __call__(self, x, axis=None):
        """evaluate model at x, for the given axis, where y = model(x)

    Input:
        x: list of input parameters, where len(x) is the number of inputs
        axis: int, index of output to evaluate (all, by default)
        """
        kwds = self.__kwds__.copy()
        kwds.pop('cached', False) #XXX: don't pass cached to function
        return self.__model__(x, axis=axis, **kwds)


class SuccessModel(OUQModel):

    def __init__(self, id=None, model=None, **kwds):
        """a model of success, where success is model(x) >= cutoff

    Input:
        id: string, unique id for model instance [default: 'success']
        model: a model function, of form y = model(x, axis=None)

    Additional Input:
        nx: int, number of model inputs, len(x) [default: None]
        ny: int, number of model outputs, len(y) [default: None]
        rnd: bool, if False, treat the model as deterministic [default: True]
        uid: bool, if True, append a random integer in [0, 1e16] to __name__
        cached: bool, if True, use a mystic.cache [default: False]
        cutoff: float, defines success, where success is model(x) >= cutoff

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        if model is None:
            msg = 'a callable model, y = model(x), is required'
            raise NotImplementedError(msg)
        model = _init_axis(model) #FIXME: class method?
        self.nx = kwds.pop('nx', getattr(model, 'nx', None))
        self.ny = kwds.pop('ny', getattr(model, 'ny', None))
        cutoff = kwds.get('cutoff', 0.0)
        self.rnd = kwds.get('rnd', getattr(model, 'rnd', True))
        import numpy as np
        def success(x, axis=None):
            "a model of success, where success is model(x) >= cutoff"
            if axis is not None and hasattr(cutoff, '__len__'):
                return np.subtract(model(x, axis), cutoff[axis]) >= 0.0
            return np.all(np.subtract(model(x, axis), cutoff) >= 0.0)
        self.__model__ = success
        self.__name__ = self.__model__.__name__ if id is None else id
        self.__model__.__name__ = self.__name__
        self.__kwds__ = kwds.copy()
        self.__kwds__['model'] = model
        #FIXME: ugly. this *must* be called *after* any subclass's init
        super(self.__class__, self).__init__(id, ok=True, **kwds)
        return

    def __call__(self, x, axis=None):
        """evaluate model at x, for the given axis, where y = model(x)

    Input:
        x: list of input parameters, where len(x) is the number of inputs
        axis: int, index of output to evaluate (all, by default)
        """
        return self.__model__(x, axis=axis)


# interpolated model:
#  - interpolate G'(x) from data

class InterpModel(OUQModel):

    def __init__(self, id=None, data=None, **kwds):
        """an interpolated model, generated from the given data

    Input:
        id: string, unique id for model instance [default: 'interp']
        data: a mystic legacydata.dataset (or callable model, y = model(x))

    Additional Input:
        nx: int, number of model inputs, len(x) [default: None]
        ny: int, number of model outputs, len(y) [default: None]
        rnd: bool, if False, treat the model as deterministic [default: True]
        uid: bool, if True, append a random integer in [0, 1e16] to __name__
        cached: bool, if True, use a mystic.cache [default: False]

    NOTE:
        any additional keyword arguments will be passed to the interpolator;
        if a singular matrix is produced, try non-zero smooth or noise.

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        if data is None:
            from mystic.math.legacydata import dataset
            data = dataset()
            #msg = 'a mystic legacydata.dataset (or callable model) is required'
            #raise NotImplementedError(msg)
        if callable(data):
            data = _init_axis(data) #FIXME: class method?
        self.nx = kwds.pop('nx', None)
        self.ny = kwds.pop('ny', None) #FIXME: should rnd check noise?
        self.rnd = kwds.pop('rnd', True) and (bool(kwds.get('noise', True)))# or callable(data) or isinstance(data, type('')))
        self.__func__ = None
        def bootstrap(x, axis=None):
            "an interpolated model, generated from the given data"
            return self(x, axis=axis)
        self.__model__ = bootstrap
        self.__kwds__ = kwds.copy()
        self.__kwds__['data'] = data
        self.__name__ = 'interp' if id is None else id
        self.__model__.__name__ = self.__name__
        #self.fit() #NOTE: commented: lazy interpf, uncommented: interpf now
        #FIXME: ugly. this *must* be called *after* any subclass's init
        super(self.__class__, self).__init__(id, ok=True, **kwds)
        return

    def fit(self, **kwds):
        """generate an interpolated model from data

    Input:
        data: a mystic legacydata.dataset (or callable model, y = model(x))
        rnd: bool, if False, treat the model as deterministic [default: True]
        cached: bool, if True, use a mystic.cache [default: False]

    NOTE:
        any additional keyword arguments will be passed to the interpolator;
        if a singular matrix is produced, try non-zero smooth or noise.

    NOTE:
        if data is a model, interpolator will use model's cached archive

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        self.__kwds__.update(kwds)
        cached = self.__kwds__.pop('cached', False)
        archive = data = self.__kwds__.pop('data', None)
        self.rnd = self.__kwds__.pop('rnd', self.rnd) and (bool(self.__kwds__.get('noise', True)))# or callable(data) or isinstance(data, type('')))
        if callable(data): #XXX: is a model, allow this?
            data = sample(data, bounds=None, pts=0) #XXX: axis? multivalue?
        elif isinstance(data, type('')): #XXX: is a name, allow this?
            import mystic.cache as mc
            import dataset as ds
            data = ds.from_archive(mc.archive.read(data))
        x = getattr(data, 'coords', getattr(data, 'x', None))
        z = getattr(data, 'values', getattr(data, 'y', None))
        from interpolator import Interpolator
        terp = Interpolator(x, z, **self.__kwds__)
        self.__func__ = terp.Interpolate() #XXX: ValueError: zero-size
        self.__model__ = _init_axis(terp.model)
        self.__model__.__name__ = self.__name__
        self.__kwds__['data'] = archive
        if cached is False:
            self.__kwds__['cached'] = False
        else: #FIXME: clear the archive??? generate new uid name?
            self.__kwds__['cached'] = cached
            self._OUQModel__init_cache()
            if hasattr(self.__model__, '__cache__'):
                c = self.__model__.__cache__()
                c.clear()
        return

    def __call__(self, x, axis=None):
        """evaluate model at x, for the given axis, where y = model(x)

    Input:
        x: list of input parameters, where len(x) is the number of inputs
        axis: int, index of output to evaluate (all, by default)

    NOTE:
        generates an interpolated model, if one does not exist (or rnd=True)
        """
        if self.__func__ is None or self.rnd:
            self.fit()
        return self.__model__(x, axis=axis)


# learned model:
#  - learn G'(x) from data

class LearnedModel(OUQModel):

    def __init__(self, id=None, data=None, **kwds):
        """a learned model, trained on the given data

    Input:
        id: string, unique id for model instance [default: 'learn']
        data: a mystic legacydata.dataset (or callable model, y = model(x))

    Additional Input:
        nx: int, number of model inputs, len(x) [default: None]
        ny: int, number of model outputs, len(y) [default: None]
        rnd: bool, if False, treat the model as deterministic [default: True]
        uid: bool, if True, append a random integer in [0, 1e16] to __name__
        cached: bool, if True, use a mystic.cache [default: False]

    NOTE:
        any additional keyword arguments will be passed to the estimator

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        if data is None:
            from mystic.math.legacydata import dataset
            data = dataset()
            #msg = 'a mystic legacydata.dataset (or callable model) is required'
            #raise NotImplementedError(msg)
        if callable(data):
            data = _init_axis(data) #FIXME: class method?
        self.nx = kwds.pop('nx', None)
        self.ny = kwds.pop('ny', None) #FIXME: should rnd check noise?
        self.rnd = kwds.pop('rnd', True) and (bool(kwds.get('noise', False)))
        self.__func__ = None
        def bootstrap(x, axis=None):
            "a learned model, trained on the given data"
            return self(x, axis=axis)
        self.__model__ = bootstrap
        self.__kwds__ = kwds.copy()
        self.__kwds__['data'] = data
        self.__name__ = 'learn' if id is None else id
        self.__model__.__name__ = self.__name__
        #self.fit() #NOTE: commented: lazy learning, uncommented: learn now
        #FIXME: ugly. this *must* be called *after* any subclass's init
        super(self.__class__, self).__init__(id, ok=True, **kwds)
        return

    def fit(self, **kwds):
        """generate a learned model, trained on the given data

    Input:
        data: a mystic legacydata.dataset (or callable model, y = model(x))
        rnd: bool, if False, treat the model as deterministic [default: True]
        cached: bool, if True, use a mystic.cache [default: False]

    NOTE:
        any additional keyword arguments will be passed to the estimator

    NOTE:
        if data is a model, estimator will use model's cached archive

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        self.__kwds__.update(kwds)
        cached = self.__kwds__.pop('cached', False)
        archive = data = self.__kwds__.pop('data', None)
        self.rnd = self.__kwds__.pop('rnd', self.rnd) and (bool(self.__kwds__.get('noise', False)))
        if callable(data): #XXX: is a model, allow this?
            data = sample(data, bounds=None, pts=0) #XXX: axis? multivalue?
        elif isinstance(data, type('')): #XXX: is a name, allow this?
            import mystic.cache as mc
            import dataset as ds
            data = ds.from_archive(mc.archive.read(data))
        x = getattr(data, 'coords', getattr(data, 'x', None))
        z = getattr(data, 'values', getattr(data, 'y', None))
        from estimator import Estimator
        estm = Estimator(x, z, **self.__kwds__)
        self.__func__ = estm.Train() #XXX: Error for zero-size?
        self.__model__ = _init_axis(estm.model)
        self.__model__.__name__ = self.__name__
        self.__kwds__['data'] = archive
        if cached is False:
            self.__kwds__['cached'] = False
        else: #FIXME: clear the archive??? generate new uid name?
            self.__kwds__['cached'] = cached
            self._OUQModel__init_cache()
            if hasattr(self.__model__, '__cache__'):
                c = self.__model__.__cache__()
                c.clear()
        return

    def __call__(self, x, axis=None):
        """evaluate model at x, for the given axis, where y = model(x)

    Input:
        x: list of input parameters, where len(x) is the number of inputs
        axis: int, index of output to evaluate (all, by default)

    NOTE:
        generates a learned model, if one does not exist (or rnd=True)
        """
        if self.__func__ is None or self.rnd:
            self.fit()
        return self.__model__(x, axis=axis)


# workflow model:
#  - optimize G(d) -> G'(x) using graphical distance (and interp/learn model)

# error model:
#  - define cost as |F(x) - G'(x)|, given 'model' and 'surrogate'

class ErrorModel(OUQModel):

    def __init__(self, id=None, model=None, surrogate=None, **kwds):
        """an error model, with metric for distance from model to surrogate

    Input:
        id: string, unique id for model instance [default: 'learn']
        model: a model function, of form y = model(x, axis=None)
        surrogate: a function, y' = surrogate(x, axis=None), approximates model

    Additional Input:
        nx: int, number of model inputs, len(x) [default: None]
        ny: int, number of model outputs, len(y) [default: None]
        rnd: bool, if False, treat the model as deterministic [default: True]
        uid: bool, if True, append a random integer in [0, 1e16] to __name__
        cached: bool, if True, use a mystic.cache [default: False]
        metric: a function of form yerr = error(y, y')

    NOTE:
        the default metric is pointwise distance (y - y')**2

    NOTE:
        if cached is True, the default is to create a klepto.dir_archive
        using the name of the model; alternately, an archive can be specified
        by passing an archive instance (or string name) to the cached keyword.
        """
        if model is None or surrogate is None:
            msg = 'a callable model, and a callable surrogate, are required'
            raise NotImplementedError(msg)
        model = _init_axis(model) #FIXME: class method?
        surrogate = _init_axis(surrogate) #FIXME: class method?
        if not hasattr(model, 'distance'):
            model = WrapModel(model=model)
        nx = self.nx = kwds.pop('nx', getattr(model, 'nx', None))
        ny = self.ny = kwds.pop('ny', getattr(model, 'ny', None))
        if not hasattr(surrogate, 'distance'):
            surrogate = WrapModel(model=surrogate, nx=nx, ny=ny)
        self.rnd = kwds.pop('rnd', getattr(model, 'rnd', True))
        self.rnd = self.rnd or getattr(surrogate, 'rnd', True)
        metric = kwds.get('metric', None)
        if metric is None:
            import numpy as np
            def metric(x, y):
                "pointwise distance, |x - y|**2, from x to y"
                z = (np.abs(np.array(x) - y)**2).tolist()
                return tuple(z) if hasattr(z, '__len__') else z
        def error(x, axis=None):
            "an error model, with metric for distance from model to surrogate"
            return metric(model(x, axis=axis), surrogate(x, axis=axis))
        self.__model__ = error
        self.__name__ = self.__model__.__name__ if id is None else id
        self.__model__.__name__ = self.__name__
        self.__kwds__ = kwds.copy()
        self.__kwds__.update(dict(model=model, surrogate=surrogate))
        #FIXME: ugly. this *must* be called *after* any subclass's init
        super(self.__class__, self).__init__(id, ok=True, **kwds)
        return

    def __call__(self, x, axis=None):
        """evaluate model at x, for the given axis, where y = model(x)

    Input:
        x: list of input parameters, where len(x) is the number of inputs
        axis: int, index of output to evaluate (all, by default)
        """
        return self.__model__(x, axis=axis)

