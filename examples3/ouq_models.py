#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
build truth model, F'(x|a'), with selected a'
generate truth data

build "expensive" model of truth, F(x|a), with F ?= F' and a ?= a'

pick hyperparm A, inerpolate G'(x|A) from G(d), trained on "truth data"
(workflow: iterate interpolate to minimize graphical distance, for given A)
expected upper bound on f(x) = F(x|a) - G'(x|A)

1) F(x|a) = F'(x|a'). Tune A for optimal G.
2) F(x|a) != F'(x|a'). Tune A for optimal G, or "a" for optimal F.
3) |F(x|a) - F'(x|a')|, no G. Tune "a" for optimal F.
*) F(x|a) "is callable". Update "d" in G(d), then actively update G(x|A).
'''
from bounds import MeasureBounds

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
        dist: a distribution type [default: numpy.random.normal]
        map: a map instance [default: builtins.map]
        multivalued: bool, True if output is multivalued [default: False]

    Returns:
        the mystic.math.legacydata.dataset of sampled data

    NOTE:
        given the model is cached, a klepto.dir_archive is created by default

    NOTE:
        if pts is negative (i.e. pts=-4), use solver-directed sampling where
        initial points chosen by the sampler, then solvers run until converged

    NOTE:
        dist can be used to add randomness to the sampler
    """
    from mystic.samplers import LatticeSampler
    searcher = kwds.get('sampler', LatticeSampler)
    from mystic.solvers import NelderMeadSimplexSolver
    solver = kwds.get('solver', NelderMeadSimplexSolver)
    mvl = getattr(model, 'ny', getattr(model, '__axis__', None)) is not None
    mvl = kwds.get('multivalued', mvl) # allow override?
    import numpy as np
    dist = kwds.get('dist', np.random.normal)
    map_ = kwds.get('map', map)
    if not hasattr(model, '__cache__') or not hasattr(model, '__inverse__'):
        import mystic.cache as mc
        name = getattr(model, '__name__', None) #XXX: do better?
        model = mc.cached(archive=name, multivalued=mvl)(model)
    cache = model.__cache__
    imodel = model.__inverse__
    axis = 0 if mvl else None #FIXME: choice of '0' is fixed
    if hasattr(pts, '__len__'):
        pts, _pts = -np.prod(pts), pts
    else:
        _pts = None
    if pts is None: pts = -1
    if pts == 0: # don't sample, just grab the archive
        pass
    elif pts > 0: # sample pts without optimizing
        _model = lambda x: model(x, axis=axis)
        s = searcher(bounds, _model, npts=pts, solver=solver, dist=dist)
        s.sample()
    else: # search for minima until terminated
        pts = -pts if _pts is None else _pts
        #FIXME: iterate over axes?
        def lower(axis=None):
            _model = lambda x: model(x, axis=axis)
            s = searcher(bounds, _model, npts=pts, solver=solver, dist=dist)
            s.sample_until(terminated=all)
            return s
        def upper(axis=None):
            model_ = lambda x: imodel(x, axis=axis)
            si = searcher(bounds, model_, npts=pts, solver=solver, dist=dist)
            si.sample_until(terminated=all)
            return si
        def _apply(f, arg):
            return f(arg)
        fs = lower, upper
        list(map_(_apply, fs, [axis]*len(fs)))
    import dataset as ds
    return ds.from_archive(cache(), axis=None) #FIXME: multi


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
        result = model(x, **kwds)
        if axis is None or not hasattr(result, '__len__'):
            return result
        return result[axis]
    # copy attributes
    dummy.__orig__ = model #XXX: better name, __wrap__ ? 
    d = model.__dict__.items()
    dummy.__dict__.update((i,j) for (i,j) in d if i not in dummy.__dict__)
    return dummy


class OUQModel(object): #NOTE: effectively, this is WrapModel

    def __init__(self, id=None, **kwds):
        """base class for models to be used with OUQ classes

    Input:
        id: string, unique id for model instance [default: '']

    Additional Input:
        cached: bool, if True, use a mystic.cache [default: False]
        """
        #HACK: ok=True enables __init__ to be called (for super-like usage)
        if not kwds.pop('ok', False) or not hasattr(self, '__name__'):
            msg = 'use a derived class (e.g. WrapModel)'
            raise NotImplementedError(msg)
        self.__init_name()
        #FIXME: make sure model has 'axis' kwd here
        if kwds.pop('cached', False):
            self.__kwds__['cached'] = True
            self.__init_cache()
        if not hasattr(self, '__func__'):
            self.__init_func()
        return

    def __init_cache(self):
        """ensure model has a mystic.cache"""
        model = self.__model__
        mvl = getattr(self, 'ny', getattr(model, 'ny', getattr(model, '__axis__', None))) is not None # True if multivalued
        name = getattr(model, '__name__', None) #XXX: do better?
        if not hasattr(model, '__cache__') or not hasattr(model, '__inverse__'):
            import mystic.cache as mc
            model = mc.cached(archive=name, multivalued=mvl)(model)
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
        model = self.__model__
        mvl = getattr(self, 'ny', getattr(model, 'ny', getattr(model, '__axis__', None))) is not None # True if multivalued
        return sample(model, bounds, pts=pts, multivalued=mvl)

    #XXX: np.sum, np.max ?
    def distance(self, data, axis=None, **kwds):
        import dataset as ds
        return ds.distance(data, self.__func__, axis=axis, **kwds)


class NoisyModel(OUQModel):

    def __init__(self, id=None, model=None, mu=0, sigma=1, seed='!', **kwds):
        # get state defined in model
        if model is None:
            return NotImplemented
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
            if isinstance(kwds.get('seed', '!'), int) \
               and isinstance(kwds.get('zseed', '!'), int):
                return False
            return True
        self.rnd = has_randomness(**kwd)
        self.__kwds__['uid'] = uid
        #FIXME: ugly. this *must* be called *after* any subclass's init
        super(self.__class__, self).__init__(id, ok=True, cached=cached, **kwds)
        return

    def __call__(self, x, axis=None):
        return self.__model__(x, axis=axis)


class WrapModel(OUQModel):

    def __init__(self, id=None, model=None, **kwds):
        # get state defined in model
        if model is None:
            return NotImplemented
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
        return self.__model__(x, axis=axis, **self.__kwds__)


class SuccessModel(OUQModel):

    def __init__(self, id=None, model=None, **kwds):
        if model is None:
            return NotImplemented
        model = _init_axis(model) #FIXME: class method?
        self.nx = kwds.pop('nx', getattr(model, 'nx', None))
        self.ny = kwds.pop('ny', getattr(model, 'ny', None))
        cutoff = kwds.get('cutoff', 0.0)
        self.rnd = kwds.get('rnd', getattr(model, 'rnd', True))
        import numpy as np
        def success(x, axis=None):
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
        return self.__model__(x, axis=axis)


# interpolated model:
#  - interpolate G'(x) from data

class InterpModel(OUQModel):

    def __init__(self, id=None, data=None, **kwds):
        if data is None:
            return NotImplemented
        if callable(data):
            data = _init_axis(data) #FIXME: class method?
        self.nx = kwds.pop('nx', None)
        self.ny = kwds.pop('ny', None)
        self.rnd = kwds.pop('rnd', False) and (bool(kwds.get('noise', -1)))# or callable(data) or isinstance(data, type('')))
        self.__func__ = None
        def bootstrap(x, axis=None):
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
        self.__kwds__.update(kwds)
        cached = self.__kwds__.pop('cached', False)
        archive = data = self.__kwds__.pop('data', None)
        self.rnd = self.__kwds__.pop('rnd', self.rnd) and (bool(self.__kwds__.get('noise', -1)))# or callable(data) or isinstance(data, type('')))
        if callable(data): #XXX: is a model, allow this?
            data = sample(data, bounds=None, pts=0)
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
        if cached: #FIXME: clear the archive??? generate new uid name?
            self.__kwds__['cached'] = True
            self._OUQModel__init_cache()
            if hasattr(self.__model__, '__cache__'):
                c = self.__model__.__cache__()
                c.clear()
        return

    def __call__(self, x, axis=None):
        if self.__func__ is None or self.rnd:
            self.fit()
        return self.__model__(x, axis=axis)


# learned model:
#  - learn G'(x) from data

class LearnedModel(OUQModel):

    def __init__(self, id=None, data=None, **kwds):
        if data is None:
            return NotImplemented
        if callable(data):
            data = _init_axis(data) #FIXME: class method?
        self.nx = kwds.pop('nx', None)
        self.ny = kwds.pop('ny', None)
        self.rnd = kwds.pop('rnd', False) and (bool(kwds.get('noise', -1)))
        self.__func__ = None
        def bootstrap(x, axis=None):
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
        self.__kwds__.update(kwds)
        cached = self.__kwds__.pop('cached', False)
        archive = data = self.__kwds__.pop('data', None)
        self.rnd = self.__kwds__.pop('rnd', self.rnd) and (bool(self.__kwds__.get('noise', -1)))
        if callable(data): #XXX: is a model, allow this?
            data = sample(data, bounds=None, pts=0)
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
        if cached: #FIXME: clear the archive??? generate new uid name?
            self.__kwds__['cached'] = True
            self._OUQModel__init_cache()
            if hasattr(self.__model__, '__cache__'):
                c = self.__model__.__cache__()
                c.clear()
        return

    def __call__(self, x, axis=None):
        if self.__func__ is None or self.rnd:
            self.fit()
        return self.__model__(x, axis=axis)


# workflow model:
#  - optimize G(d) -> G'(x) using graphical distance (and interp/learn model)

# error model:
#  - define cost as |F(x) - G'(x)|, given 'model' and 'surrogate'

class ErrorModel(OUQModel):

    def __init__(self, id=None, model=None, surrogate=None, **kwds):
        if model is None or surrogate is None:
            return NotImplemented
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
                z = (np.abs(np.array(x) - y)**2).tolist()
                return tuple(z) if hasattr(z, '__len__') else z
        def error(x, axis=None):
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
        return self.__model__(x, axis=axis)

