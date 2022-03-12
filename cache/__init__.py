#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2013-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''decorators for caching function outputs, with function inputs as the keys,
and interactors for reading and writing to databases of functions and data.

several klepto.cache strategies are included here for backward compatability;
please also see the klepto package for available archives (i.e. dir_archive,
file_archive, hdf_archive, sql_archive), cache strategies, filters, encodings,
serialization, and other configuration options
'''
from . import archive
from . import function

# backward compatability
from klepto import lru_cache, lfu_cache, mru_cache
from klepto import rr_cache, inf_cache, no_cache

class _cached(object):
    def __init__(self, func, multivalued=False):
        self.func = func
        self.multivalued = multivalued
    def mvmodel(self, x, *argz, **kwdz):
        axis = kwdz.pop('axis', None)
        if axis is None: axis = slice(None)
        return self.func(x, *argz, **kwdz)[axis]
    def model(self, x, *argz, **kwdz):
        axis = kwdz.pop('axis', None)
        return self.func(x, *argz, **kwdz)
    def __call__(self, x, *argz, **kwdz):
        doit = self.mvmodel if self.multivalued else self.model
        return doit(x, *argz, **kwdz)

class _cache(object):
    def __init__(self, func):
       self.func = func
    def __call__(self):
       return self.func.__cache__()

class _imodel(object):
    def __init__(self, model):
       self.model = model
    def __call__(self, *args, **kwds):
        return -self.model(*args, **kwds)


def cached(**kwds):
    """build a caching archive for an objective function

    Input:
      type: the type of klepto.cache [default: inf_cache]
      archive: the archive (str name for a new archive, or archive instance)
      maxsize: maximum cache size [default: None]
      keymap: cache key encoder [default: klepto.keymap.keymap()]
      ignore: function argument names to ignore [default: '**']
      tol: int tolerance for rounding [default: None]
      deep: bool rounding depth (default: False]
      purge: bool for purging cache to archive [default: False]
      multivalued: bool if multivalued return of objective [default: False]

    Returns: 
      cached objective function

    Notes:
      inverse (y = -objective(x)) is at objective.__inverse__
      cache of objective is at objective.__cache__
      inverse and objective cache to the same archive
    """ #FIXME: ignore not handle *args as expected due to *args HACK
    _type = kwds.pop('type', inf_cache)
    multivalued = kwds.pop('multivalued', False)
    from klepto.keymaps import keymap as _keymap
    db = kwds.pop('archive', None)
    kwds.setdefault('keymap', _keymap())
    kwds.setdefault('ignore', '**') #FIXME: also ignores '*'
    if db is None: 
        kwds['cache'] = archive.read('archive')
    elif type(db) in (str, (u''.__class__)):
        kwds['cache'] = archive.read(db)
    else:
        kwds['cache'] = db
    # produce a cache with an archive backend
    cache = _type(**kwds)

    def dec(objective):
        """wrap a caching archive around an objective
        """
        # use a secretkey to store the objective's args
        SECRETKEY = '*' #XXX: too simple? #FIXME: ignore w/ignore='*', not names
        # wrap the cache around the objective function
        @cache
        def inner(*x, **kwds):
            args = kwds.pop(SECRETKEY, ())
            return objective(x, *args, **kwds)
        def _model(x, *args, **kwds):
            if args: kwds[SECRETKEY] = args
            return inner(*x, **kwds)
        _model.__inner__ = inner

        # when caching, always cache the multi-valued tuple
        model = _cached(_model, multivalued)
        # produce objective function that caches multi-valued output
        model.__cache__ = _cache(inner)
        model.__doc__ = objective.__doc__

        # produce model inverse with shared cache
        model.__inverse__ = imodel = _imodel(model)
        imodel.__inverse__ = model
        imodel.__cache__ = model.__cache__

        return model
    return dec

