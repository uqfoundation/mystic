#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''klepto archive readers and writers, for functions and data
'''
from klepto.archives import (dict_archive, dir_archive, file_archive,
                             hdf_archive, hdfdir_archive, null_archive,
                             sql_archive, sqltable_archive)

def read(name, keys=None, keymap=None, type=None):
    """read klepto db with name 'name'

    Args:
        name (string): filename of the klepto db
        keys (iterable): keys to load; or boolean to load all/no keys
        keymap (klepto.keymap): used for key encoding
        type (klepto.archive): type of klepto archive

    Returns:
        klepto db object (has dictionary-like interface)

    Notes:
        If keys is None, create a direct handle to the db.
        If a key in keys is not found, it will be ignored.
    """
    if type is None:
        from klepto.archives import dir_archive as _archive
    else:
        _archive = type
    if keys is None:
        return _archive(name, cached=False)
    if keymap is None:
        from klepto.keymaps import keymap as _kmap
        keymap = _kmap()
    archive = _archive(name, cached=True)
    if keys is False:
        return archive
    if keys is True:
        archive.load()
    else: # apply keymap to keys
        archive.load(*(keymap(*(k if hasattr(k, '__len__') else (k,))) for k in keys))
    return archive # archive with keymapped keys


def write(archive, entries, keymap=None):
    """write entries to klepto db instance 'archive'

    Args:
        archive (klepto.archive): archive instance
        entries (iterable): iterable/dict to update the archive with
        keymap (klepto.keymap): keymap used for key encoding

    Returns:
        None

    Examples:
        >>> write(foo, {x0:y0, x1:y1, ...})
        >>> write(foo, ((x0,y0), (x1,y1), ...))

    Notes:
        Within ``entries``, ``xi`` is a tuple of floats, and ``yi`` is a float.
    """
    if isinstance(archive, (str, (u'').__class__)):
        archive = read(archive, keymap=keymap) # assumes default type
    if keymap is not None: # apply keymap to keys
        entries = {keymap(*(k if hasattr(k, '__len__') else (k,))):v for k, v in entries.items()}
    # archive = read(name, keys=(False if cached else None))
    cached = archive.archive is not archive
    archive.update(entries) #XXX: for non-cached, __setitem__ is faster?
    if cached: archive.dump()
    return


def _read_func(name, keymap=None, type=None):
    """read function db with name 'name'

    Args:
        name (string): filename of the klepto db
        keymap (klepto.keymap): keymap used for key encoding
        type (klepto.archive): type of klepto archive

    Returns:
        klepto db object (has dictionary-like interface)
    """
    return read(name, keymap=keymap, type=type)


def read_func(name, keymap=None, type=None, n=0):
    """read stored function from db with name 'name'

    Args:
        name (string): filename of the klepto db
        keymap (klepto.keymap): keymap used for key encoding
        type (klepto.archive): type of klepto archive
        n (int): db entry in reverse order (i.e. most recent is ``0``)

    Returns:
        tuple of (stored function, distance information)

    Notes:
        If the db is empty, or ``n`` produces a bad index, returns ``None``.
        Alternately, ``name`` can be the relevant klepto.archive instance.
    """
    if not isinstance(name, (str, (u'').__class__)):
        if type is not None:
           #msg = 'if a klepto.archive instance is provided, type must be None'
           #raise TypeError(msg)
           type = None #NOTE: ignore type
        archive = getattr(name, '__archive__', name) # need cached == False
    else:
        archive = _read_func(name, type=type) # read entire archive to get size
    size = len(archive) - n - 1
    entry = size if keymap is None else keymap(size)
    return archive.get(entry, None) # apply keymap


def _write_func(archive, func, dist, keymap=None):
    """write stored function and distance information to archive

    Args:
        archive (klepto.archive): function archive (output of ``_read_func``)
        func (function): with interface ``y = f(x)``, ``x`` is a list of floats
        dist (dict): distance information
        keymap (klepto.keymap): keymap used for key encoding

    Returns:
        None
    """
    arxiv = getattr(archive, '__archive__', archive)
    entries = {len(arxiv): (func, dist)}
    write(archive, entries, keymap=keymap) # apply keymap to keys
    return


def get_dist(archive, func, keymap=None):
    """get the graphical distance of func from data in archive

    Args:
        archive (klepto.archive): run archive (output of ``read``)
        func (function): with interface ``y = f(x)``, ``x`` is a list of floats
        keymap (klepto.keymap): keymap used for key encoding

    Returns:
        array of floats, graphical distance from func to each point in archive
    """
    from mystic.math.legacydata import dataset
    data = dataset()
    arxiv = getattr(archive, '__archive__', archive)
    if not len(arxiv):
        return
    #FIXME: need to implement generalized inverse of keymap
    #HACK: only works in certain special cases
    kind = getattr(keymap, '__stub__', '')
    if kind in ('encoding', 'algorithm'): #FIXME always assumes used repr
        inv = lambda k: eval(k)
    elif kind in ('serializer', ): #FIXME: ignores all config options
        import importlib
        inv = lambda k: importlib.import_module(keymap.__type__).loads(k)
    else: #FIXME: give up, ignore keymap
        inv = lambda k: k
    y = ((inv(k), v) for k, v in arxiv.items())
    #HACK end
    import numpy as np
    y = np.array(list(y), dtype=object).T
    data.load(y[0].tolist(), y[1].tolist(), ids=list(range(len(y[0]))))
    from mystic.math.distance import graphical_distance
    return graphical_distance(func, data)


def _prep_dist(distances):
    """create a dict of mean and max graphical distance

    Args:
        distances (array[float]): indicates the graphical distance

    Returns:
        ``dict(mean=distances.mean(), max=distances.max())``
    """
    if distances is None:
        import numpy as np
        return dict(mean=np.nan, max=np.nan)
    return dict(mean=distances.mean(), max=distances.max())


'''
### tests ###
def _test(fundb, sql=False):
    """read function database, returning tuple of (function, distance)

    Args:
        fundb (string): function db name
        sql (bool): True if fundb is a ``sql_archive`` (w/ ``stringmap``)

    Returns:
        tuple of (stored function, distance information)
    """
    if sql:
        from klepto.keymaps import stringmap as _keymap
        from klepto.archives import sql_archive as ar
        km = _keymap()
    else:
        km = ar = None
        from klepto.archives import hdf_archive as ar
    return read_func(fundb, keymap=km, type=ar)



def _make_test(func, rundb, fundb, runsql=False, funsql=False):
    """create/increment function sql database

    Args:
        func (function): with interface ``y = f(x)``, ``x`` is a list of floats
        rundb (string): name of the klepto db archive of runs
        fundb (string): name of the klepto function archive
        runsql (bool): True if rundb is a ``sql_archive`` (w/ ``stringmap``)
        funsql (bool): True if fundb is a ``sql_archive`` (w/ ``stringmap``)

    Returns:
        None
    """
    if runsql:
        from klepto.keymaps import stringmap as _keymap
        from klepto.archives import sql_archive as r_ar
        r_km = _keymap()
    else:
        r_km = r_ar = None
    if funsql:
        from klepto.keymaps import stringmap as _keymap
        from klepto.archives import sql_archive as f_ar
        f_km = _keymap()
    else:
        f_km = f_ar = None
        from klepto.archives import hdf_archive as f_ar
    db = read(rundb, keymap=r_km, type=r_ar)
    dist = _prep_dist(get_dist(db, func, keymap=r_km))
    db = _read_func(fundb, keymap=f_km, type=f_ar)
    _write_func(db, func, dist, keymap=f_km)
    return


def _dump(obj, path):
    """serialize obj to file located at path

    Args:
        obj (object): any python object
        path (string): path of new pickle file
    """
    import dill
    dill.dump(obj, open(path, 'wb'))


def _load(path):
    """load serializade obj from file located at path

    Args:
        path (string): path of existing pickle file
    """
    import dill
    return dill.load(open(path, 'rb'))
'''
