#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''higher-level readers and writers for stored functions
'''

__all__ = ['db','write','read']

from . import archive as rb


def db(name):
    """get function db with the name 'name'"""
    if not isinstance(name, (str, (u'').__class__)):
        return getattr(name, '__archive__', name) # need cached == False
    return rb._read_func(name) #, type=type)


def write(function, archives):
    """write function to corresponding archives"""
    if isinstance(archives, (str, (u'').__class__)):
        archives = db(archives)
    if getattr(function, '__axis__', None) is None: #XXX: and no len(archives)
        rb._write_func(archives, function, {})
    else:
        [rb._write_func(a, f, {}) for a,f in zip(archives,function.__axis__)]


def read(archives): # method?
    """read stored functions from the list of dbs

    Args:
        archives (list[string]): list of names of function archives

    Returns:
        a klepto.archive instance

    Notes:
        The order of the dbs is important, with the index of ``archives`` 
        corresponding to the desired axis. If a db is empty, returns ``None``
        for the empty db. Also, a klepto.archive instance can be provided
        instead of the ``name`` of the db.
    """
    if isinstance(archives, (str, (u'').__class__)):
        archives = db(archives)
    if type(archives).__module__.startswith('klepto.'):
        f = rb.read_func(db(archives))
        if f is None:
            return f
        f = f[0]
        if not hasattr(f, '__axis__'):
            f.__axis__ = None
        return f
    from mystic.math.interpolate import interpf
    f = interpf([[1,2],[2,3]],[[1,2],[2,3]],method='thin_plate')
    f.__axis__[:] = [(i if i is None else i[0]) for i in map(rb.read_func, map(db, archives))]
    return f

