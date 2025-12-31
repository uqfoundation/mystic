#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2026 The Uncertainty Quantification Foundation.
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


def read(archives, keymap=None, type=None, n=0): # method?
    """read stored functions from the list of dbs

    Args:
        archives (list[string]): list of names of function archives
        keymap (klepto.keymap): keymap used for key encoding
        type (klepto.archive): type of klepto archive
        n (int): db entry in reverse order (i.e. most recent is ``0``)

    Returns:
        a klepto.archive instance

    Notes:
        The order of the dbs is important, with the index of ``archives`` 
        corresponding to the desired axis. If a db is empty, returns ``None``
        for the empty db. Also, a klepto.archive instance can be provided
        instead of the ``name`` of the db.
    """ #XXX: n as tuple for each axis?
    from builtins import type as _type
    if isinstance(archives, (str, (u'').__class__)):
        archives = db(archives)
    if _type(archives).__module__.startswith('klepto.'):
        f = rb.read_func(db(archives), keymap=keymap, type=type, n=n)
        if f is None:
            return f
        f = f[0]
        if not hasattr(f, '__axis__'):
            f.__axis__ = None
        return f
    from mystic.math.interpolate import interpf
    f = interpf([[1,2],[2,3]],[[1,2],[2,3]],method='thin_plate')
    read_func = lambda x: rb.read_func(x, keymap=keymap, type=type, n=n)
    f.__axis__[:] = [(i if i is None else i[0]) for i in map(read_func, map(db, archives))]
    return f

