#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
This module contains map and pipe interfaces to standard (i.e. serial) python.

Pipe methods provided:
    pipe        - blocking communication pipe             [returns: value]

Map methods provided:
    map         - blocking and ordered worker pool      [returns: list]
    imap        - non-blocking and ordered worker pool  [returns: iterator]


Usage
=====

A typical call to a pathos python map will roughly follow this example:

    >>> # instantiate and configure the worker pool
    >>> from pathos.serial import SerialPool
    >>> pool = SerialPool()
    >>>
    >>> # do a blocking map on the chosen function
    >>> print pool.map(pow, [1,2,3,4], [5,6,7,8])
    >>>
    >>> # do a non-blocking map, then extract the results from the iterator
    >>> results = pool.imap(pow, [1,2,3,4], [5,6,7,8])
    >>> print "..."
    >>> print list(results)
    >>>
    >>> # do one item at a time, using a pipe
    >>> print pool.pipe(pow, 1, 5)
    >>> print pool.pipe(pow, 2, 6)


Notes
=====

This worker pool leverages the built-in python maps, and thus does not have
limitations due to serialization of the function f or the sequences in args.
The maps in this worker pool have full functionality whether run from a script
or in the python interpreter, and work reliably for both imported and
interactively-defined functions.

"""
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
__all__ = ['SerialPool']

from .abstract_launcher import AbstractWorkerPool
__get_nodes__ = AbstractWorkerPool._AbstractWorkerPool__get_nodes
__set_nodes__ = AbstractWorkerPool._AbstractWorkerPool__set_nodes

from builtins import map as _map

class SerialPool(AbstractWorkerPool):
    """
Mapper that leverages standard (i.e. serial) python maps.
    """
    def map(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        return _map(f, *args)#, **kwds)
    map.__doc__ = AbstractWorkerPool.map.__doc__
    def imap(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        return _imap(f, *args)#, **kwds)
    imap.__doc__ = AbstractWorkerPool.imap.__doc__
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__pipe(self, f, *args, **kwds)
        return f(*args, **kwds)
    pipe.__doc__ = AbstractWorkerPool.pipe.__doc__
    #XXX: generator/yield provides simple ipipe? apipe? what about coroutines?
    ########################################################################
    # interface
    __get_nodes = __get_nodes__
    __set_nodes = __set_nodes__
    nodes = property(__get_nodes, __set_nodes)
    pass


# backward compatibility
PythonSerial = SerialPool

# EOF
