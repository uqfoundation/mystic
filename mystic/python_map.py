#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Defaults for mapper and launcher. These should be available as a minimal
(dependency-free) pure-python install from ``pathos``::

    serial_launcher -- syntax for standard python execution
    python_map      -- wrapper around the standard python map
    worker_pool     -- the worker_pool map strategy
"""

import os
_pid = '.' + str(os.getpid()) + '.'
defaults = {
    'nodes' : '1',
    'program' : '',
    'python' : '`which python`' ,
    'progargs' : '',

    'outfile' : 'results%sout' % _pid,
    'errfile' : 'errors%sout' % _pid,
    'jobfile' : 'job%sid' % _pid,

    'scheduler' : '',
    'timelimit' : '00:02',
    'queue' : '',

    'workdir' : '.'
    }

def serial_launcher(kdict={}):
    """
prepare launch for standard execution
syntax:  (python) (program) (progargs)

Notes:
    run non-python shell commands by setting ``python`` to a null string: 
    ``kdict = {'python':'', ...}``
    """
    mydict = defaults.copy()
    mydict.update(kdict)
    str = """ %(python)s %(program)s %(progargs)s""" % mydict
    return str

def python_map(func, *arglist, **kwds):
    """maps function *func* across arguments *arglist*.

Provides the standard python map function, however also accepts *kwds* in
order to conform with the (deprecated) ``pyina.ez_map`` interface.

Notes:
    The following *kwds* used in ``ez_map`` are accepted, but disabled:
        * nodes -- the number of parallel nodes
        * launcher -- the launcher object
        * scheduler -- the scheduler object
        * mapper -- the mapper object
        * timelimit -- string representation of maximum run time (e.g. '00:02')
        * queue -- string name of selected queue (e.g. 'normal')
"""
   #print("ignoring: %s" % kwds)  #XXX: should allow use of **kwds
    result = list(map(func, *arglist)) #     see pathos.pyina.ez_map
    return result

def worker_pool():
    """use the 'worker pool' strategy; hence one job is allocated to each
worker, and the next new work item is provided when a node completes its work
"""
    #from mpi_pool import parallel_map as map
    #return map
    return "mpi_pool"

# backward compatibility
carddealer_mapper = worker_pool

del os


if __name__=='__main__':
    f = lambda x:x**2
    print(python_map(f,range(5),nodes=10))

    import subprocess
    d = {'progargs': """-c "print('hello')" """}
    subprocess.call(serial_launcher(d), shell=True)


# End of file
