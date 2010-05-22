#!/usr/bin/env python

"""
Defaults for mapper and launcher. These should be
available as a minimal (dependency-free) pure-python
install from pathos::
    - serial_launcher:   syntax for standard python execution
    - python_map:        wrapper around the standard python map
    - carddealer_mapper: the carddealer map strategy
"""

defaults = { 'timelimit' : '00:05:00',
             'file' : '',
             'progname' : '',
             'outfile' : './results.out',
             'errfile' : './errors.out',
             'jobfile' : './jobid',
             'queue' : '',
             'python' : '`which python`' ,
             'nodes' : '1',
             'progargs' : ''
           }

def serial_launcher(kdict={}):
    """
prepare launch for standard execution
syntax:  (python) (file) (progargs)

NOTES:
    run non-python commands with: {'python':'', ...} 
    """
    mydict = defaults.copy()
    mydict.update(kdict)
    str = """ %(python)s %(file)s %(progargs)s""" % mydict
    return str

def python_map(func, *arglist, **kwds):
    """...

maps function 'func' across arguments 'arglist'.  Provides the
standard python map function, however also accepts **kwds in order
to conform with the pathos.pyina.map interface.

Further Input: [***disabled***]
    nnodes -- the number of parallel nodes
    launcher -- the launcher object
    mapper -- the mapper object
    timelimit -- string representation of maximum run time (e.g. '00:02')
    queue -- string name of selected queue (e.g. 'normal')
"""
   #print "ignoring: %s" % kwds  #XXX: should allow use of **kwds
    result = map(func, *arglist) #     see pathos.pyina.ez_map
    return result

def carddealer_mapper():
    """deal work out to all available resources,
then deal out the next new work item when a node completes its work """
    #from parallel_map import parallel_map as map
    #return map
    return "parallel_map"


if __name__=='__main__':
    f = lambda x:x**2
    print python_map(f,range(5),nnodes=10)

    import os
    d = {'progargs': """-c "print('hello')" """}
    os.system(serial_launcher(d))


# End of file
