#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from __future__ import with_statement
import os
import sys

# set version numbers
stable_version = '0.2a1'
target_version = '0.2a2'
is_release = False

# check if easy_install is available
try:
#   import __force_distutils__ #XXX: uncomment to force use of distutills
    from setuptools import setup
    has_setuptools = True
except ImportError:
    from distutils.core import setup
    has_setuptools = False

# generate version number
if os.path.exists('mystic/info.py'):
    # is a source distribution, so use existing version
   #from mystic.info import this_version #FIXME?
    this_version = stable_version
elif stable_version == target_version:
    # we are building a stable release
    this_version = stable_version
else:
    # we are building a distribution
    this_version = target_version + '.dev'
    if is_release:
      from datetime import date
      today = "".join(date.isoformat(date.today()).split('-'))
      this_version += "-" + today

# get the license info
with open('LICENSE') as file:
    license_text = file.read()

# generate the readme text
long_description = \
"""------------------------------------------------------
mystic: a simple model-independent inversion framework
------------------------------------------------------

The mystic framework provides a collection of optimization algorithms
and tools that allows the user to more robustly (and readily) solve
optimization problems. All optimization algorithms included in mystic
provide workflow at the fitting layer, not just access to the algorithms
as function calls. Mystic gives the user fine-grained power to both
monitor and steer optimizations as the fit processes are running.

Where possible, mystic optimizers share a common interface, and thus can
be easily swapped without the user having to write any new code. Mystic
solvers all conform to a solver API, thus also have common method calls
to configure and launch an optimization job. For more details, see
`mystic.abstract_solver`. The API also makes it easy to bind a favorite
3rd party solver into the mystic framework.

By providing a robust interface designed to allow the user to easily
configure and control solvers, mystic reduces the barrier to implementing
a target fitting problem as stable code. Thus the user can focus on
building their physical models, and not spend time hacking together an
interface to optimization code.

Mystic is in the early development stages, and any user feedback is
highly appreciated. Contact Mike McKerns [mmckerns at caltech dot edu]
with comments, suggestions, and any bugs you may find.  A list of known
issues is maintained at http://dev.danse.us/trac/mystic/query.


Major Features
==============

Mystic provides a stock set of configurable, controllable solvers with::

    - a common interface
    - the ability to impose solver-independent bounds constraints
    - the ability to apply solver-independent monitors
    - the ability to configure solver-independent termination conditions
    - a control handler yielding: [pause, continue, exit, and user_callback]
    - ease in selecting initial conditions: [initial_guess, random]
    - ease in selecting mutation strategies (for differential evolution)

To get up and running quickly, mystic also provides infrastructure to::

    - easily generate a fit model (several example models are included)
    - configure and auto-generate a cost function from a model
    - extend fit jobs to parallel & distributed resources
    - couple models with optimization parameter constraints [COMING SOON]


Current Release
===============

The latest stable release version is mystic-%(relver)s. You can download it here.
The latest stable version of mystic is always available at:

    http://dev.danse.us/trac/mystic


Development Release
===================

If you like living on the edge, and don't mind the promise
of a little instability, you can get the latest development
release with all the shiny new features at:

    http://dev.danse.us/packages.


Installation
============

Mystic is packaged to install from source, so you must
download the tarball, unzip, and run the installer::

    [download]
    $ tar -xvzf mystic-%(thisver)s.tgz
    $ cd mystic-%(thisver)s
    $ python setup py build
    $ python setup py install

You will be warned of any missing dependencies and/or settings
after you run the "build" step above. Mystic depends on dill, numpy
and sympy, so you should install them first. There are several
functions within mystic where scipy is used if it is available;
however, scipy is an optional dependency. Having matplotlib installed
is necessary for running several of the examples, and you should
probably go get it even though it's not required. Matplotlib is
also required by mystic's "analysis viewers".

Alternately, mystic can be installed with easy_install::

    [download]
    $ easy_install -f . mystic

For Windows users, source code and examples are available in zip format.
A binary installer is also provided::

    [download]
    [double-click]


Requirements
============

Mystic requires::

    - python, version >= 2.5, version < 3.0
    - numpy, version >= 1.0
    - sympy, version >= 0.6.7, version < 0.7.4
    - dill, version >= 0.2.1
    - klepto, version >= 0.1.1

Optional requirements::

    - setuptools, version >= 0.6
    - matplotlib, version >= 0.91
    - scipy, version >= 0.6.0
    - pathos, version >= 0.2a1.dev
    - pyina, version >= 0.2a1.dev


Usage Notes
===========

Probably the best way to get started is to look at a few of the
examples provided within mystic. See `mystic.examples` for a
set of scripts that demonstrate the configuration and launching of 
optimization jobs for one of the sample models in `mystic.models`.
Many of the included examples are standard optimization test problems.

Instr1ctions on building a new model are in `mystic.models.abstract_model`.
Mystic provides base classes for two types of models::

    - AbstractFunction   [evaluates f(x) for given evaluation points x]
    - AbstractModel      [generates f(x,p) for given coefficients p]

It is, however, not necessary to use the base classes in your own model.
Mystic also provides some convienence functions to help you build a
model instance and a cost function instance on-the-fly. For more
information, see `mystic.mystic.forward_model`.

All mystic solvers are highly configurable, and provide a robust set of
methods to help customize the solver for your particular optimization
problem. For each solver, a minimal interface is also provided for users
who prefer to configure their solvers in a single function call. For more
information, see `mystic.mystic.abstract_solver` for the solver API, and
each of the individual solvers for their minimal (non-API compliant)
interface.

Mystic extends the solver API to parallel computing by providing a solver
class that utilizes the parallel map-reduce algorithm. Mystic includes
a set of defaults in `mystic.mystic.python_map` that mirror the behavior
of serial python and the built-in python map function. Mystic solvers
built on map-reduce can utilize the distributed and parallel tools provided
by the `pathos` package, and thus with little new code solvers are
extended to high-performance computing. For more information, see
`mystic.mystic.abstract_map_solver`, `mystic.mystic.abstract_ensemble_solver`,
and the pathos documentation at http://dev.danse.us/trac/pathos.

Important classes and functions are found here::

    - mystic.mystic.solvers                  [solver optimization algorithms]
    - mystic.mystic.termination              [solver termination conditions]
    - mystic.mystic.strategy                 [solver population mutation strategies]
    - mystic.mystic.monitors                 [optimization monitors]
    - mystic.mystic.tools                    [function wrappers, etc]
    - mystic.mystic.forward_model            [cost function generator]
    - mystic.models                          [a collection of standard models]
    - mystic.math                            [some mathematical functions and tools]

Solver and model API definitions are found here::

    - mystic.mystic.abstract_solver          [the solver API definition]
    - mystic.mystic.abstract_map_solver      [the parallel solver API]
    - mystic.mystic.abstract_ensemble_solver [the ensemble solver API]
    - mystic.models.abstract_model           [the model API definition]


License
=======

Mystic is distributed under a 3-clause BSD license.

    >>> import mystic
    >>> print mystic.license()


Citation
========

If you use mystic to do research that leads to publication,
we ask that you acknowledge use of mystic by citing the
following in your publication::

    M.M. McKerns, L. Strand, T. Sullivan, A. Fang, M.A.G. Aivazis,
    "Building a framework for predictive science", Proceedings of
    the 10th Python in Science Conference, 2011;
    http://arxiv.org/pdf/1202.1056

    Michael McKerns, Patrick Hung, and Michael Aivazis,
    "mystic: a simple model-independent inversion framework", 2009- ;
    http://dev.danse.us/trac/mystic


More Information
================

Please see http://dev.danse.us/trac/mystic or http://arxiv.org/pdf/1202.1056 for further information.

""" % {'relver' : stable_version, 'thisver' : this_version}

# write readme file
with open('README', 'w') as file:
    file.write(long_description)

# generate 'info' file contents
def write_info_py(filename='mystic/info.py'):
    contents = """# THIS FILE GENERATED FROM SETUP.PY
this_version = '%(this_version)s'
stable_version = '%(stable_version)s'
readme = '''%(long_description)s'''
license = '''%(license_text)s'''
"""
    with open(filename, 'w') as file:
        file.write(contents % {'this_version' : this_version,
                               'stable_version' : stable_version,
                               'long_description' : long_description,
                               'license_text' : license_text })
    return

# write info file
write_info_py()

# build the 'setup' call
setup_code = """
setup(name='mystic',
      version='%s',
      description='a simple interactive inversion analysis framework',
      long_description = '''%s''',
      author = 'Mike McKerns',
      maintainer = 'Mike McKerns',
      maintainer_email = 'mmckerns@caltech.edu',
      license = 'BSD',
      platforms = ['any'],
      url = 'http://www.cacr.caltech.edu/~mmckerns',
      classifiers = ('Intended Audience :: Developers',
                     'Programming Language :: Python',
                     'Topic :: Physics Programming'),

      packages = ['mystic','mystic.models','mystic.math','mystic.cache'],
      package_dir = {'mystic':'mystic','mystic.models':'models',
                     'mystic.math':'_math','mystic.cache':'cache'},
""" % (target_version, long_description)

# add dependencies
if sys.version_info[:2] < (2.6):
    numpy_version = '>=1.0, <1.8.0'
    sympy_version = '>=0.6.7, <0.7.1'
else:
    numpy_version = '>=1.0'
    sympy_version = '>=0.6.7, <0.7.4'
dill_version = '>=0.2.1'
klepto_version = '>=0.1.1'
scipy_version = '>=0.6.0'
matplotlib_version = '>=0.91'
if has_setuptools:
    setup_code += """
      zip_safe=False,
      dependency_links = ['http://dev.danse.us/packages/'],
      install_requires = ('numpy%s', 'sympy%s', 'klepto%s', 'dill%s'),
""" % (numpy_version, sympy_version, klepto_version, dill_version)

# add the scripts, and close 'setup' call
setup_code += """
    scripts=['scripts/mystic_log_reader.py',
             'scripts/support_convergence.py',
             'scripts/support_hypercube.py',
             'scripts/support_hypercube_measures.py',
             'scripts/support_hypercube_scenario.py'])
"""

# exec the 'setup' code
exec setup_code

# if dependencies are missing, print a warning
try:
    import numpy
    import sympy
    import klepto
    import dill
    #import scipy
    #import matplotlib #XXX: has issues being zip_safe
except ImportError:
    print "\n***********************************************************"
    print "WARNING: One of the following dependencies is unresolved:"
    print "    numpy %s" % numpy_version
    print "    sympy %s" % sympy_version
    print "    klepto %s" % klepto_version
    print "    dill %s" % dill_version
    print "    scipy %s (optional)" % scipy_version
    print "    matplotlib %s (optional)" % matplotlib_version
    print "***********************************************************\n"


if __name__=='__main__':
    pass

# end of file
