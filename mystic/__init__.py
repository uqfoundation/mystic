#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1997-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
"""
mystic: a simple model-independent inversion framework

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
'mystic.abstract_solver'. The API also makes it easy to bind a favorite
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

Mystic provides a stock set of configurable, controllable solvers with:
    - a common interface
    - the ability to impose solver-independent bounds constraints
    - the ability to apply solver-independent monitors
    - the ability to configure selver-independent termination conditions
    - a control handler yeilding: [pause, continue, exit, and user_callback]
    - ease in selecting initial conditions: [initial_guess, random]
    - ease in selecting mutation strategies (for differential evolution)

To get up and running quickly, mystic also provides infrastructure to:
    - easily generate a fit model (several example models are included)
    - configure and auto-generate a cost function from a model
    - couple models with optimization parameter constraints [COMING SOON]
    - extend fit jobs to parallel & distributed resources [COMING SOON]


Installation
============

The latest version of mystic is available from
    http://dev.danse.us/trac/mystic

Mystic is packaged to install from source, so you must
download the tarball, unzip, and run the installer.
    [download]
    $ tar -xvzf mystic-x.x.tgz
    $ cd mystic-x.x
    $ python setup py build
    $ python setup py install

You will be warned of any missing dependencies and/or settings
after you run the "build" step above. Mystic depends on numpy,
so you should install it first. Having matplotlib is necessary
for running several of the examples, and you should probably go
get it even though it's not required. A few of the examples
require other packages, however they can be installed as needed.

Alternately, mystic can be installed with easy_install:
    [download]
    $ easy_install -f . mystic


Usage Notes
===========

Probably the best way to get started is to look at a few of the
examples provided within mystic. See 'mystic.examples' for a
set of scripts that demonstrate the configuration and launching of 
optimization jobs for one of the sample models in 'mystic.models'.
Many of the included examples are standard optimization test problems.

Instructions on building a new model are in 'mystic.models.abstract_model'.
Mystic provides base classes for two types of models:
  - AbstractFunction   [evaluates f(x) for given evaluation points x]
  - AbstractModel      [generates f(x,p) for given coefficients p]

It is, however, not necessary to use the base classes in your own model.
Mystic also provides some convienence functions to help you build a
model instance and a cost function instance on-the-fly. For more
information, see 'mystic.mystic.forward_model'.

All mystic solvers are highly configurable, and provide a robust set of
methods to help customize the solver for your particular optimization
problem. For each solver, a minimal interface is also provided for users
who prefer to configure their solvers in a single function call. For more
information, see 'mystic.mystic.abstract_solver' for the solver API, and
each of the individual solvers for their minimal (non-API compliant)
interface.

Important classes and functions are found here:
'mystic.mystic.abstract_solver', 'mystic.models.abstract_model',
'mystic.mystic.tools.Sow', 'mystic.mystic.termination'

Solvers are found here:
'mystic.mystic.differential_evolution', 'mystic.mystic.scipy_optimize'

"""

# solvers
import differential_evolution, scipy_optimize

# strategies, termination conditions
import termination
import strategy

# monitors, function wrappers, and other tools
from tools import *

# end of file
