#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                 Mike McKerns, Alta Fang, & Patrick Hung, Caltech
#                        (C) 1997-2011  All Rights Reserved
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

This release version is mystic-0.2a2.dev. You can download it here.
The latest version of mystic is available from::
    http://dev.danse.us/trac/mystic

Mystic is distributed under a modified BSD license.


Installation
============

Mystic is packaged to install from source, so you must
download the tarball, unzip, and run the installer::
    [download]
    $ tar -xvzf mystic-0.2a2.dev.tgz
    $ cd mystic-0.2a2.dev
    $ python setup py build
    $ python setup py install

You will be warned of any missing dependencies and/or settings
after you run the "build" step above. Mystic depends on numpy,
so you should install it first. Having matplotlib is necessary
for running several of the examples, and you should probably go
get it even though it's not required.

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

Optional requirements::
    - setuptools, version >= 0.6
    - matplotlib, version >= 0.91
    - pathos, version >= 0.1a1


Usage Notes
===========

Probably the best way to get started is to look at a few of the
examples provided within mystic. See `mystic.examples` for a
set of scripts that demonstrate the configuration and launching of 
optimization jobs for one of the sample models in `mystic.models`.
Many of the included examples are standard optimization test problems.

Instructions on building a new model are in `mystic.models.abstract_model`.
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
`mystic.mystic.abstract_map_solver`, `mystic.mystic.abstract_nested_solver`,
and the pathos documentation at http://dev.danse.us/trac/pathos.

Important classes and functions are found here::
    - mystic.mystic.abstract_solver        [the solver API definition]
    - mystic.mystic.abstract_map_solver    [the parallel solver API]
    - mystic.mystic.abstract_nested_solver [the nested solver API]
    - mystic.mystic.termination            [solver termination conditions]
    - mystic.mystic.strategy               [solver population mutation strategies]
    - mystic.models.abstract_model         [the model API definition]
    - mystic.models.forward_model          [cost function generator]
    - mystic.mystic.tools                  [monitors, function wrappers, and other tools]
    - mystic.mystic.math                   [some useful mathematical functions and tools]

Solvers are found here::
    - mystic.mystic.differential_evolution [Differential Evolution solvers]
    - mystic.mystic.scipy_optimize         [Nelder-Mead and Powell's Directional solvers]
    - mystic.mystic.nested                 [Batch Grid and Scattershot solvers]


More Information
================

Please see http://dev.danse.us/trac/mystic for further information.
"""
__version__ = '0.2a2.dev'
__author__ = 'Mike McKerns'

__license__ = """
This software is part of the open-source DANSE project at the California
Institute of Technology, and is available subject to the conditions and
terms laid out below. By downloading and using this software you are
agreeing to the following conditions.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met::

    - Redistribution of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistribution in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentations and/or other materials provided with the distribution.

    - Neither the name of the California Institute of Technology nor
      the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Copyright (c) 2011 California Institute of Technology. All rights reserved.


If you use this software to do productive scientific research that leads to
publication, we ask that you acknowledge use of the software by citing the
following in your publication::

    M.M. McKerns, L. Strand, T. Sullivan, A. Fang, M.A.G. Aivazis,
    "Building a framework for predictive science", Proceedings of
    the 10th Python in Science Conference, (submitted 2011).

    Michael McKerns, Patrick Hung, and Michael Aivazis,
    "mystic: a simple model-independent inversion framework", 2009- ;
    http://dev.danse.us/trac/mystic

"""

# solvers
import differential_evolution, scipy_optimize, nested

# strategies, termination conditions
import termination
import strategy

# monitors, function wrappers, and other tools
from tools import *

def copyright():
    """print copyright and reference"""
    print __license__[-439:]
    return

# end of file
