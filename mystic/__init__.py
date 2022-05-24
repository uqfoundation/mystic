#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

# author, version, license, and long description
__version__ = '0.3.9'
__author__ = 'Mike McKerns'

__doc__ = """
---------------------------------------------------------------------------------
mystic: highly-constrained non-convex optimization and uncertainty quantification
---------------------------------------------------------------------------------

About Mystic
============

The ``mystic`` framework provides a collection of optimization algorithms
and tools that allows the user to more robustly (and easily) solve hard
optimization problems. All optimization algorithms included in ``mystic``
provide workflow at the fitting layer, not just access to the algorithms
as function calls. ``mystic`` gives the user fine-grained power to both
monitor and steer optimizations as the fit processes are running.
Optimizers can advance one iteration with ``Step``, or run to completion
with ``Solve``.  Users can customize optimizer stop conditions, where both
compound and user-provided conditions may be used. Optimizers can save
state, can be reconfigured dynamically, and can be restarted from a
saved solver or from a results file.  All solvers can also leverage
parallel computing, either within each iteration or as an ensemble of
solvers.

Where possible, ``mystic`` optimizers share a common interface, and thus can
be easily swapped without the user having to write any new code. ``mystic``
solvers all conform to a solver API, thus also have common method calls
to configure and launch an optimization job. For more details, see
``mystic.abstract_solver``. The API also makes it easy to bind a favorite
3rd party solver into the ``mystic`` framework.

Optimization algorithms in ``mystic`` can accept parameter constraints,
either in the form of penaties (which "penalize" regions of solution
space that violate the constraints), or as constraints (which "constrain" 
the solver to only search in regions of solution space where the
constraints are respected), or both. ``mystic`` provides a large 
selection of constraints, including probabistic and dimensionally
reducing constraints. By providing a robust interface designed to
enable the user to easily configure and control solvers, ``mystic``
greatly reduces the barrier to solving hard optimization problems.

``mystic`` is in active development, so any user feedback, bug reports, comments,
or suggestions are highly appreciated.  list of issues is located at https://github.com/uqfoundation/mystic/issues, with a legacy list maintained at https://uqfoundation.github.io/project/mystic/query.


Major Features
==============

``mystic`` provides a stock set of configurable, controllable solvers with:

    -  a common interface
    -  a control handler with: pause, continue, exit, and callback
    -  ease in selecting initial population conditions: guess, random, etc
    -  ease in checkpointing and restarting from a log or saved state
    -  the ability to leverage parallel & distributed computing
    -  the ability to apply a selection of logging and/or verbose monitors
    -  the ability to configure solver-independent termination conditions
    -  the ability to impose custom and user-defined penalties and constraints

To get up and running quickly, ``mystic`` also provides infrastructure to:

    - easily generate a model (several standard test models are included)
    - configure and auto-generate a cost function from a model
    - configure an ensemble of solvers to perform a specific task


Current Release
===============

The latest released version of ``mystic`` is available from:

    https://pypi.org/project/mystic

``mystic`` is distributed under a 3-clause BSD license.


Development Version 
===================

You can get the latest development version with all the shiny new features at:

    https://github.com/uqfoundation

If you have a new contribution, please submit a pull request.


Installation
============

``mystic`` can be installed with ``pip``::

    $ pip install mystic

To include optional scientific python support, with ``scipy``, install::

    $ pip install mystic[math]

To include optional plotting support with ``matplotlib``, install::

    $ pip install mystic[plotting]

To include optional parallel computing support, with ``pathos``, install::

    $ pip install mystic[parallel]


Requirements
============

``mystic`` requires:

    - ``python`` (or ``pypy``), **==2.7** or **>=3.7**
    - ``setuptools``, **>=42**
    - ``cython``, **>=0.29.22**
    - ``numpy``, **>=1.0**
    - ``sympy``, **>=0.6.7**
    - ``mpmath``, **>=0.19**
    - ``dill``, **>=0.3.5.1**
    - ``klepto``, **>=0.2.2**

Optional requirements:

    - ``matplotlib``, **>=0.91**
    - ``scipy``, **>=0.6.0**
    - ``pathos``, **>=0.2.9**
    - ``pyina``, **>=0.2.6**


More Information
================

Probably the best way to get started is to look at the documentation at
http://mystic.rtfd.io. Also see ``mystic.tests`` for a set of scripts that
demonstrate several of the many features of the ``mystic`` framework.
You can run the test suite with ``python -m mystic.tests``. There are
several plotting scripts that are installed with ``mystic``, primary of which
are `mystic_log_reader`` (also available with ``python -m mystic``) and the
``mystic_model_plotter`` (also available with ``python -m mystic.models``).
There are several other plotting scripts that come with ``mystic``, and they
are detailed elsewhere in the documentation.  See ``mystic.examples`` for
examples that demonstrate the basic use cases for configuration and launching
of optimization jobs using one of the sample models provided in
``mystic.models``. Many of the included examples are standard optimization
test problems. The use of constraints and penalties are detailed in
``mystic.examples2``, while more advanced features leveraging ensemble solvers
and dimensional collapse are found in ``mystic.examples3``. The scripts in
``mystic.examples4`` demonstrate leveraging ``pathos`` for parallel computing,
as well as demonstrate some auto-partitioning schemes. ``mystic`` has the
ability to work in product measure space, and the scripts in
``mystic.examples5`` show to work with product measures.  The source code is
generally well documented, so further questions may be resolved by inspecting
the code itself.  Please feel free to submit a ticket on github, or ask a
question on stackoverflow (**@Mike McKerns**).
If you would like to share how you use ``mystic`` in your work, please send an
email (to **mmckerns at uqfoundation dot org**).

Instructions on building a new model are in ``mystic.models.abstract_model``.
``mystic`` provides base classes for two types of models:

    - ``AbstractFunction``   [evaluates ``f(x)`` for given evaluation points ``x``]
    - ``AbstractModel``      [generates ``f(x,p)`` for given coefficients ``p``]

``mystic`` also provides some convienence functions to help you build a
model instance and a cost function instance on-the-fly. For more
information, see ``mystic.forward_model``.  It is, however, not necessary
to use base classes or the model builder in building your own model or
cost function, as any standard python function can be used as long as it
meets the basic ``AbstractFunction`` interface of ``cost = f(x)``.

All ``mystic`` solvers are highly configurable, and provide a robust set of
methods to help customize the solver for your particular optimization
problem. For each solver, a minimal (``scipy.optimize``) interface is also
provided for users who prefer to configure and launch their solvers as a
single function call. For more information, see ``mystic.abstract_solver``
for the solver API, and each of the individual solvers for their minimal
functional interface.

``mystic`` enables solvers to use parallel computing whenever the user provides
a replacement for the (serial) python ``map`` function.  ``mystic`` includes a
sample ``map`` in ``mystic.python_map`` that mirrors the behavior of the
built-in python ``map``, and a ``pool`` in ``mystic.pools`` that provides ``map``
functions using the ``pathos`` (i.e. ``multiprocessing``) interface. ``mystic``
solvers are designed to utilize distributed and parallel tools provided by
the ``pathos`` package. For more information, see ``mystic.abstract_map_solver``,
``mystic.abstract_ensemble_solver``, and the ``pathos`` documentation at
http://pathos.rtfd.io.

Important classes and functions are found here:

    - ``mystic.solvers``                  [solver optimization algorithms]
    - ``mystic.termination``              [solver termination conditions]
    - ``mystic.strategy``                 [solver population mutation strategies]
    - ``mystic.monitors``                 [optimization monitors]
    - ``mystic.symbolic``                 [symbolic math in constaints]
    - ``mystic.constraints``              [constraints functions]
    - ``mystic.penalty``                  [penalty functions]
    - ``mystic.collapse``                 [checks for dimensional collapse]
    - ``mystic.coupler``                  [decorators for function coupling]
    - ``mystic.pools``                    [parallel worker pool interface]
    - ``mystic.munge``                    [file readers and writers]
    - ``mystic.scripts``                  [model and convergence plotting]
    - ``mystic.samplers``                 [optimizer-guided sampling]
    - ``mystic.support``                  [hypercube measure support plotting]
    - ``mystic.forward_model``            [cost function generator]
    - ``mystic.tools``                    [constraints, wrappers, and other tools]
    - ``mystic.cache``                    [results caching and archiving]
    - ``mystic.models``                   [models and test functions]
    - ``mystic.math``                     [mathematical functions and tools]

Important functions within ``mystic.math`` are found here:

    - ``mystic.math.Distribution``        [a sampling distribution object]
    - ``mystic.math.legacydata``          [classes for legacy data observations]
    - ``mystic.math.discrete``            [classes for discrete measures]
    - ``mystic.math.measures``            [tools to support discrete measures]
    - ``mystic.math.approx``              [tools for measuring equality]
    - ``mystic.math.grid``                [tools for generating points on a grid]
    - ``mystic.math.distance``            [tools for measuring distance and norms]
    - ``mystic.math.poly``                [tools for polynomial functions]
    - ``mystic.math.samples``             [tools related to sampling]
    - ``mystic.math.integrate``           [tools related to integration]
    - ``mystic.math.interpolate``         [tools related to interpolation]
    - ``mystic.math.stats``               [tools related to distributions]

Solver, Sampler, and model API definitions are found here:

    - ``mystic.abstract_sampler``         [the sampler API definition]
    - ``mystic.abstract_solver``          [the solver API definition]
    - ``mystic.abstract_map_solver``      [the parallel solver API]
    - ``mystic.abstract_ensemble_solver`` [the ensemble solver API]
    - ``mystic.models.abstract_model``    [the model API definition]

``mystic`` also provides several convience scripts that are used to visualize
models, convergence, and support on the hypercube. These scripts are installed
to a directory on the user's ``$PATH``, and thus can be run from anywhere:

   - ``mystic_log_reader``               [parameter and cost convergence]
   - ``mystic_collapse_plotter``         [convergence and dimensional collapse]
   - ``mystic_model_plotter``            [model surfaces and solver trajectory]
   - ``support_convergence``             [convergence plots for measures]
   - ``support_hypercube``               [parameter support on the hypercube]
   - ``support_hypercube_measures``      [measure support on the hypercube]
   - ``support_hypercube_scenario``      [scenario support on the hypercube]

Typing ``--help`` as an argument to any of the above scripts will print out an
instructive help message.


Citation
========

If you use ``mystic`` to do research that leads to publication, we ask that you
acknowledge use of ``mystic`` by citing the following in your publication::

    M.M. McKerns, L. Strand, T. Sullivan, A. Fang, M.A.G. Aivazis,
    "Building a framework for predictive science", Proceedings of
    the 10th Python in Science Conference, 2011;
    http://arxiv.org/pdf/1202.1056

    Michael McKerns, Patrick Hung, and Michael Aivazis,
    "mystic: highly-constrained non-convex optimization and UQ", 2009- ;
    https://uqfoundation.github.io/project/mystic

Please see https://uqfoundation.github.io/project/mystic or
http://arxiv.org/pdf/1202.1056 for further information.

"""

__license__ = """
Copyright (c) 2004-2016 California Institute of Technology.
Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
All rights reserved.

This software is available subject to the conditions and terms laid
out below. By downloading and using this software you are agreeing
to the following conditions.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met::

    - Redistribution of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistribution in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentations and/or other materials provided with the distribution.

    - Neither the names of the copyright holders nor the names of any of
      the contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

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

"""

__all__ = ['solvers','termination','strategy','munge','tools','support', \
           'penalty','coupler','symbolic','monitors','license','citation', \
           'constraints','model_plotter','log_reader','collapse_plotter']

# solvers
import mystic.solvers as solvers

# strategies, termination conditions
import mystic.termination as termination
import mystic.strategy as strategy

# constraints and penalties
import mystic.constraints as constraints
import mystic.penalty as penalty
import mystic.coupler as coupler
import mystic.symbolic as symbolic

# monitors, function wrappers, and other tools
import mystic.monitors as monitors
import mystic.munge as munge
import mystic.tools as tools

# scripts
from mystic.scripts import model_plotter, log_reader, collapse_plotter
import mystic.support as support

# backward compatibility
from mystic.tools import *


def license():
    """print license"""
    print(__license__)
    return

def citation():
    """print citation"""
    print(__doc__[-516:-118])
    return

# end of file
