mystic
======
highly-constrained non-convex optimization and uncertainty quantification

About Mystic
------------
The `mystic` framework provides a collection of optimization algorithms
and tools that allows the user to more robustly (and readily) solve
optimization problems. All optimization algorithms included in `mystic`
provide workflow at the fitting layer, not just access to the algorithms
as function calls. `mystic` gives the user fine-grained power to both
monitor and steer optimizations as the fit processes are running.

Where possible, `mystic` optimizers share a common interface, and thus can
be easily swapped without the user having to write any new code. `mystic`
solvers all conform to a solver API, thus also have common method calls
to configure and launch an optimization job. For more details, see
`mystic.abstract_solver`. The API also makes it easy to bind a favorite
3rd party solver into the `mystic` framework.

By providing a robust interface designed to allow the user to easily
configure and control solvers, `mystic` reduces the barrier to implementing
a target fitting problem as stable code. Thus the user can focus on
building their physical models, and not spend time hacking together an
interface to optimization code.

`mystic` is in active development, so any user feedback, bug reports, comments,
or suggestions are highly appreciated.  A list of known issues is maintained
at http://trac.mystic.cacr.caltech.edu/project/mystic/query, with a public
ticket list at https://github.com/uqfoundation/mystic/issues.


Major Features
--------------
`mystic` provides a stock set of configurable, controllable solvers with::

* a common interface
* the ability to impose solver-independent bounds constraints
* the ability to apply solver-independent monitors
* the ability to configure solver-independent termination conditions
* a control handler yielding: [pause, continue, exit, and callback]
* ease in selecting initial conditions: [guess, random]
* ease in selecting mutation strategies (for differential evolution)

To get up and running quickly, `mystic` also provides infrastructure to::

* easily generate a fit model (several example models are included)
* configure and auto-generate a cost function from a model
* extend fit jobs to parallel & distributed resources


Current Release
---------------
The latest released version of `mystic` is available from::
    http://trac.mystic.cacr.caltech.edu/project/mystic

`mystic` is distributed under a 3-clause BSD license.


Development Version
-------------------
You can get the latest development version with all the shiny new features at::
    https://github.com/uqfoundation

If you have a new contribution, please submit a pull request.


More Information
----------------
Probably the best way to get started is to look at a few of the
examples provided within `mystic`. See `mystic.examples` for a
set of scripts that demonstrate the configuration and launching of 
optimization jobs for one of the sample models in `mystic.models`.
Many of the included examples are standard optimization test problems.
The source code is also generally well documented, so further questions
may be resolved by inspecting the code itself.  Please also feel free to
submit a ticket on github, or ask a question on stackoverflow (@Mike McKerns).

`mystic` is an active research tool. There are a growing number of publications
and presentations that discuss real-world examples and new features of `mystic`
in greater detail than presented in the user's guide.  If you would like to
share how you use `mystic` in your work, please post a link or send an email
(to mmckerns at uqfoundation dot org).

Instructions on building a new model are in `mystic.models.abstract_model`.
`mystic` provides base classes for two types of models::

* `AbstractFunction`   [evaluates f(x) for given evaluation points x]
* `AbstractModel`      [generates f(x,p) for given coefficients p]

It is, however, not necessary to use the base classes in your own model.
`mystic` also provides some convienence functions to help you build a
model instance and a cost function instance on-the-fly. For more
information, see `mystic.forward_model`.

All `mystic` solvers are highly configurable, and provide a robust set of
methods to help customize the solver for your particular optimization
problem. For each solver, a minimal interface is also provided for users
who prefer to configure their solvers in a single function call. For more
information, see `mystic.abstract_solver` for the solver API, and
each of the individual solvers for their minimal (non-API compliant)
interface.

`mystic` extends the solver API to parallel computing by providing a solver
class that utilizes the parallel map-reduce algorithm. `mystic` includes
a set of defaults in `mystic.python_map` that mirror the behavior
of serial python and the built-in python map function. `mystic` solvers
built on map-reduce can utilize the distributed and parallel tools provided
by the `pathos` package, and thus with little new code solvers are
extended to high-performance computing. For more information, see
`mystic.abstract_map_solver`, `mystic.abstract_ensemble_solver`,
and the `pathos` documentation at http://dev.danse.us/trac/pathos.

Important classes and functions are found here::

* `mystic.solvers`                  [solver optimization algorithms]
* `mystic.termination`              [solver termination conditions]
* `mystic.strategy`                 [solver population mutation strategies]
* `mystic.monitors`                 [optimization monitors]
* `mystic.tools`                    [function wrappers, etc]
* `mystic.forward_model`            [cost function generator]
* `mystic.models`                   [a collection of standard models]
* `mystic.math`                     [some mathematical functions and tools]

Solver and model API definitions are found here::

* `mystic.abstract_solver`          [the solver API definition]
* `mystic.abstract_map_solver`      [the parallel solver API]
* `mystic.abstract_ensemble_solver` [the ensemble solver API]
* `mystic.models.abstract_model`    [the model API definition]


Citation
--------
If you use `mystic` to do research that leads to publication, we ask that you
acknowledge use of `mystic` by citing the following in your publication::

    M.M. McKerns, L. Strand, T. Sullivan, A. Fang, M.A.G. Aivazis,
    "Building a framework for predictive science", Proceedings of
    the 10th Python in Science Conference, 2011;
    http://arxiv.org/pdf/1202.1056

    Michael McKerns, Patrick Hung, and Michael Aivazis,
    "mystic: a simple model-independent inversion framework", 2009- ;
    http://trac.mystic.cacr.caltech.edu/project/mystic

Please see http://trac.mystic.cacr.caltech.edu/project/mystic or
http://arxiv.org/pdf/1202.1056 for further information.

