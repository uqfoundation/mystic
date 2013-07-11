a framework for highly-constrained non-convex optimization and uncertainty quantification

About Mystic
------------
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
issues is maintained at http://trac.mystic.cacr.caltech.edu/project/mystic/query.

Major Features
--------------
Mystic provides a stock set of configurable, controllable solvers with::
    * a common interface
    * the ability to impose solver-independent bounds constraints
    * the ability to apply solver-independent monitors
    * the ability to configure solver-independent termination conditions
    * a control handler yielding: [pause, continue, exit, and user_callback]
    * ease in selecting initial conditions: [initial_guess, random]
    * ease in selecting mutation strategies (for differential evolution)

To get up and running quickly, mystic also provides infrastructure to::
    * easily generate a fit model (several example models are included)
    * configure and auto-generate a cost function from a model
    * extend fit jobs to parallel & distributed resources
    * couple models with optimization parameter constraints [COMING SOON]


Current Release
---------------
The latest released version of mystic is available from::
    http://trac.mystic.cacr.caltech.edu/project/mystic

Mystic is distributed under a modified BSD license.

Development Release
-------------------
You can get the latest development release with all the shiny new features at::
    http://dev.danse.us/packages.

or even better, fork us on our github mirror of the svn trunk::
    https://github.com/uqfoundation

Citation
--------
If you use mystic to do research that leads to publication, we ask that you
acknowledge use of mystic by citing the following in your publication::

    M.M. McKerns, L. Strand, T. Sullivan, A. Fang, M.A.G. Aivazis,
    "Building a framework for predictive science", Proceedings of
    the 10th Python in Science Conference, 2011;
    http://arxiv.org/pdf/1202.1056

    Michael McKerns, Patrick Hung, and Michael Aivazis,
    "mystic: a simple model-independent inversion framework", 2009- ;
    http://trac.mystic.cacr.caltech.edu/project/mystic

More Information
----------------
Probably the best way to get started is to look at the tutorial examples provided
within the user's guide.  The source code is also generally well documented,
so further questions may be resolved by inspecting the code itself, or through 
browsing the reference manual. For those who like to leap before
they look, you can jump right to the installation instructions. If the aforementioned documents
do not adequately address your needs, please send us feedback.

Mystic is an active research tool. There are a growing number of publications and presentations that
discuss real-world examples and new features of mystic in greater detail than presented in the user's guide. 
If you would like to share how you use mystic in your work, please send us a link.
