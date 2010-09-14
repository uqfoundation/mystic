#!/usr/bin/env python
#
# coded by Alta Fang, 2010
"""
Tools for imposing constraints on an optimization.

The main functions exported are::
    - verify_constraints_satisfied: checks if a set of parameters x satisfies
          the constraints, and can calculate the norm of residual errors.
    - matrix_to_symbolic: converts constraints from matrix form to a symbolic
          string that can be input into a constraints-function generator.

Several 'advanced' functions are also included::
    - wrap_constraints_to_cost: takes a string of constraints of any form
          (inequality or equality, and/or mean and range constraints), and
          wraps it to the cost function using the `penalty' method.
    - find_feasible_point: tries to find a point to satisfy given constraints.
    - form_constraints_function: creates a constraints function from a symbolic
          string. This only handles equality constraints, and no mean or range
          constraints. It first tries to call `form_constraints_linear', and
          if that fails, then tries `form_constraints_nonlinear'. It returns a
          function that imposes the constraints on an x value, and this
          function can then be passed to the solver with the `constraints'
          keyword in solver.Solve. This function is called when using the
          `direct' method.
    - constraints_inside_bounds: check if constraints_function is inside bounds.
    - form_constraints_directly: parses a string into a constraints function.
    - form_constraints_linear: parses a string of linear equations into a
          constraints function.
    - form_constraints_nonlinear: parses a string of equations that are not
          necessarily linear into a constraints function.
    - simplify: solves an equation for each given variable, or,
          optionally, for a single target variable.
    - get_variables: extract a list of the string variable names from the given
          equations string.
    - sumt: solves several successive optimization problems with slightly
          different cost functions each time, and produces a solver instance
          containing all of the optimal information.
"""
from __future__ import division
from math import *
from numpy import *
import random
from mystic.math import approx_equal
from mystic.tools import Null

#------------------------------------------------------------------
# Useful helper functions.

def matrix_to_symbolic(A=None, b=None, G=None, h=None):
    """Convert linear equality and inequality constraints from matrices to a 
symbolic string of the form required by mystic's solver.Solve method.

Inputs:
    A -- (ndarray) matrix of coefficients of linear equality constraints
    b -- (ndarray) vector of solutions of linear equality constraints
    G -- (ndarray) matrix of coefficients of linear inequality constraints
    h -- (ndarray) vector of solutions of linear inequality constraints

    NOTE: Must provide A and b; G and h; or A, b, G, and h;
          where Ax = b and Gx <= h. 

    For example:
        A = [[3., 4., 5.],
             [1., 6., -9.]]
        b = [0., 0.]
        G = [1., 0., 0.]
        h = [5.]
"""
    eqstring = ""
    # Equality constraints
    if A != None and b != None:
        # If one-dimensional and not in a nested list, add a list layer
        try:
            ndim = len(A[0])
        except:
            ndim = len(A)
            A = [A]

        # Flatten b, in case it's in the form [[0, 1, 2]] for example.
        if len(b) == 1:
            b = list(ndarray.flatten(asarray(b)))

        # Check dimensions and give errors if incorrect.
        if len(A) != len(b):
            raise Exception("Dimensions of A and b are not consistent.")

        # 'matrix multiply' and form the string
        for i in range(len(b)):
            Asum = ""
            for j in range(ndim):
                Asum += str(A[i][j]) + '*x' + str(j+1) + ' + '
            eqstring += Asum.rstrip(' + ') + '=' + str(b[i]) + '\n'

    # Inequality constraints
    ineqstring = ""
    if G != None and h != None:
        # If one-dimensional and not in a nested list, add a list layer
        try:
            ndim = len(G[0])
        except:
            ndim = len(G)
            G = [G]

        # Flatten h, in case it's in the form [[0, 1, 2]] for example.
        if len(h) == 1:
            h = list(ndarray.flatten(asarray(h)))

        # Check dimensions and give errors if incorrect.
        if len(G) != len(h):
            raise Exception("Dimensions of G and h are not consistent.")

        # 'matrix multiply' and form the string
        for i in range(len(h)):
            Gsum = ""
            for j in range(ndim):
                Gsum += str(G[i][j]) + '*x' + str(j+1) + ' + '
            ineqstring += Gsum.rstrip(' + ') + '<=' + str(h[i]) + '\n'
    totalconstraints = ineqstring + eqstring
    return totalconstraints 

#Tips on using the symbolic string:
#* When inputting the constraints, ignore the keywords "varname" and
#  "varnamelist". The default, 'x', is used.
def verify_constraints_satisfied(constraints_string, x,\
                                 varname='x', disp=False, tol=1e-3,\
                                 varnamelist=None, returnerror=True, \
                                 eqcon_funcs=[], ineqcon_funcs=[]):
    """Calculates the amount of constraints violation (norm of residual errors)
and returns whether constraints were violated or not within some tolerance. 
A tuple of (satisfied, norm) with type (boolean, float) is returned, unless 
returnerror=False. If returnerror=False, only a boolean for whether the
constraints are satisfied or not is returned.

Inputs:
    constraints_string -- string of symbolic constraints.
    x -- (list) parameter vector to check if satisfies the given constraints.

Additional Inputs:
    ineqcon_funcs -- list of 'inequality' functions, f, where f(x) <= 0.
    eqcon_funcs -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints_string' and provide ineqcon_funcs and/or eqcon_funcs.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: ineqcon_funcs=[f1], eqcon_funcs=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.

    disp -- True to print values on each side of the equality or inequality
        for each constraint. Default is False.
    tol -- size limit of norm of residual errors for which constraints
        are considered to be solved. Default is 1e-3.
    returnerror -- False to only return a boolean. Default is True.
"""
    from mystic.math import approx_equal

    if varnamelist:
        constraints_string = _replace_variable_names(constraints_string, varnamelist)
        varname = '$'

    ndim = len(x)

    # Iterate in reverse in case ndim > 9.
    indices = list(range(ndim))
    indices.reverse()
    for i in indices:
        variable = varname + str(i+1)
        constraints_string = constraints_string.replace(variable, 'x[' + str(i) + ']')

    constraints_list = constraints_string.splitlines()

    # Remove empty strings:
    actual_eqns = []
    for j in range(len(constraints_list)):
        if constraints_list[j].strip():
           actual_eqns.append(constraints_list[j].strip())

    error = 0.
    for item in actual_eqns:
        # Strip 'strict' from the strings
        if item.find('strict') != -1:
            item = item.rstrip('strict').strip().rstrip(',')

        if item.find('mean') != -1: 
            mean = average(asarray(x))
            if disp: 
                print 'actual mean = ', mean 
            split = item.split('>')
            if len(split) == 1:
                split = item.split('<')
            if len(split) == 1:
                split = item.split('=')
                item = item.replace('=', '==')
            if not eval(item):
                # Only add to error if the constraint is not satisfied
                error += (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
            continue
        if item.find('range') != -1:
            r = max(x) - min(x)
            item = item.replace('range', 'r')
            if disp:
                print 'actual range = ', r
            split = item.split('>')
            if len(split) == 1:
                split = item.split('<')
            if len(split) == 1:
                split = item.split('=')
                item = item.replace('=', '==')
            if not eval(item):
                # Only add to error if the constraint is not satisfied
                error += (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
            continue
        split = item.split('>')
        if len(split) != 1:
            if disp:
                print eval(split[0]), ' ?>= ', eval(split[1].lstrip(' = '))
            if not eval(split[0] + '>' + split[1]):
                # Only add to error if the constraint is not satisfied
                error += (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
            continue
        split = item.split('<')
        if len(split) != 1:
            if disp:
                print eval(split[0]), ' ?<= ', eval(split[1].lstrip('='))
            if not eval(split[0] + '<' + split[1]): 
                # Only add to error if the constraint is not satisfied
                error += (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
            continue
        split = item.split('=')
        if len(split) != 1:
            if disp:
                print eval(split[0]), ' ?= ', eval(split[1])
            if not eval(split[0]) == eval(split[1]):
                # Only add to error if the constraint is not satisfied
                error += (eval(split[0]) - eval(split[1]))**2
            continue

    # Loop through eqcon_funcs and ineqcon_funcs
    for func in eqcon_funcs:
        f = func(x)
        error += f**2
        if disp:
            print f, ' ?= 0.0'
    for func in ineqcon_funcs:
        f = func(x)
        error += max(0., f)**2
        if disp:
            print f, ' ?<= 0.0'

    if error**0.5 <= tol:
        satisfied = True
    else:
        satisfied = False

    if returnerror:
        return (satisfied, error**0.5)
    return satisfied

#--------------------------------------------------------------------------------
# wrap_constraints_to_cost is mostly used as a helper function for the penalty
# and barrier methods, but could be used directly too, if for some reason
# one wants to wrap a cost function with this instead of using keywords to 
# Solve().
# find_feasible_point is mostly a helper function for the barrier method, but
# could also be used by users to find a feasible point without writing too much
# code.

def wrap_constraints_to_cost(constraints_string, ndim, costfunc,\
                             varname='x', penalty=1e4, varnamelist=None, \
                             ineqcon_funcs=[], eqcon_funcs=[], \
                             strict_constraints=None, **kwds):
    """Wraps a cost function with a set of constraints. The constraints are
imposed using the `penalty' method, using a fixed penalty parameter. Returns
a cost function with the constraints built-in.

Inputs:
    constraints_string -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints_string = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...

    ndim -- number of variables. Includes xi not explicit in constraints_string.
        Should correspond to len(x) for x in costfunc(x).
    costfunc -- the cost function to be minimized.

Additional Inputs:
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.

    ineqcon_funcs -- list of 'inequality' functions, f, where f(x) <= 0.
    eqcon_funcs -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints_string' and provide ineqcon_funcs and/or eqcon_funcs.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: ineqcon_funcs=[f1], eqcon_funcs=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    penalty -- penalty multiplier if the constraints are violated. Default
        is 1e4. It is not recommended to use infinity.
    strict_constraints -- list of constraint strings, where if any constraint
        is violated, the penalty is set to infinity (i.e. cost(x) = inf).

References: 
    [1] http://en.wikipedia.org/wiki/Penalty_method
    [2] Applied Optimization with MATLAB programming by Venkataraman,
        section 7.2.1: Exterior Penalty Function Method
    [3] http://www.srl.gatech.edu/education/ME6103/Penalty-Barrier.ppt
"""
    if varnamelist:
        constraints_string = _replace_variable_names(constraints_string, \
                                                     varnamelist)
        varname = '$'

    # Parse the constraints string
    lines = constraints_string.splitlines()
    eqconstraints = []
    ineqconstraints = []
    for line in lines:
        if line.strip():
            fixed = line
            # Iterate in reverse in case ndim > 9.
            indices = list(range(1, ndim+1))
            indices.reverse()
            for i in indices:
                fixed = fixed.replace(varname + str(i), 'x[' + str(i-1) + ']') 
            constraint = fixed.strip()
            # Replace 'mean' with actual expression for calculating mean
            if constraint.find('mean') != -1:
                constraint = constraint.replace('mean', 'average(asarray(x))')
            # Replace 'range' with actual expression for calculating range
            if constraint.find('range') != -1:
                constraint = constraint.replace('range', 'max(x) - min(x)')
            # Sorting into equality and inequality constraints, and making all
            # inequality constraints in the form expression <= 0. and all 
            # equality constraints of the form expression = 0.
            split = constraint.split('>')
            direction = '>'
            if len(split) == 1:
                split = constraint.split('<')
                direction = '<'
            if len(split) == 1:
                split = constraint.split('=')
                direction = '='
            if len(split) == 1:
                raise Exception("Invalid constraint: %s" % line.strip())
            expression = split[0].rstrip('=') + '-(' + split[1].lstrip('=') + ')'
            if direction == '=':
                eqconstraints.append(expression)
            elif direction == '<':
                ineqconstraints.append(expression)
            else:
                ineqconstraints.append('-(' + expression + ')')

    # Use exterior penalty function method, with fixed penalty. 
    # It tolerates infeasible starting points. Ideally, the penalty should
    # be higher if the function values are higher, but replacing penalty
    # with costfunc(x)*penalty is no good because that is too sensitive
    # to the costfunc value. Increasing the penalty by some fixed factor 
    # if result > penalty*0.5, for example, is too abrupt and also arbitrary.
    def wrapped_costfunc(x):
        if strict_constraints:
            for constraint in strict_constraints:
                if not eval(constraint):
                    return inf
        result = costfunc(x)
        # For constraints that were input symbolically
        for constraint in ineqconstraints:
            result += float(penalty)*max(0., eval(constraint))**2
        for constraint in eqconstraints:
            result += float(penalty)*eval(constraint)**2
        # For constraints in function form
        for constraint in ineqcon_funcs:
            result += float(penalty)*max(0., constraint(x))**2
        for constraint in eqcon_funcs:
            result += float(penalty)*constraint(x)**2
        return result

    return wrapped_costfunc


def find_feasible_point(constraints, ndim, x0=None, \
                        varnamelist=None, lb=None, ub=None, \
                        varname='x', eqcon_funcs=[], ineqcon_funcs=[]):
    """Use optimization to find a solution to a set of constraints. Returns
the solution, if one is found. If no solution is found, returns None.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints_string = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...

    ndim -- number of variables. Includes xi not explicit in constraints_string.

Additional Inputs:
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.

    ineqcon_funcs -- list of 'inequality' functions, f, where f(x) <= 0.
    eqcon_funcs -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints_string' and provide ineqcon_funcs and/or eqcon_funcs.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: ineqcon_funcs=[f1], eqcon_funcs=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    x0 -- list of initial parameter values.
    lower_bounds -- list of lower bounds on parameters.
    upper_bounds -- list of upper bounds on parameters.
"""
    def dummy_func(x):
        return 1.

    from mystic.differential_evolution import DifferentialEvolutionSolver
    from mystic.termination import ChangeOverGeneration as COG
    solver = DifferentialEvolutionSolver(ndim, 40)
    if x0 != None:
        solver.SetInitialPoints(x0)
    else:
        if lb and ub:
            solver.SetRandomInitialPoints(lb, ub)
        else:
            solver.SetRandomInitialPoints()
    if lb != None and len(lb) > 0 and ub != None and len(ub) > 0:
        solver.SetStrictRanges(lb, ub)
    #solver.enable_signal_handler()
    solver.Solve(dummy_func, COG(), constraints=constraints, eqcon_funcs=\
                 eqcon_funcs, ineqcon_funcs=ineqcon_funcs, varname=varname,\
                 varnamelist=varnamelist, constraints_method='penalty')
    soln = solver.Solution()
    if not verify_constraints_satisfied(constraints, soln, varname=varname,\
                                disp=False, varnamelist=varnamelist, \
                                returnerror=False, eqcon_funcs=eqcon_funcs,\
                                ineqcon_funcs=ineqcon_funcs):
        soln = None
    return soln

#------------------------------------------------------------------
# Purely helper functions

def _replace_variable_names(constraints_string, varnamelist):
    """Replace variables in constraints_string with a marker '$i',
where i = 1,2,3,...  Returns a modified constraints_string.

Inputs:
    constraints_string -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    For example:
        >>> varnamelist = ['spam', 'eggs']
        >>> constraints = '''spam + eggs - 42'''
        >>> con_string = _replace_variable_names(constraints, varnamelist)
        >>> print con_string
        '$1 + $2 - 42'
"""
    # Helper function for when varnamelist is specified

    # Sort by decreasing length of variable name, so that if one variable name 
    # is a substring of another, that won't be a problem. 
    varnamelistcopy = varnamelist[:]
    def comparator(x, y):
        return len(y) - len(x)
    varnamelistcopy.sort(comparator)

    # Figure out which index goes with which variable.
    indices = []
    for item in varnamelistcopy:
        indices.append(varnamelist.index(item))

    # Use '$' for varname, as it is not a special symbol in Python and it seems
    # unlikely a user will chose it for a variable name.
    for i in indices:
        constraints_string = constraints_string.replace(varnamelist[i], '$' + str(i+1))
    return constraints_string

#XXX: probably needs redesign -- should be a property of a constraints method?
def _process_constraints(solver_instance, constraints, costfunc, \
                        termination, sigint_callback, EvaluationMonitor,\
                        StepMonitor, **kwds):
    """Helper function used by solvers to process constraints keywords and 
set the solver up with constraints imposed using the specified method.

Inputs:
    solver_instance -- a mystic solver object.
    constraints -- string of symbolic constraints.
    costfunc -- the function or method to be minimized.
    termination -- callable object providing termination conditions.
    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval
        after the end of a solver iteration.

Further Inputs:
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...
    constraints_method -- string name of constraints method. Valid method
        names are ['direct', 'barrier', 'penalty', 'auglag'].
"""
    #XXX Should the eqcon_funcs/ineqcon_funcs interface have the user input a
    # list of functions or one function that returns a list of results?

    import types

    # Default values for optional keyword arguments
    varname = 'x'
    varnamelist = None
    constraints_method = 'auglag' # Is this a suitable default?

    # Get user-input keyword arguments. Delete them so the remaining kwds
    # can be passed back to the solver.
    if kwds.has_key('varname'): 
        varname = kwds['varname']
        del kwds['varname']
    if kwds.has_key('varnamelist'): 
        varnamelist = kwds['varnamelist']
        del kwds['varnamelist']
    if kwds.has_key('constraints_method'): 
        constraints_method = kwds['constraints_method']
        del kwds['constraints_method']

    returnflag = False
    method = constraints_method
    x0 = solver_instance.population[0]
    ndim = len(x0)
    constraints_beforestrict = constraints

    # If there are strict constraints, wrap the cost function to ensure that
    # they are never violated, like SetStrictRanges, except not just for
    # box bounds. If x0 violates these 'strict' constraints,
    # raise an error. Note: currently only works for inequalities. Would using
    # a logarithmic barrier be better?
    #TODO Should also check for if there are logs or fractional exponents in 
    # constraints, and generate domain constraints if there are?
    constraints_unstrict = ""
    constraints_strict = []
    if type(constraints) == str:
        if constraints.find('strict') != -1:
            lines = constraints.splitlines()
            for line in lines:
                if line:
                    if line.find('strict') != -1:
                        strict_con = (line.rstrip('strict').\
                                      rstrip().rstrip(',')) + '\n'
                        # Replace varname and set up for calling 'eval' later
                        if varnamelist:
                            strict_con = _replace_variable_names(strict_con, \
                                                             varnamelist)
                            varname = '$'
                        # Iterate in reverse in case ndim > 9.
                        indices = list(range(1, ndim+1))
                        indices.reverse()
                        for i in indices:
                            strict_con = strict_con.replace(varname + str(i), \
                                                        'x[' + str(i-1) + ']') 

                        # If it is an equality, put it into constraints_unstrict
                        if strict_con.find('>') == -1 and strict_con.find('<') == -1:
                            #print "Strict constraints are for inequalities only."
                            constraints_unstrict += line.rstrip('strict').rstrip().\
                                                    rstrip(',') + '\n'
                        else:
                            constraints_strict.append(strict_con.strip())

                    else:
                        constraints_unstrict += line.rstrip('strict').rstrip().\
                                                rstrip(',') + '\n'
            constraints = constraints_unstrict
        
    # If type of constraints is a function, use the direct method.
    if type(constraints) == types.FunctionType:
        return [solver_instance, costfunc, constraints, returnflag]

    # Check that the method is valid
    valid_methods = ['direct', 'barrier', 'penalty', 'auglag']
    if valid_methods.count(method) == 0:
        print "Method" + str(method) + "is not a valid option. Proceeding \
with `penalty' method."
        method = 'penalty'

    # If direct and symbolic string input, pass to form_constraints_function
    if method == 'direct':
        if type(constraints) == str:
            suggestedorder = None
            if kwds.has_key('suggestedorder'): 
                suggestedorder = kwds['suggestedorder']

            # Throw an exception if there are ineqcon_funcs or eqcon_funcs
            if kwds.has_key('ineqcon_funcs'):
                raise Exception("The keyword 'ineqcon_funcs' cannot be used \
with the `direct' method.")
            if kwds.has_key('eqcon_funcs'):
                raise Exception("The keyword 'eqcon_funcs' cannot be used \
with the `direct' method.")

            # Also throw an exception if there are inequalities
            if constraints.find('>') != -1 or constraints.find('<') != -1:
                raise Exception("Inequality constraints are not supported \
with the symbolic interface.")

            constraints = form_constraints_function(constraints, ndim, \
                            varname=varname, varnamelist=varnamelist,\
                            suggestedorder=suggestedorder)
            # If there are strict constraints, wrap them. 
            if constraints_strict:
                def strict_costfunc(x):
                    for constraint in constraints_strict:
                        if not eval(constraint):
                            return inf
                    return costfunc(x)
                return [solver_instance, strict_costfunc, constraints, returnflag]

            # Use slack variables to handle inequality constraints. This doesn't
            # seem to work very well right now, so commenting it out.
            """# Sort into inequality and equality constraints
            lines = constraints.splitlines()
            new_constraints = ""
            n_slack = 0 # Number of slack variables (number of inequalities)
            for line in lines:
                if line.strip():
                    constraint = line.strip()
                    split = constraint.split('>')
                    direction = '>'
                    if len(split) == 1:
                        split = constraint.split('<')
                        direction = '<'
                    # If equality, just re-append it to constraints string
                    if len(split) == 1:
                        split = constraint.split('=')
                        direction = '='
                        new_constraints += constraint + '\n'
                    if len(split) == 1:
                        print "Invalid constraint: ", constraint
                    if direction == '<':
                        n_slack += 1
                        converted_constr = split[0] + '-' + varname + str(ndim + \
                                n_slack) + '=' + split[1].lstrip('=')
                        new_constraints += converted_constr + '\n'
                    if direction == '>':
                        n_slack += 1
                        converted_constr = split[0] + '+' + varname + str(ndim + \
                                n_slack) + '=' + split[1].lstrip('=')
                        new_constraints += converted_constr + '\n'

            if varnamelist != None:
                varnamelist += [varname + str(ndim + j) for j in range(1, n_slack+1)]

            if suggestedorder != None:
                suggestedorder += range(ndim + 1, ndim + n_slack)

            constraints_func = form_constraints_function(new_constraints, \
                            ndim + n_slack, varname=varname, varnamelist=varnamelist,\
                            suggestedorder=suggestedorder)

            def new_costfunc(x):
                for constraint in constraints_strict:
                    if not eval(constraint):
                        return inf
                return costfunc(x[:ndim])

            # Reset some of the solver attributes to match the larger ndim
            solver_instance.nDim = ndim + n_slack
            solver_instance.population = [[0.0 for i in range(ndim + n_slack)] \
                                        for j in range(solver_instance.nPop)]
            solver_instance.SetInitialPoints(list(x0) + [0.]*n_slack)
            if solver_instance._useStrictRange:
                solver_instance.SetStrictRanges(list(solver_instance._strictMin) + \
                        [0.]*n_slack, list(solver_instance._strictMax) + [1e20]*n_slack)

            # Do all actual solving here
            solver_instance.Solve(costfunc, termination, constraints=constraints_func,\
                                  constraints_method='direct', **kwds)

            # Now remove the slack variables so that the solver attributes are again
            # of the original dimension. One problem here is that StepMonitor will still
            # capture the larger-dimensional parameters.
            #print solver_instance.bestSolution
            solver_instance.bestSolution = solver_instance.bestSolution[:-1]
            solver_instance.trialSolution = solver_instance.trialSolution[:-1]
            if solver_instance._useStrictRange:
                solver_instance._strictMin = solver_instance._strictMin[:-1]
                solver_instance._strictMax = solver_instance._strictMax[:-1]
            solver_instance.nDim = ndim
            solver_instance.population = [solver_instance.population[j][:-1] \
                                        for j in range(solver_instance.nPop)]
            returnflag = True"""
            
    if method == 'barrier':
        # Check if x0 satisfies constraints.
        ineqcon_funcs = []
        eqcon_funcs = []
        if kwds.has_key('ineqcon_funcs'): ineqcon_funcs = kwds['ineqcon_funcs']
        if kwds.has_key('eqcon_funcs'): eqcon_funcs = kwds['eqcon_funcs']
        if not verify_constraints_satisfied(constraints, x0, varname=varname,\
                            disp=False, varnamelist=varnamelist, eqcon_funcs=\
                            eqcon_funcs, ineqcon_funcs=ineqcon_funcs)[0]:
            # If constraints are not satisfied, do an optimization to find
            # a feasible point. If one is found, continue optimizing with that.
            # Should print a message to say a preliminary optimization is being
            # performed?
            feasible_point = find_feasible_point(constraints_beforestrict, len(x0),\
                          eqcon_funcs=eqcon_funcs, ineqcon_funcs=ineqcon_funcs,\
                                varname=varname, varnamelist=varnamelist,\
                                lb=solver_instance._strictMin, x0=x0, \
                                ub=solver_instance._strictMax)
            if feasible_point != None:
                solver_instance.population[0] = feasible_point
            #else:
            #    print "No feasible point could be found."
            # Some solvers seem to do ok even though the 'no feasible point'
            # message is printed, so for now, don't print.

        solverinstance = solver_instance
        end_solver = sumt(constraints, ndim, costfunc, \
                solverinstance, termination, varname=varname,\
                varnamelist=varnamelist, strict_constraints=constraints_strict,\
                StepMonitor=StepMonitor, EvaluationMonitor=EvaluationMonitor,\
                iterated_barrier=True, **kwds)
        solver_instance = end_solver
        returnflag = True

    if method == 'penalty':
        costfunc = wrap_constraints_to_cost(constraints, ndim, \
                          costfunc, varname=varname,\
                          varnamelist=varnamelist, \
                          strict_constraints=constraints_strict, **kwds)
        constraints = lambda x: x
        #XXX solver.bestEnergy does not give costfunc(bestSolution)! It gives
        # wrapped_costfunc(bestSolution), so if the constraints are not 
        # satisfied well, it can be inaccurate. Perhaps the solving should be 
        # wrapped in this function so that this can be adjusted for?

    # If using auglag, pass the solver to the augmented_lagrangian function,
    # which may do several optimizations using that solver. end_solver below
    # will already contain the solution, and thus returnFlag is set to True so
    # the solver that called this _process_constraints method will terminate.
    if method == 'auglag':
        # Pass the random state so that Rand1Bin works properly when there is no
        # random seed.
        randomstate = random.getstate() 

        solverinstance = solver_instance
        end_solver = sumt(constraints, ndim, costfunc, \
                solverinstance, termination, varname=varname,\
                varnamelist=varnamelist, strict_constraints=constraints_strict,\
                StepMonitor=StepMonitor, EvaluationMonitor=EvaluationMonitor,\
                randomstate=randomstate, **kwds)
        solver_instance = end_solver
        returnflag = True

    return [solver_instance, costfunc, constraints, returnflag]

#------------------------------------------------------------------------
# Functions for the `direct' method of imposing constraints. May contain
# some unused/unuseful code. For example, it is not highly recommended to input
# x0 and bounds to form_constraints_function or those types of functions,
# but let the code remain for now in case it is useful in certain special
# situations.

try:
    from itertools import permutations
except ImportError:
    def permutations(iterable, r=None):
        """return successive r-length permutations of elements in the iterable.
Produces a generator object.

For example, 
    >>> print list( permutations(range(3),2) ) 
    [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    >>> print list( permutations(range(3)) )
    [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        """
        # code from http://docs.python.org/library/itertools.html
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return
        indices = range(n)
        cycles = range(n, n-r, -1)
        yield tuple(pool[i] for i in indices[:r])
        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i+1:] + indices[i:i+1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return
        return

def form_constraints_directly(constraints_str, ndim, varname='x',\
                              varnamelist=None):
    """Build a constraints function given a constraints string. 
Returns a constraints function.

Inputs:
    constraints_str -- a string, where each line is a constraint equation.
        For each equation, one variable must be isolated on the left side.
    ndim -- number of variables. Includes xi not explicit in constraints_str.

    For example:
        >>> constraints = '''x1 = cos(x2) + 2.
        ... x2 = x3*2.'''
        >>> f = form_constraints_directly(constraints, 3)
        >>> f([1.0, 0.0, 1.0])
        [3.0, 2.0, 1.0]

    NOTE: By default, variables are named {x1, x2, ..., xn}. However, the base
        variable name can be changed from 'x' to any string using "varname".

    NOTE: Regular python math conventions are used. For example, if an 'int'
        is used in a constraint equation, one or more variable may be evaluate
        to an 'int' -- this can affect solved values for the variables.

Additional Inputs:
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.
"""
    if varnamelist:
        constraints_str = _replace_variable_names(constraints_str, varnamelist)
        varname = '$'

    # Parse the string
    parsed = ""
    lines = constraints_str.splitlines()
    for line in lines:
        fixed = line
        # Iterate in reverse in case ndim > 9.
        indices = list(range(1, ndim+1))
        indices.reverse()
        for i in indices:
            fixed = fixed.replace(varname + str(i), '_params[' + str(i-1) + ']') 
        parsed += fixed.strip() + '\n'
    #FIXME: parsed throws SyntaxError in cf if LHS has more than one variable

    #print parsed # debugging

    # form the constraints function
    def cf(params):
        try:
            _params = params.copy() # for numpy arrays
        except:
            _params = params[:]
        exec parsed in globals(), locals()
        return _params

    return cf

def form_constraints_linear(equations_string, ndim, x0=None, \
                            lower_bounds=None, upper_bounds=None, \
                            varname='x', disp=True, details=False, \
                            suggestedorder=None, varnamelist=None, **kwds):
    """Build a constraints function given a string of linear constraints.
Returns a constraints function. 

Inputs:
    equations_string -- a string, where each line is a constraint equation.
    ndim -- number of parameters. Includes xi not explicit in equations_string.

    For example:
        >>> constraints = '''x1 = cos(x2) + 2.
        ... x2 = x3*2.'''
        >>> f = form_constraints_linear(constraints, 3)
        >>> f([1.0, 0.0, 1.0])
        [3.0, 2.0, 1.0]

Additional Inputs:
    x0 -- list of initial parameter values.
    lower_bounds -- list of lower bounds on parameters.
    upper_bounds -- list of upper bounds on parameters.
    details -- boolean for whether or not to also return 'variable_info'.
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.

    suggestedorder -- tuple containing the order in which the variables should
        be solved for. The first 'neqns' variables will be independent, and
        the rest will be dependent.

    NOTE: For example, if suggestedorder=(3, 1, 2) and there are two
        constraints equations, x3 and x1 will be constrained in terms of x2.
        By default, increasing order (i.e. 1, 2, ...) is used. suggestedorder
        must enumerate all variables, hence len(sugestedorder) == ndim.

    NOTE: For optimization problem with initial values and either lower or
        upper bounds, the constraints function must be formulated in a manner
        such that it does not immediately violate the given bounds.
"""
    """
    For example:
        >>> '''x2 = x1 - 2.
        ... x2 = x3*2.'''

    should be identical to:
        >>> '''x1 = x2 + 2.
        ... x2 = x3*2.'''

    should be identical to:
        >>> '''x1 = x2 + 2.
        ... x3 = 0.5*x2'''

    should be identical to:
        >>> '''x2 - x1 = -2.
        ... x2 - 2.*x3 = 0.'''

    Returns a constraints function with the constraints simplified nicely.
    In this example, the constraints become:

    def f(params):
        params[1] = 2.0*params[2]
        params[0] = 2.0 + 2.0*params[2]
        return params

    This is good for systems of equations, which must be linear for Sympy to
    simplify them. However, using the nonlinear all-purpose equation 
    inverter/simplifier will probably also work! It just won't get rid of
    redundancies, but that should be ok.
"""
    warn = True  # if True, don't supress warning about old versions of sympy
    if kwds.has_key('warn'): warn = kwds['warn']

    if varnamelist:
        equations_string = _replace_variable_names(equations_string, varnamelist)
        varname = '$'

    # The following code attempts to construct something like:
    # x1 = Symbol('x1')
    # x2 = Symbol('x2')
    # x3 = Symbol('x3')
    # eq1 = Eq(x2, x1 - 2.)
    # eq2 = Eq(x2, x3*2.)
    # soln = solve([eq2, eq1], [x1, x2, x3])

    # If no Sympy installed, just call form_constraints_directly
    try:
        from sympy import Eq, solve, Symbol
    except ImportError:
        if warn: print "Warning: sympy not installed."# Equation will not be simplified."
        return form_constraints_directly(equations_string, ndim, varname=varname)

    code, left, right, xlist, neqns = _sympy_setup(equations_string, ndim, varname)

    eqlist = ""
    for i in range(1, neqns + 1):
        eqn = 'eq' + str(i)
        eqlist += eqn + ","
        code += eqn + '= Eq(' + left[i-1] + ',' + right[i-1] + ')\n'

    # Figure out if trying various permutations is necessary
    strict = False
    if (x0 and lower_bounds) or (x0 and upper_bounds):
        strict = True

    xinlist = xlist.split(',')[:-1]
    if suggestedorder:
        suggestedorder = tuple(asarray(suggestedorder) - 1)
        xorder = list(asarray(xinlist)[list(suggestedorder)])

    if strict:
        # If there are strict bounds and initial x value, form a constraints 
        # function for each permutation.
        # For sympy, change the order of the x variables passed to solve()
        # to get different variables solved for.
        solns = []
        xperms = list(permutations(xinlist)) #XXX Gets stuck here if ndim is 
                                             # on the order of 10....
        if suggestedorder:
            xperms.remove(xorder)
            xperms.insert(0, xorder)
        for perm in xperms: 
            tempcode = ""
            tempcode += code
            xstring = ""
            for item in perm:
                xstring += item + ","
            tempcode += 'soln = solve([' + eqlist.rstrip(',') + '], [' + \
                        xstring.rstrip(',') + '])'
            print tempcode
            exec tempcode in globals(), locals()

            if soln == None:
                print "Constraints seem to be inconsistent..."

            solvedstring = ""
            for key, value in soln.iteritems():
                solvedstring += str(key) + ' = ' + str(value) + '\n'
            solns.append(solvedstring)

        # Create strings of all permutations of the solved equations.
        # First remove duplicates, then take permutations of the lines of equations
        # to create equations in different orders.
        noduplicates = list(set(solns)) 
        stringperms = []
        for item in noduplicates:
            spl = item.splitlines()
            for perm in permutations(spl):
                permstring = ""
                for line in perm:
                    permstring += line + '\n'
                stringperms.append(permstring)

        # feed each solved set of equations into form_constraints_directly to get 
        # constraints functions, and check if constraints(x0) is inside the bounds.
        for string in stringperms:
            cf = form_constraints_directly(string, ndim, varname=varname)
            compatible = constraints_inside_bounds(cf, x0, lower_bounds=lower_bounds,\
                                                   upper_bounds=upper_bounds)
            if compatible:
                #print string # Debugging
                if details:
                    info = get_variable_info(string, ndim, varname=varname)
                    return [cf, info]
                else:
                    return cf# Return the first compatible constraints function
            else:
                constraints = cf # Save a value for later and try other permutations
        # Perhaps raising an Exception is better? But sometimes it's ok...
        print 'Warning: x0 does not satisfy both constraints and bounds.\n\
Constraints function may be faulty.'
        if details:
            info = get_variable_info(string, ndim, varname=varname)
            return [cf, info]
        return cf

    else:
        # If no bounds and x0, no need for permutations. Just form code and get whatever
        # solution sympy comes up with.
        if suggestedorder:
            xlist = ','.join(xorder)
        code += 'soln = solve([' + eqlist.rstrip(',') + '], [' + xlist.rstrip(',') + '])'
        exec code in globals(), locals()
        if soln == None:
            print "Constraints seem to be inconsistent..."

        solvedstring = ""
        for key, value in soln.iteritems():
            solvedstring += str(key) + ' = ' + str(value) + '\n'

        cf = form_constraints_directly(solvedstring, ndim, varname=varname)
        if details:
            info = get_variable_info(solvedstring, ndim, varname=varname)
            return [cf, info]
        return cf


def constraints_inside_bounds(cf, x, lower_bounds=None, upper_bounds=None):
    """returns False if cf(x) is outside given bounding box, True otherwise.

Inputs:
    cf -- a function of x.
    x -- a list of parameters.

Additional Inputs:
    lower_bounds -- list of lower bounds on parameters.
    upper_bounds -- list of upper bounds on parameters.
"""
    from mystic.tools import wrap_bounds
    wrapped = wrap_bounds(cf, min=lower_bounds, max=upper_bounds)
    if wrapped(cf(x)) == inf:
        return False    
    else:
        return True

def verify_constraints_consistent(x, constraints, tol=1e-5):
    """returns True if constraints(x) is equal to x (i.e. is self-consistent).

Inputs:
    x -- a list of parameters.
    constraints -- a function of x.

Additional Inputs:
    tol -- absolute error tolerance for each parameter in x.
"""
    from mystic.math import approx_equal
    constrained = constraints(x)
    for i in range(len(x)):
        if not approx_equal(constrained[i], x[i], tol=tol):
            return False
    return True

def get_variable_info(equations, ndim, varname='x'): 
    """Takes a string of constraint equations and determines which variables
are dependent, independent, and unconstrained. Assumes the string is already 
simplified to the form: '''xi = expression''', and there are no inequalities.
Returns a dictionary with keys: 'dependent', 'independent', and 'unconstrained',and values that enumerate the variables that match each variable type.

Inputs:
    equations -- a string, where each line is a constraint equation.
    ndim -- number of variables. Includes xi not explicit in equations.

    For example:
        >>> equations = '''x1 = x5**2
        ... x3 = x4 + x5'''
        >>> print get_variable_info(equations, 5)
        {'dependent':[1, 3], 'independent':[4, 5], 'unconstrained':[2]}

Additional Inputs:
    varname -- variable name. Default is 'x'.
"""
    eqns = equations.splitlines()
    variables = range(1, ndim+1)
    dep = []
    indep = []
    for eqn in eqns:
        if eqn:
            split = eqn.split('=')
            for var in variables:
                if split[0].find(varname + str(var)) != -1:
                    dep.append(var)
                    variables.remove(var)
                    break
            for var in range(1, ndim+1):
                if variables.count(var) != 0:
                    if split[1].find(varname + str(var)) != -1:
                        indep.append(var)
                        variables.remove(var)
    #FIXME: Known bug is as follows...
    """
    >>> equations = '''x1 = x5**2 
    ... x3 = x4 + x5'''
    >>> get_variable_info(equations, 5)
    {'dependent': [1, 3], 'independent': [4, 5], 'unconstrained': [2]}
    >>> equations = '''x1 = x5**2
    ... x3 - x5 = x4'''
    >>> get_variable_info(equations, 5)
    {'dependent': [1, 3], 'independent': [4, 5], 'unconstrained': [2]}

    >>> equations = '''x1 = x5**2
    ... x3 - x4 = x5'''
    >>> get_variable_info(equations, 5)
    {'dependent': [1, 3], 'independent': [5], 'unconstrained': [2, 4]}
    >>> equations = '''x1 = x5**2
    ... x3 - x4 - x5 = 0'''
    >>> get_variable_info(equations, 5)
    {'dependent': [1, 3], 'independent': [5], 'unconstrained': [2, 4]}
    """
    dep.sort()
    indep.sort()
    d = {'dependent':dep, 'independent':indep, 'unconstrained':variables}
    return d

def get_variables(equations_string, varname='x'):
    """extract a list of the string variable names from equations_string

Inputs:
    equations_string -- string of symbolic constraints.

    For example:
        >>> equations_string = '''x1 + x2 = x3*4
        ... x3 = x2*x4'''
        >>> print get_variables(equations_string)
        ['x1', 'x2', 'x3', 'x4'] 

Additional Inputs:
    varname -- base variable name. Default is 'x'.
"""
    import re
    target = varname+'[0-9]+'
    variables = []
    equation_list = equations_string.splitlines()
    for equation in equation_list:
        vars = re.findall(target,equation)
        for var in vars:
            if var not in variables:
                variables.append(var)
    return variables

def _sympy_setup(equations, ndim, varname='x'):
    """Parse an equation string and prepare input for sympy. Returns a tuple
of sympy-specific input. Used as a helper function for form_constraints_linear
and form_constraints_nonlinear.

Inputs:
    equations -- a string, where each line is a constraint equation.
    ndim -- number of variables. Includes xi not explicit in equations.
    
Additional Inputs:
    varname -- variable name. Default is 'x'.
"""

    # split constraints_str into lists of left hand sides and right hand sides
    eacheqn = equations.splitlines()
    neqns = 0
    left = []
    right = []
    for eq in eacheqn:
        splitlist = eq.split('=')
        if len(splitlist) == 2:

            # If equation is blank on one side, raise error.
            if len(splitlist[0].strip()) == 0 or len(splitlist[1].strip()) == 0:
                print eq, "is not an equation!" # Raise exception?
            else:
                left.append(splitlist[0])
                right.append(splitlist[1])
                neqns += 1

        # If equation doesn't have one equal sign, raise error.
        if len(splitlist) != 2 and len(splitlist) != 1:
            print eq, "is not an equation!" # Raise exception?

    # First create list of x variables
    xlist = ""
    for i in range(1, ndim + 1):
        xn = varname + str(i)
        xlist += xn + ","

    # Start constructing the code string
    code = ""
    for i in range(1, ndim + 1):
        xn = varname + str(i)
        code += xn + '=' + "Symbol('" + xn + "')\n"

    code += "rand = Symbol('rand')\n"

    return code, left, right, xlist, neqns

#FIXME: must set ndim >= max(i) for all xi in equation_string
#       might want eqn = "x2 = 1. + x4"... and only use ndim = 2.
#       could ndim = len(get_variables(eqn)) be used to implement this?
def simplify(equation_string, ndim, varname='x', target=None, **kwds):
    """Solve a single equation for each variable found within the equation.
Returns permutations of the equation_string, solved for each variable.  If
the equations fail to simplify, the original equation is returned.

    equation_string -- a string representing a single constraint equation.
        Only a single equation should be provided.
    ndim -- number of variables. Includes variables not explicit in equation_string.

    For example:
        >>> equation = "x2 - 3. = x1*x3"
        >>> print simplify(equation, 3)
        x3 = -(3.0 - x2)/x1
        x2 = 3.0 + x1*x3
        x1 = -(3.0 - x2)/x3

Additional Inputs:
    varname -- variable name. Default is 'x'.
    target -- specify the variable to isolate on the left side upon return.

    For example:
        >>> equation = "x2 - 3. = x1*x3"
        >>> print simplify(equation, 3, target='x2')
        x2 = 3.0 + x1*x3
"""
    warn = True  # if True, don't supress warning about old versions of sympy
    if kwds.has_key('warn'): warn = kwds['warn']

    try:
        from sympy import Eq, solve, Symbol
    except ImportError:
        if warn: print "Warning: sympy not installed."# Equation will not be simplified."
        return equation_string
       #return form_constraints_directly(equation_string, ndim, varname=varname)

    code, left, right, xlist, neqns = _sympy_setup(equation_string, ndim, varname)

    code += 'eq = Eq(' + left[0] + ',' + right[0] + ')\n' 

    if not target:
        code += 'soln = solve(eq, [' + xlist + '])\n'
    else:
        code += 'soln = solve(eq, [' + target + '])\n'

    try: 
        exec code in globals(), locals()
    except NotImplementedError: # catch 'multivariate' error for older sympy
        if warn: print "Warning: sympy could not simplify equation."
       #if warn: print "Warning: older sympy version. Equation will not be simplified."
        return equation_string
       #return form_constraints_directly(equation_string, ndim, varname=varname)

    if not target:
        solvedstring = ""
        for key, value in soln.iteritems():
            if value:
                for v in value:
                    solvedstring += str(key) + ' = ' + str(v) + '\n'
        #XXX Not the best way to handle multiple solutions?
        return solvedstring
    #XXX: consistent to return string (above) and None (below)?
    else:
        if not soln:
            print "Target variable is invalid. Returning None."
            return None
        return target + ' = ' + str(soln[0])


def form_constraints_function(equations_string, ndim, x0=None, \
                              lower_bounds=None, upper_bounds=None, \
                              varname='x', details=False,\
                              suggestedorder=None, varnamelist=None):
    """Build a constraints function given a constraints string. 
Returns a constraints function.

Inputs:
    equations_string -- a string, where each line is a constraint equation.
    ndim -- number of parameters. Includes xi not explicit in equations_string.

    For example:
        >>> constraints = '''x1 = cos(x2) + 2.
        ... x2 = x3*2.'''
        >>> f = form_constraints_function(constraints, 3)
        >>> f([1.0, 0.0, 1.0])
        [3.0, 2.0, 1.0]

    NOTE: Will attempt to process equations_string as linear constraints first;
        if fails, will attempt to process as nonlinear constraints.

Additional Inputs:
    x0 -- list of initial parameter values.
    lower_bounds -- list of lower bounds on parameters.
    upper_bounds -- list of upper bounds on parameters.
    details -- boolean for whether or not to also return 'variable_info'.
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.

    suggestedorder -- tuple containing the order in which the variables should
        be solved for. The first 'neqns' variables will be independent, and
        the rest will be dependent.

    NOTE: For example, if suggestedorder=(3, 1, 2) and there are two
        constraints equations, x3 and x1 will be constrained in terms of x2.
        By default, increasing order (i.e. 1, 2, ...) is used. suggestedorder
        must enumerate all variables, hence len(sugestedorder) == ndim.

    NOTE: For optimization problem with initial values and either lower or
        upper bounds, the constraints function must be formulated in a manner
        such that it does not immediately violate the given bounds.
"""
    try:
        return form_constraints_linear(equations_string, ndim, x0=x0, \
                      lower_bounds=lower_bounds, upper_bounds=upper_bounds,\
                      varname=varname, details=details, \
                      suggestedorder=suggestedorder, varnamelist=varnamelist)
    except:
        return form_constraints_nonlinear(equations_string, ndim, x0=x0,\
                lower_bounds=lower_bounds, upper_bounds=upper_bounds,\
                varname=varname, details=details, varnamelist=varnamelist,\
                suggestedorder=suggestedorder)


def form_constraints_nonlinear(equations_string, ndim, varname='x', x0=None,\
                               lower_bounds=None, upper_bounds=None, \
                               details=False, suggestedorder=None, \
                               allperms=False, varnamelist=None):
    """Build a constraints function given a string of nonlinear constraints.
Returns a constraints function. 

Inputs:
    equations_string -- a string, where each line is a constraint equation.
    ndim -- number of parameters. Includes xi not explicit in equations_string.

    For example:
        >>> constraints = '''x2 = x4*3. + (x1*x3)**x1'''
        >>> f = form_constraints_nonlinear(constraints, 4)
        >>> f([1.0, 1.0, 1.0, 1.0])
        [1.0, 4.0, 1.0, 1.0]

Additional Inputs:
    x0 -- list of initial parameter values.
    lower_bounds -- list of lower bounds on parameters.
    upper_bounds -- list of upper bounds on parameters.
    details -- boolean for whether or not to also return 'variable_info'.
    allperms -- boolean for whether or not to return a list of constraints
        functions composed from all permutations of solutions of the given
        constraints equations.
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.

    suggestedorder -- tuple containing the order in which the variables should
        be solved for. The first 'neqns' variables will be independent, and
        the rest will be dependent.

    NOTE: For example, if suggestedorder=(3, 1, 2) and there are two
        constraints equations, x3 and x1 will be constrained in terms of x2.
        By default, increasing order (i.e. 1, 2, ...) is used. suggestedorder
        must enumerate all variables, hence len(sugestedorder) == ndim.

    NOTE: For optimization problem with initial values and either lower or
        upper bounds, the constraints function must be formulated in a manner
        such that it does not immediately violate the given bounds.
"""
    warn = True  # if True, don't supress warning about old versions of sympy
    if varnamelist:
        equations_string = _replace_variable_names(equations_string, varnamelist)
        varname = '$'

    # Figure out if trying various permutations is necessary
    strict = False
    if (x0 and lower_bounds) or (x0 and upper_bounds):
        strict = True

    eqns = equations_string.splitlines()
    # Remove empty strings:
    actual_eqns = []
    for j in range(len(eqns)):
        if eqns[j].strip():
           actual_eqns.append(eqns[j].strip())
    orig_eqns = actual_eqns[:]

    neqns = len(actual_eqns)

    # Getting all permutations will take a really really long time for ndim
    # as low as 10.
    perms = list(permutations(range(ndim)))
    if suggestedorder: # Try the suggested order first.
        suggestedorder = tuple(asarray(suggestedorder) - 1)
        perms.remove(suggestedorder)
        perms.insert(0, suggestedorder)

    complete_list = []

    constraints_function_list = []
    # Some of the permutations will give the same answer; look into reducing the number
    # of repeats?
    for p in perms:
        thisorder = p
        # Sort the list actual_eqns so that any equation containing x1 is first, etc.
        sorted_eqns = []
        actual_eqns_copy = orig_eqns[:]
        usedvars = []
        for i in thisorder:#range(ndim):
            variable = varname + str(i+1)
            for eqn in actual_eqns_copy:
                if eqn.find(variable) != -1:
                    sorted_eqns.append(eqn)
                    actual_eqns_copy.remove(eqn)
                    usedvars.append(variable)
                    break
        if actual_eqns_copy: # Append the remaining equations
            for item in actual_eqns_copy:
                sorted_eqns.append(item)
        actual_eqns = sorted_eqns

        # Append the remaining variables to usedvars
        tempusedvar = usedvars[:]
        tempusedvar.sort()
        nmissing = ndim - len(tempusedvar)
        for m in range(1, nmissing+1):
            usedvars.append(varname + str(len(tempusedvar) + m))

        for i in range(neqns):
            # Trying to use xi as a pivot. Loop through the equations looking for one
            # containing xi.
            target = usedvars[i]
            for eqn in actual_eqns[i:]:
                invertedstring = simplify(eqn, ndim, varname=varname, target=target, warn=warn)
                if invertedstring:
                    warn = False
                    break
            # substitute into the remaining equations. the equations' order in the list
            # newsystem is like in a linear coefficient matrix.
            newsystem = ['']*neqns
            j = actual_eqns.index(eqn)
            newsystem[j] = eqn
            othereqns = actual_eqns[:j] + actual_eqns[j+1:]
            for othereqn in othereqns:
                expression = invertedstring.split("=")[1]
                fixed = othereqn.replace(target, '(' + expression + ')')
                k = actual_eqns.index(othereqn)
                newsystem[k] = fixed
            actual_eqns = newsystem
            
        # Invert so that it can be fed properly to form_constraints_directly
        simplified = []
        for eqn in actual_eqns:
            target = usedvars[actual_eqns.index(eqn)]
            simplified.append(simplify(eqn, ndim, varname=varname, target=target, warn=warn))

        cf = form_constraints_directly('\n'.join(simplified), ndim, varname=varname) 

        if allperms:
            complete_list.append(cf)
            continue

        if not strict:
            if details:
                info = get_variable_info('\n'.join(simplified), ndim, varname=varname)
                return [cf, info]
            else:
                return cf
        else:
            try: # catches trying to square root a negative number, for example.
                cf(x0)
            except ValueError:
                continue
            compatible = constraints_inside_bounds(cf, x0, lower_bounds=\
                 lower_bounds, upper_bounds=upper_bounds)
            satisfied = verify_constraints_satisfied(equations_string, cf(x0), disp=False)
            if compatible and satisfied:
                if details:
                    info = get_variable_info('\n'.join(simplified), ndim, varname=varname)
                    return [cf, info]
                else:
                    return cf

    print 'Warning: x0 does not satisfy both constraints and bounds, or constraints\n\
may not be enforced correctly. Constraints function may be faulty.'
    if details:
        info = get_variable_info('\n'.join(simplified), ndim, varname=varname)
        return [cf, info]

    if allperms:
        return complete_list
    return cf

#-------------------------------------------------------------------
# Method for successive `helper' optimizations
 
#FIXME: interface needs some cleaning.
def sumt(constraints_string, ndim, costfunc, solverinstance, term, \
         varname='x', eps = 1e-4, max_SUMT_iters = 10, \
         StepMonitor=Null, EvaluationMonitor=Null, sigint_callback=None, \
         varnamelist=None, eqcon_funcs=[], ineqcon_funcs=[], \
         strict_constraints = [], iterated_barrier=False, **kwds):
    """solves several successive optimization problems, with the goal of 
producing results that also satisfy the given constraints. The cost function
is slightly altered for each attempt. Produces a solver instance with the
optimal information contained in the solver methods. For example, the value
of the optimizers can be obtained with 'solver.Solution()' for the resuting
solver.

Inputs:
    constraints_string -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints_string = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...

    ndim -- number of variables. Includes xi not explicit in constraints_string.
        Should correspond to len(x) for x in costfunc(x).
    costfunc -- the cost function to be minimized.
    solverinstance -- a mystic solver object.
    term -- callable object providing termination conditions.

Additional Inputs:
    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval
        after the end of a solver iteration.
    varname -- variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'width', 'height'] which specifies
        the variable names used in the constraints string. The variable names
        must be provided in the same order as in the constraints string.

    ineqcon_funcs -- list of 'inequality' functions, f, where f(x) <= 0.
    eqcon_funcs -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints_string' and provide ineqcon_funcs and/or eqcon_funcs.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: ineqcon_funcs=[f1], eqcon_funcs=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    eps -- tolerance for exiting the outer optimization loop. If the absolute
       value of the improvement in cost is less than eps, exit. Default is 1e-4.
    max_SUMT_iters -- maximum number of outer loop iterations to perform. 
       Default is 10.
    strict_constraints -- list of constraint strings, where if any constraint
        is violated, the penalty is set to infinity (i.e. cost(x) = inf).
    iterated_barrier -- boolean for whether or not to use a logarithmic barrier         for inequality constraints and a quadratic penalty for equality
        constraints (as opposed to lagrangian terms). Default is False.

Further inputs:
    rh -- initial penalty parameter for equality constraints. Default is 20 for
        the 'auglag' method, and 100 for 'iterated barrier' and 'iterated
        penalty' methods.
    rg -- initial penalty parameter for inequality constraints. Default is 20
        for the 'auglag' method, and 200 for 'iterated barrier' and 'iterated
        penalty' methods.
    ch -- scaling factor for the equality penalty parameter. Default is 5.
    cg -- scaling factory for the inequality penalty parameter. Default is 5.
    randomstate -- random state. Typically obtained with random.getstate().
    iterated_penalty -- boolean similar to 'iterated_barrier'; however, in this
        case, use a infinite sharp barrier. Will override 'iterated_barrier'.
        Default is False.

References:
    [1] Applied Optimization with MATLAB Programming, by Venkataraman.
        Wiley, 2nd edition, 2009.
    [2] "An Augmented Lagrange Multiplier Based Method for Mixed Integer
        Discrete Continuous Optimization and Its Applications to Mechanical
        Design", by Kannan and Kramer. 1994.
"""
    from mystic.math import approx_equal
    disp = False # to match the defaults in the solvers
    iterated_penalty = False
    randomstate = None
    if kwds.has_key('disp'): disp = kwds['disp']
    if kwds.has_key('iterated_penalty'): disp = kwds['iterated_penalty']
    if kwds.has_key('randomstate'): disp = kwds['randomstate']

    # kwds whose defaults vary depending on other kwds. iterated_penalty 
    # currently is not used in the constraints interface in Solve().
    if iterated_barrier or iterated_penalty:
        rg = 200. 
        rh = 100.
        cg = 5. 
        ch = 5.
    else: # augmented Lagrangian.
        rg = 20. 
        rh = 20. 
        cg = 5.  
        ch = 5.  
    if kwds.has_key('rg'): rg = kwds['rg']
    if kwds.has_key('rh'): rh = kwds['rh']
    if kwds.has_key('cg'): cg = kwds['cg']
    if kwds.has_key('ch'): ch = kwds['ch']        

    if varnamelist:
        constraints_string = _replace_variable_names(constraints_string, varnamelist)
        varname = '$'

    # Parse the constraints string
    lines = constraints_string.splitlines()
    eqconstraints = []
    ineqconstraints = []
    for line in lines:
        if line.strip():
            fixed = line
            # Iterate in reverse in case ndim > 9.
            indices = list(range(1, ndim+1))
            indices.reverse()
            for i in indices:
                fixed = fixed.replace(varname + str(i), 'x[' + str(i-1) + ']') 
            constraint = fixed.strip()
            # Replace 'mean' with actual expression for calculating mean
            if constraint.find('mean') != -1:
                constraint = constraint.replace('mean', 'average(asarray(x))')
            # Replace 'range' with actual expression for calculating range
            if constraint.find('range') != -1:
                constraint = constraint.replace('range', 'max(x) - min(x)')
            # Sorting into equality and inequality constraints, and making all
            # inequality constraints in the form expression <= 0. and all 
            # equality constraints of the form expression = 0.
            split = constraint.split('>')
            direction = '>'
            if len(split) == 1:
                split = constraint.split('<')
                direction = '<'
            if len(split) == 1:
                split = constraint.split('=')
                direction = '='
            if len(split) == 1:
                print "Invalid constraint: ", constraint
            expression = split[0].rstrip('=') + '-(' + split[1].lstrip('=') + ')'
            if direction == '=':
                eqconstraints.append(expression)
            elif direction == '<':
                ineqconstraints.append(expression)
            else:
                ineqconstraints.append('-(' + expression + ')')

    neqcons = len(eqconstraints) + len(eqcon_funcs)
    nineqcons = len(ineqconstraints) + len(ineqcon_funcs)
    nstringcons = len(eqconstraints) + len(ineqconstraints)

    if iterated_penalty:
        def wrapped_costfunc(x):
            if strict_constraints:
                for constraint in strict_constraints:
                    if not eval(constraint):
                        return inf
            result = costfunc(x)
            # For constraints that were input symbolically
            for constraint in ineqconstraints:
                result += float(rg)*max(0., eval(constraint))**2
            for constraint in eqconstraints:
                result += float(rh)*eval(constraint)**2

            # For constraints in function form
            for constraint in ineqcon_funcs:
                result += float(rg)*max(0., constraint(x))**2
            for constraint in eqcon_funcs:
                result += float(rh)*constraint(x)**2
            return result

    elif iterated_barrier:
        def wrapped_costfunc(x):
            if strict_constraints:
                for constraint in strict_constraints:
                    if not eval(constraint):
                        return inf
            y = costfunc(x)
            # For constraints that were input symbolically
            for constraint in ineqconstraints:
                c = eval(constraint)
                if c <= 0.:
                    y += -1./rg*log(-c)
                else:
                    return inf
            # Handle equality constraints with quadratic penalty
            for constraint in eqconstraints:
                y += float(rh)*eval(constraint)**2

            # For constraints in function form
            for constraint in ineqcon_funcs:
                if constraint(x) <= 0.:
                    y += -1./rg*log(-constraint(x))
                else:
                    return inf
            # Handle equality constraints with quadratic penalty
            for constraint in eqcon_funcs:
                y += float(rh)*constraint(x)**2
            return y

    else: # Do augmented Lagrangian
        lam = [0.]*neqcons  # Initial lambda and beta. 
        beta = [0.]*nineqcons

        # Implementing eqn 7.24, page 349 of Venkataraman.
        # This is the modified cost function with the lagrangian/penalty terms.
        def wrapped_costfunc(x):
            if strict_constraints:
                for constraint in strict_constraints:
                    if not eval(constraint):
                        return inf
            result = costfunc(x)
            # For constraints that were input symbolically
            tempsum = 0.
            for k in range(len(eqconstraints)):
                h = eval(eqconstraints[k])
                tempsum += h**2
            result += rh*tempsum
            for k in range(len(eqconstraints)):
                h = eval(eqconstraints[k])
                result += lam[k]*h
            tempsum2 = 0.
            for j in range(len(ineqconstraints)):
                g = eval(ineqconstraints[j])
                tempsum2 += max(g, -beta[j]/(2.*rg))**2
            result += rg*tempsum2
            for j in range(len(ineqconstraints)):
                g = eval(ineqconstraints[j])
                result += beta[j]*max(g, -beta[j]/(2.*rg))
            # For constraints in function form
            tempsum = 0.
            for k in range(len(eqcon_funcs)):
                h = eqcon_funcs[k](x)
                tempsum += h**2
            result += rh*tempsum
            for k in range(len(eqcon_funcs)):
                h = eqcon_funcs[k](x)
                result += lam[nstringcons + k]*h
            tempsum2 = 0.
            for j in range(len(ineqcon_funcs)):
                g = ineqcon_funcs[j](x)
                tempsum2 += max(g, -beta[nstringcons + j]/(2.*rg))**2
            result += rg*tempsum2
            for j in range(len(ineqcon_funcs)):
                g = ineqcon_funcs[j](x)
                result += beta[nstringcons + j]*max(g, \
                        -beta[nstringcons + j]/(2.*rg))
            return result

    F_values = []
    x_values = []
    total_iterations = 0.
    total_energy_history = []
    exitflag = 0
    # The outer SUMT for loop. In each iteration, an optimization will be 
    # performed.
    for q in range(max_SUMT_iters):
        if q == 0:
            x0 = solverinstance.population[0]
        else:
            x0 = x_values[q-1]

        if disp:
            print 'SUMT iteration %d: x = ' % q, x0

        if randomstate:
            random.setstate(randomstate)

        # 'Clear' some of the attributes of the solver
        solver = solverinstance
        solver.population = [[0.0 for i in range(solver.nDim)] for j in range(solver.nPop)]
        solver.generations = 0
        solver.bestEnergy = 0.0
        solver.bestSolution = [0.0] * solver.nDim
        solver.trialSolution = [0.0] * solver.nDim
        solver._init_popEnergy  = 1.0E20 #XXX: or numpy.inf?
        solver.popEnergy = [solver._init_popEnergy] * solver.nPop
        solver.energy_history = []

        # Perform the 'inner' optimization
        solver.SetInitialPoints(x0)
        solver.Solve(wrapped_costfunc, term, StepMonitor=StepMonitor,\
                    EvaluationMonitor=EvaluationMonitor,\
                    sigint_callback=sigint_callback, **kwds)
        x = solver.Solution()

        # When a user sets maxiter and maxfun, that is actually the maxiter 
        # and maxfun *each* SUMT iteration. However, it is difficult to check 
        # total maxfun and maxiter each SUMT iteration because then what 
        # would the inner maxiter and maxfun be? Better to just leave maxiter 
        # and maxfun as limits for each inner optimization...
        total_iterations += solver.generations
        total_energy_history += solver.energy_history
        
        F_values.append(wrapped_costfunc(x)) 
        x_values.append(x)

        # For the signal handler
        if solver._EARLYEXIT:
            exitflag = 3
            break

        # Commenting out breaking if constraints are satisfied, as that seems 
        # to cause premature exiting sometimes. 
        """# If constraints are satisifed, break
        exit = True
        for constraint in ineqconstraints:
            if not eval(constraint) <= 0.0:
                exit = False
        for constraint in eqconstraints:
            if not eval(constraint) == 0.0:
                exit = False
        if exit:
            exitflag = 1
            break"""

        # If not improving, break
        if q != 0:
            if abs(F_values[q] - F_values[q-1]) <= eps:
                exitflag = 2
                break
            #if linalg.norm(asarray(x_values[q])-asarray(x_values[q-1])) <= eps:
            #    exitflag = 4
            #    break

        # Update parameters
        if not (iterated_barrier or iterated_penalty): # if augmented lagrangian 
            # For constraints that were input symbolically
            for i in range(len(eqconstraints)):
                lam[i] += 2.*rh*eval(eqconstraints[i])
            for j in range(len(ineqconstraints)):
                beta[j] += 2.*rg*max(eval(ineqconstraints[j]), -beta[j]/(2.*rg))

            # For constraints in function form
            for i in range(len(eqcon_funcs)):
                lam[nstringcons + i] += 2.*rh*eqcon_funcs[i](x)
            for j in range(len(ineqcon_funcs)):
                beta[nstringcons + j] += 2.*rg*max(ineqcon_funcs[j](x),\
                                       -beta[nstringcons + j]/(2.*rg))
        rh *= ch
        rg *= cg
        
    if disp:
        if exitflag == 0:
            print "SUMT exited because maximum number of SUMT iterations reached."
        elif exitflag == 1:
            print "SUMT exited because constraints were satisfied."
        elif exitflag == 2:
            print "SUMT exited because F values reached tolerance for improving."
        elif exitflag == 3:
            print "SUMT exited because signal handler 'exit' was selected."
        #elif exitflag == 4:
        #    print "SUMT exited because x values reached tolerance for improving."    
        else:
            print "SUMT exited for other reason."
        print "%d SUMT iterations were performed." % (q + 1)

    # Return the last solver, which contains all of the important information
    # First modify the solver to include total data.
    solver.generations = total_iterations
    solver.energy_history = total_energy_history

    #XXX solver.bestEnergy is not exactly costfunc(solver.bestSolution),
    # since it is wrapped_costfunc(bestSolution). Compensate for that
    # here? Then, it is inconsistent with solver.energy_history, which 
    # also records wrapped_costfunc(bestSolution)....
    #solver.bestEnergy = costfunc(solver.bestSolution) 
    return solver

if __name__ == '__main__':
    pass

#EOF
