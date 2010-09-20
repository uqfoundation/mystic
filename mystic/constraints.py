#!/usr/bin/env python
#
# coded by Alta Fang, 2010
# 
# updated by mmckerns@caltech.edu
#FIXME: Major improvement would be to use a constraints class,
#       where __call__ works like the current constraints function
#       then, most of the following functions would take constraints objects
#       and configuration could be seperated from the below methods.
#       This should improve speed and clarity.
"""
Tools for imposing constraints on an optimization.

The main functions exported are::
    - rnorm: calculate the norm of residual errors.
    - issolution: checks if a set of parameters x satisfies the constraints.
    - linear_symbolic: converts constraints from matrix form to a symbolic
          string that can be input into a constraints-function generator.
    - wrap_constraints: takes a string of constraints of any form, and
          imposes the constraints on a function using the 'penalty' method.

Several 'advanced' functions are also included::
    - solve: tries to find a point to satisfy given constraints.
    - isbounded: check if func(x) evaluates to inside the bounds.
    - simplify_symbolic: solves an equation for each given variable, or,
          optionally, for a single target variable.
    - get_variables: extract a list of the string variable names from the given
          equations string.
    - parse: creates a constraints function from a symbolic
          string. This only handles equality constraints, and no mean or range
          constraints. It first tries to call `parse_linear', and
          if that fails, then tries `parse_nonlinear'. It returns a
          function that imposes the constraints on an x value, and this
          function can then be passed to the solver with the `constraints'
          keyword in solver.Solve. This function is called when using the
          `direct' method.
    - parse_simplified: parses a string into a constraints function.
    - parse_linear: parses a string of linear equations into a
          constraints function.
    - parse_nonlinear: parses a string of equations that are not
          necessarily linear into a constraints function.
    - sumt: solves several successive optimization problems with slightly
          different cost functions each time, and produces a solver instance
          containing all of the optimal information.
"""
from __future__ import division
from math import *
from numpy import *
import random
from mystic.math import approx_equal
from mystic.tools import Null, list_or_tuple_or_ndarray, permutations

def linear_symbolic(A=None, b=None, G=None, h=None):
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
            eqstring += Asum.rstrip(' + ') + ' = ' + str(b[i]) + '\n'

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
            ineqstring += Gsum.rstrip(' + ') + ' <= ' + str(h[i]) + '\n'
    totalconstraints = ineqstring + eqstring
    return totalconstraints 


def issolution(constraints, solution, tol=1e-3, **kwds):
    """Returns whether constraints were violated or not within some tolerance. 

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...

    solution -- list of parameter values proposed to solve the constraints.
    tol -- size limit of norm of residual errors for which constraints
        are considered to be solved. Default is 1e-3.

Further Inputs:
    verbose -- True to print details for each of the equality or
        inequality constraints. Default is False.
    fineq -- list of 'inequality' functions, f, where f(x) <= 0.
    feq -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints' and provide fineq and/or feq.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: fineq=[f1], feq=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
    """
    #XXX: see 'rnorm' for not about constraints as a function (not a string)
    if rnorm(constraints, solution, **kwds) <= tol:
        return True
    return False


def rnorm(constraints, solution, verbose=False,
          variables='x', feq=[], fineq=[]):
    """Calculates the amount of constraints violation (norm of residual errors).

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...

    solution -- list of parameter values proposed to solve the constraints.

Additional Inputs:
    verbose -- True to print details for each of the equality or
        inequality constraints. Default is False.
    fineq -- list of 'inequality' functions, f, where f(x) <= 0.
    feq -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints' and provide fineq and/or feq.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: fineq=[f1], feq=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
    """
    #FIXME: needs to be optimized for speed
    from mystic.math import approx_equal
    if not isinstance(constraints, str):
        #FIXME: The following is undocumented in the interface.
        #       Get 'rms error', where constraits is a function.
        #       The kwds = ['tol'] are active. All other kwds are ignored.
        #       Alternately, use constraints='' and a f built with feq & fineq.
        error = 0.0
        constrained = constraints(solution)
        for i in range(len(solution)):
            error += (constrained[i] - solution[i])**2  #XXX: not really rnorm
        return error**0.5

    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraints, variables)
        variables = '$'

    ndim = len(solution)

    # Iterate in reverse in case ndim > 9.
    indices = list(range(ndim))
    indices.reverse()
    for i in indices:
        variable = variables + str(i+1)
        constraints = constraints.replace(variable, 'solution[' + str(i) + ']')

    constraints_list = constraints.splitlines()

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
            myerror = 'None'
            mean = average(asarray(solution))
            split = item.split('>')
            if len(split) == 1:
                split = item.split('<')
            if len(split) == 1:
                split = item.split('=')
                item = item.replace('=', '==')
            if not eval(item):
                # Only add to error if the constraint is not satisfied
                myerror = (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
                error += myerror
            if verbose: 
                print 'actual mean =', mean, '   error:', myerror
            continue
        if item.find('range') != -1:
            myerror = 'None'
            r = max(solution) - min(solution)
            item = item.replace('range', 'r')
            split = item.split('>')
            if len(split) == 1:
                split = item.split('<')
            if len(split) == 1:
                split = item.split('=')
                item = item.replace('=', '==')
            if not eval(item):
                # Only add to error if the constraint is not satisfied
                myerror = (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
                error += myerror
            if verbose:
                print 'actual range =', r, '   error:', myerror
            continue
        myerror = 'None'
        split = item.split('>')
        if len(split) != 1:
            if not eval(split[0] + '>' + split[1]):
                # Only add to error if the constraint is not satisfied
                myerror = (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
                error += myerror
            if verbose:
                print eval(split[0]), '?>=', eval(split[1].lstrip(' = ')), '   error:', myerror
            continue
        myerror = 'None'
        split = item.split('<')
        if len(split) != 1:
            if not eval(split[0] + '<' + split[1]): 
                # Only add to error if the constraint is not satisfied
                myerror = (eval(split[0].rstrip(' = ')) - eval(split[1].lstrip(' = ')))**2
                error += myerror
            if verbose:
                print eval(split[0]), '?<=', eval(split[1].lstrip('=')), '   error:', myerror
            continue
        myerror = 'None'
        split = item.split('=')
        if len(split) != 1:
            if not eval(split[0]) == eval(split[1]):
                # Only add to error if the constraint is not satisfied
                myerror = (eval(split[0]) - eval(split[1]))**2
                error += myerror
            if verbose:
                print eval(split[0]), '?=', eval(split[1]), '   error:', myerror
            continue

    # Loop through feq and fineq
    for func in feq:
        f = func(solution)
        myerror = f**2
        error += myerror  #XXX: always add to error? (as opposed to above)
        if verbose:
            print f, '?= 0.0', '   error:', myerror
    for func in fineq:
        f = func(solution)
        myerror = max(0., f)**2
        error += myerror  #XXX: always add to error? (as opposed to above)
        if verbose:
            print f, '?<= 0.0', '   error:', myerror

    return error**0.5


def wrap_constraints(constraints, func, variables='x', \
                     feq=[], fineq=[], penalty=1e4, strict=[]):
    """Wraps a function with a set of constraints. The constraints are
imposed using the 'penalty' method, using a fixed penalty parameter. Returns
a function with the constraints built-in.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...

    func -- the function to be constrained.

Additional Inputs:
    fineq -- list of 'inequality' functions, f, where f(x) <= 0.
    feq -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints' and provide fineq and/or feq.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: fineq=[f1], feq=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.

    penalty -- penalty multiplier if the constraints are violated. Default
        is 1e4. It is not recommended to use infinity.
    strict -- list of constraint strings, where if any constraint is
        violated, the penalty is set to infinity (i.e. func(x) = inf).
        Each string must be a single line of valid python (usable by eval).

References: 
    [1] http://en.wikipedia.org/wiki/Penalty_method
    [2] Applied Optimization with MATLAB programming by Venkataraman,
        section 7.2.1: Exterior Penalty Function Method
    [3] http://www.srl.gatech.edu/education/ME6103/Penalty-Barrier.ppt
"""
    #FIXME: need to add the abilty to take a constraints function constrain(x)
    #       and wrap func(x) with constrain(x).  (see rnorm)
    #       Alternately, use constraints='' and a f built with feq & fineq.
   #from mystic.tools import src
   #ndim = len(get_variables(src(func), variables))

    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraints, variables)
        ndim = len(variables)
        variables = '$'
    else:
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(variables)) for v in myvar])
        else: ndim = 0

    # Parse the constraints string
    lines = constraints.splitlines()
    eqconstraints = []
    ineqconstraints = []
    for line in lines:
        if line.strip():
            fixed = line
            # Iterate in reverse in case ndim > 9.
            indices = list(range(1, ndim+1))
            indices.reverse()
            for i in indices:
                fixed = fixed.replace(variables + str(i), 'x[' + str(i-1) + ']') 
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
    # with func(x)*penalty is no good because that is too sensitive
    # to the func value. Increasing the penalty by some fixed factor 
    # if result > penalty*0.5, for example, is too abrupt and also arbitrary.
    def wrapped_func(x):
        for constraint in strict:
            if not eval(constraint):
                return inf
        result = func(x)
        # For constraints that were input symbolically
        for constraint in ineqconstraints:
            result += float(penalty)*max(0., eval(constraint))**2
        for constraint in eqconstraints:
            result += float(penalty)*eval(constraint)**2
        # For constraints in function form
        for constraint in fineq:
            result += float(penalty)*max(0., constraint(x))**2
        for constraint in feq:
            result += float(penalty)*constraint(x)**2
        return result

    return wrapped_func


def solve(constraints, guess=None, lower_bounds=None, upper_bounds=None, \
          nvars=None, variables='x', feq=[], fineq=[]):
    """Use optimization to find a solution to a set of constraints. Returns
the solution, if one is found. If no solution is found, returns None.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...

Additional Inputs:
    guess -- list of parameter values proposed to solve the constraints.
    lower_bounds -- list of lower bounds on solution values.
    upper_bounds -- list of upper bounds on solution values.
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x4' in the example above).
    fineq -- list of 'inequality' functions, f, where f(x) <= 0.
    feq -- list of 'equality' functions, f, where f(x) == 0.

    NOTE: If all constraints are in functional form, enter an empty string
        for 'constraints' and provide fineq and/or feq.

    For example, the constraint equations f1(x) <= 0 and f2(x) == 0
        would be entered as: fineq=[f1], feq=[f2]
        >>> def f1(x): return -x[0] - x[1] + 2.
        ...
        >>> def f2(x): return x[0] + x[1]
        ...

    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
    """
    #FIXME: need to add the abilty to take a constraints function constrain(x)
    #       and solve for x.  (see rnorm)
    #       Alternately, use constraints='' and a f built with feq & fineq.

    if list_or_tuple_or_ndarray(variables):
        varnamelist = variables
        varname = 'x'
        ndim = len(variables)
    else:
        varnamelist = None
        varname = variables
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar])
        else: ndim = 0
    if nvars: ndim = nvars
    elif guess: ndim = len(guess)
    elif lower_bounds: ndim = len(lower_bounds)
    elif upper_bounds: ndim = len(upper_bounds)

    def dummy_func(x):
        return 1.

    from mystic.differential_evolution import DifferentialEvolutionSolver
    from mystic.termination import ChangeOverGeneration as COG
    solver = DifferentialEvolutionSolver(ndim, 40)
    if guess != None:
        solver.SetInitialPoints(guess)
    else:
        if lower_bounds and upper_bounds:
            solver.SetRandomInitialPoints(lower_bounds, upper_bounds)
        else:
            solver.SetRandomInitialPoints()
    if lower_bounds != None and len(lower_bounds) > 0 and \
       upper_bounds != None and len(upper_bounds) > 0:
        solver.SetStrictRanges(lower_bounds, upper_bounds)
    #solver.enable_signal_handler()
    solver.Solve(dummy_func, COG(), constraints=constraints, \
                 eqcon_funcs=feq, ineqcon_funcs=fineq, varname=varname,\
                 varnamelist=varnamelist, constraints_method='penalty')
    soln = solver.Solution()
    if not issolution(constraints, soln, variables=variables, verbose=False, \
                      feq=feq, fineq=fineq):
        soln = None
    return soln


def substitute_symbolic(constraints, varnamelist, variables='$'):
    """Replace variables in constraints string with a marker '$i',
where i = 1,2,3,...  Returns a modified constraints string.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).
    varnamelist -- list of variable name strings. The variable names must
        be provided in the same order as in the constraints string.

    For example:
        >>> varnamelist = ['spam', 'eggs']
        >>> constraints = '''spam + eggs - 42'''
        >>> print substitute_symbolic(constraints, varnamelist)
        '$1 + $2 - 42'

Additional Inputs:
    variables -- variable name. Default is '$'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
"""
    # substitite one list of strings for another
    if list_or_tuple_or_ndarray(variables):
        equations = substitute_symbolic(constraints,varnamelist,'_')
        vars = get_variables(equations,'_')
        indices = [int(v.strip('_'))-1 for v in vars]
        for i in range(len(vars)):
            equations = equations.replace(vars[i],variables[indices[i]])
        return equations

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

    # Default is variables='$', as '$' is not a special symbol in Python,
    # and it is unlikely a user will choose it for a variable name.
    if variables in varnamelist:
        marker = '_$$$$$$$$$$' # even less likely...
    else:
        marker = variables

    '''Bug demonstrated here:
    >>> equation = """x3 = max(y,x) + x"""
    >>> vars = ['x','y','z','x3']
    >>> print substitute_symbolic(equation,vars)
    '$4 = ma$1($2,$1) + $1'
    '''
    for i in indices:
        constraints = constraints.replace(varnamelist[i], marker + str(i+1))
    return constraints.replace(marker, variables)


def isbounded(func, x, lower_bounds=None, upper_bounds=None):
    """return False if func(x) evaluates outside the bounds, True otherwise.

Inputs:
    func -- a function of x.
    x -- a list of parameters.

Additional Inputs:
    lower_bounds -- list of lower bounds on parameters.
    upper_bounds -- list of upper bounds on parameters.
"""
    from mystic.tools import wrap_bounds
    wrapped = wrap_bounds(func, min=lower_bounds, max=upper_bounds)
    if wrapped(func(x)) == inf:
        return False    
    else:
        return True

def classify_variables(constraints, nvars=None, variables='x'): 
    """Takes a string of constraint equations and determines which variables
are dependent, independent, and unconstrained. Assumes the string is already 
simplified to the form: '''xi = expression''', and there are no inequalities.
Returns a dictionary with keys: 'dependent', 'independent', and 'unconstrained',and values that enumerate the variables that match each variable type.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints = '''x1 = x5**2
        ...     x3 = x4 + x5'''
        >>> print classify_variables(constraints, 5)
        {'dependent':[1, 3], 'independent':[4, 5], 'unconstrained':[2]}

Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x2' in the example above).
    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
"""
    #XXX: use simplify_symbolic first if not in form xi = ... ?
    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraints, variables)
        varname = '$'
        ndim = len(variables)
    else:
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar])
        else: ndim = 0
    if nvars: ndim = nvars

    eqns = constraints.splitlines()
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
    #FIXME: This is a bug, as non-simplified eqations don't throw errors
    """Bugs (?) demonstrated here:
    >>> constraints = '''x1 = x5**2 
    ...     x3 = x4 + x5'''
    >>> classify_variables(constraints, 5)
    {'dependent': [1, 3], 'independent': [4, 5], 'unconstrained': [2]}
    >>> constraints = '''x1 = x5**2
    ...     x3 - x5 = x4'''
    >>> classify_variables(constraints, 5)
    {'dependent': [1, 3], 'independent': [4, 5], 'unconstrained': [2]}

    >>> constraints = '''x1 = x5**2
    ...     x3 - x4 = x5'''
    >>> classify_variables(constraints, 5)
    {'dependent': [1, 3], 'independent': [5], 'unconstrained': [2, 4]}
    >>> constraints = '''x1 = x5**2
    ...     x3 - x4 - x5 = 0'''
    >>> classify_variables(constraints, 5)
    {'dependent': [1, 3], 'independent': [5], 'unconstrained': [2, 4]}
    """
    dep.sort()
    indep.sort()
    d = {'dependent':dep, 'independent':indep, 'unconstrained':variables}
    return d

def get_variables(constraints, variables='x'):
    """extract a list of the string variable names from constraints string

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints = '''x1 + x2 = x3*4
        ...     x3 = x2*x4'''
        >>> print get_variables(constraints)
        ['x1', 'x2', 'x3', 'x4'] 

Additional Inputs:
    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
"""
    if list_or_tuple_or_ndarray(variables):
        equations = substitute_symbolic(constraints,variables,'_')
        vars = get_variables(equations,'_')
        indices = [int(v.strip('_'))-1 for v in vars]
        varnamelist = []
        for i in sort(indices):
            varnamelist.append(variables[i])
        return varnamelist

    import re
    target = variables+'[0-9]+'
    varnamelist = []
    equation_list = constraints.splitlines()
    for equation in equation_list:
        vars = re.findall(target,equation)
        for var in vars:
            if var not in varnamelist:
                varnamelist.append(var)
    return varnamelist

def _prepare_sympy(constraints, nvars=None, variables='x'):
    """Parse an equation string and prepare input for sympy. Returns a tuple
of sympy-specific input.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    For example:
        >>> constraints = '''
        ...     x1**2 = 2.5*x2 - 5.0
        ...     exp(x3/x1) >= 7.0'''
        ...
    
Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x4' in the example above).
    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
"""
    #FIXME: if constraints contain x1,x2,x4 should x3 be in code,xlist?
    #FIXME: _3 appears in equation if nvars=2,variables=x1,x2,x4. Should it?
    '''Bugs (?) demonstrated here:
    >>> print _prepare_sympy(equation)
    ("x1=Symbol('x1')\nx2=Symbol('x2')\nx3=Symbol('x3')\nx4=Symbol('x4')\nrand = Symbol('rand')\n", ['x2 + 3. '], [' x1*x4'], 'x1,x2,x3,x4,', 1)

    >>> print _prepare_sympy(equation,2,variables=['x1','x2','x4'])
    ("_1=Symbol('_1')\n_2=Symbol('_2')\nrand = Symbol('rand')\n", ['_2 + 3. '], [' _1*_3'], '_1,_2,', 1)
    '''

    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraints, variables, variables='_')
        varname = '_'
        ndim = len(variables)
    else:
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar])
        else: ndim = 0
    if nvars: ndim = nvars

    # split constraints_str into lists of left hand sides and right hand sides
    eacheqn = constraints.splitlines()
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

def simplify_symbolic(constraint, variables='x', target=None, **kwds):
    """Solve a single equation for each variable found within the constraint.
Returns permutations of the constraint equation, solved for each variable.
If the equation fails to simplify, the original equation is returned.

    constraint -- a string representing a single constraint equation.
        Only a single constraint equation should be provided.

    For example:
        >>> equation = "x2 - 3. = x1*x3"
        >>> print simplify_symbolic(equation)
        x3 = -(3.0 - x2)/x1
        x2 = 3.0 + x1*x3
        x1 = -(3.0 - x2)/x3

Additional Inputs:
    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.

    target -- specify the variable to isolate on the left side upon return.

    For example:
        >>> equation = "x2 - 3. = x1*x3"
        >>> print simplify_symbolic(equation, target='x2')
        x2 = 3.0 + x1*x3
"""
    #FIXME: different behavior for target='x' (NameError) & target='xi' (None)
    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraint, variables, variables='_')
        varname = '_'
        ndim = len(variables)
        if variables.count(target):
            target = substitute_symbolic(target, variables, variables='_')
    else:
        constraints = constraint # constraints used below
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraint, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar])
        else: ndim = 0
    '''Bug (?) demonstrated here:
    >>> equation = 'x2 + 3. = x1*x4'
    >>> print simplify_symbolic(equation,target='x2')
    x2 = -3.0 + x1*x4
    >>> print simplify_symbolic(equation,target='x3')
    Target variable is invalid. Returning None.
    None
    >>> print simplify_symbolic(equation,target='x')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "constraints.py", line 948, in simplify_symbolic
        exec code in globals(), locals()
      File "<string>", line 7, in <module>
    NameError: name 'x' is not defined
    '''
    warn = True  # if True, don't supress warning about old versions of sympy
    if kwds.has_key('warn'): warn = kwds['warn']

    try:
        from sympy import Eq, Symbol
        from sympy import solve as symsol
    except ImportError: # Equation will not be simplified."
        if warn: print "Warning: sympy not installed."
        return constraint

    code,left,right,xlist,neqns = _prepare_sympy(constraints, ndim, varname)

    code += 'eq = Eq(' + left[0] + ',' + right[0] + ')\n' 

    if not target:
        code += 'soln = symsol(eq, [' + xlist + '])\n'
    else:
        code += 'soln = symsol(eq, [' + target + '])\n'

    try: 
        exec code in globals(), locals()
    except NotImplementedError: # catch 'multivariate' error for older sympy
        if warn: print "Warning: sympy could not simplify equation."
        return constraint

    #XXX Not the best way to handle multiple solutions?
    if not target:
        solvedstring = ""
        for key, value in soln.iteritems():
            if value:
                for v in value:
                    solvedstring += str(key) + ' = ' + str(v) + '\n'
        if solvedstring: solvedstring = solvedstring[:-1]
    #XXX: consistent to return string (above) and None (below)?
    else:
        if not soln:
            print "Target variable is invalid. Returning None."
            return None
        solvedstring = target + ' = ' + str(soln[0])
    # replace '_' with the original variable names
    if list_or_tuple_or_ndarray(variables):
        vars = get_variables(solvedstring,'_')
        indices = [int(v.strip('_'))-1 for v in vars]
        for i in range(len(vars)):
            solvedstring = solvedstring.replace(vars[i],variables[indices[i]])
    return solvedstring

def parse_simplified(constraints, variables='x', **kwds):
    """Build a constraints function given a constraints string. 
Returns a constraints function.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    NOTE: For each equation, one variable must be isolated on the left side.
        Thus, an equation is of the form "xi = expression".

    For example:
        >>> constraints = '''x1 = cos(x2) + 2.
        ...     x2 = x3*2.'''
        >>> f = parse_simplified(constraints)
        >>> f([1.0, 0.0, 1.0])
        [3.0, 2.0, 1.0]

    NOTE: Regular python math conventions are used. For example, if an 'int'
        is used in a constraint equation, one or more variable may be evaluate
        to an 'int' -- this can affect solved values for the variables.

Additional Inputs:
    variables -- variable name. Default is 'x'. Also, a list of variable
        name strings are accepted. Use a list if variable names don't have
        the same base name.

    NOTE: For example, if constraints = '''length = height**2 - 3*width''',
        we will have variables = ['length', 'height', 'width'] which
        specifies the variable names used in the constraints string. The
        variable names must be provided in the same order as in the
        constraints string.
"""
    #FIXME: should handle guess,lower_bounds,upper_bounds ?
    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraints, variables, '_')
        ndim = len(variables)
        varname = '_'
    else:
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(variables)) for v in myvar])
        else: ndim = 0
        varname = variables

    # Parse the string
    parsed = ""
    lines = constraints.splitlines()
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


def parse_linear(constraints, variables='x', suggestedorder=None, **kwds):
    """Build a constraints function given a string of linear constraints.
Returns a constraints function. 

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    FIXME: 'math' throws an 'AttributeError' -- try 'parse_simplified'.

    For example:
        >>> constraints = '''x1 = x2 + 2.
        ...     x2 = x3*2.'''
        >>> f = parse_linear(constraints)
        >>> f([1.0, 0.0, 1.0])
        [4.0, 2.0, 1.0]

Additional Inputs:
    suggestedorder -- tuple containing the order in which the variables should
        be solved for. The first 'neqns' variables will be independent, and
        the rest will be dependent.

    NOTE: For example, if suggestedorder=(3, 1, 2) and there are two
        constraints equations, x3 and x1 will be constrained in terms of x2.
        By default, increasing order (i.e. 1, 2, ...) is used. suggestedorder
        must enumerate all variables, hence len(sugestedorder) == nvars.

Further Inputs:
    details -- boolean for whether or not to also return 'variable_info'.
    guess -- list of parameter values proposed to solve the constraints.
    lower_bounds -- list of lower bounds on solution values.
    upper_bounds -- list of upper bounds on solution values.

    NOTE: For optimization problems with initial values and either lower or
        upper bounds, the constraints function must be formulated in a manner
        such that it does not immediately violate the given bounds.
    """
    """
    Returns a constraints function with the constraints simplified nicely.
    For example, the constraints function is:

    def f(params):
        params[1] = 2.0*params[2]
        params[0] = 2.0 + 2.0*params[2]
        return params

    This is good for systems of equations, which must be linear for Sympy to
    simplify them. However, using the nonlinear all-purpose equation 
    inverter/simplifier will probably also work! It just won't get rid of
    redundancies, but that should be ok.
    """
    # FIXME: exec seems to fail on 'math' functions...
    #        workaound might be to catch AttributeError to 'parse_simplified'
    nvars = None # number of variables. Should be determined automatically
    strict = False # if True, force to use 'permutations' code
    warn = True  # if True, don't supress warning about old versions of sympy
    verbose = False # if False, keep information to a minimum
    details = False # if True, print details from classify_variables
    guess = None
    upper_bounds = None
    lower_bounds = None
    #-----------------------undocumented-------------------------------
    if kwds.has_key('nvars'): nvars = kwds['nvars']
    if kwds.has_key('strict'): strict = kwds['strict']
    if kwds.has_key('warn'): warn = kwds['warn']
    if kwds.has_key('verbose'): verbose = kwds['verbose']
    #------------------------------------------------------------------
    if kwds.has_key('details'): details = kwds['details']
    if kwds.has_key('guess'): guess = kwds['guess']
    if kwds.has_key('upper_bounds'): upper_bounds = kwds['upper_bounds']
    if kwds.has_key('lower_bounds'): lower_bounds = kwds['lower_bounds']

    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraints, variables, '_')
        ndim = len(variables)
        varname = '_'
    else:
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(variables)) for v in myvar])
        else: ndim = 0
        varname = variables
    if nvars: ndim = nvars
    elif guess: ndim = len(guess)
    elif lower_bounds: ndim = len(lower_bounds)
    elif upper_bounds: ndim = len(upper_bounds)

    # The following code attempts to construct something like:
    # >>> from sympy import Eq, Symbol
    # >>> from sympy import solve as symsol
    # >>> x1 = Symbol('x1')
    # >>> x2 = Symbol('x2')
    # >>> x3 = Symbol('x3')
    # >>> eq1 = Eq(x2, x1 - 2.)
    # >>> eq2 = Eq(x2, x3*2.)
    # >>> soln = symsol([eq2, eq1], [x1, x2, x3])

    # If no Sympy installed, just call parse_simplified
    try:
        from sympy import Eq, Symbol
        from sympy import solve as symsol
    except ImportError:
        if warn: print "Warning: sympy not installed."# Equation will not be simplified."
        return parse_simplified(constraints, variables=varname)

    code, left, right, xlist, neqns = _prepare_sympy(constraints, ndim, varname)

    eqlist = ""
    for i in range(1, neqns + 1):
        eqn = 'eq' + str(i)
        eqlist += eqn + ","
        code += eqn + '= Eq(' + left[i-1] + ',' + right[i-1] + ')\n'

    # Figure out if trying various permutations is necessary
    if (guess and lower_bounds) or (guess and upper_bounds):
        strict = True

    xinlist = xlist.split(',')[:-1]
    if suggestedorder:
        suggestedorder = tuple(asarray(suggestedorder) - 1)
        xorder = list(asarray(xinlist)[list(suggestedorder)])

    if strict:
        # If there are strict bounds and initial x value, form a constraints 
        # function for each permutation.
        # For sympy, change the order of the x variables passed to symsol()
        # to get different variables solved for.
        solns = []
        xperms = list(permutations(xinlist)) #XXX Gets stuck here if nvars
                                             # is on the order of 10....
        if suggestedorder:
            xperms.remove(xorder)
            xperms.insert(0, xorder)
        for perm in xperms: 
            tempcode = ""
            tempcode += code
            xstring = ""
            for item in perm:
                xstring += item + ","
            tempcode += 'soln = symsol([' + eqlist.rstrip(',') + '], [' + \
                        xstring.rstrip(',') + '])'
            if verbose: print tempcode
            exec tempcode in globals(), locals()

            if soln == None and warn:
                print "Warning: constraints seem to be inconsistent."

            solvedstring = ""
            for key, value in soln.iteritems():
                solvedstring += str(key) + ' = ' + str(value) + '\n'
            solns.append(solvedstring)

        # Create strings of all permutations of the solved equations.
        # Remove duplicates, then take permutations of the lines of equations
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

        # Feed each solved set of equations into parse_simplified to get 
        # constraints functions. Check if constraints(guess) is in the bounds.
        for string in stringperms:
            cf = parse_simplified(string, variables=varname)
            if guess: compatible = isbounded(cf, guess, \
                                             lower_bounds=lower_bounds,\
                                             upper_bounds=upper_bounds)
            else: compatible = True
            if compatible:
                #print string # Debugging
                if details:
                    info = classify_variables(string, ndim, variables=varname)
                    return [cf, info]
                else:
                    return cf# Return the first compatible constraints function
            else:
                _constraints = cf # Save for later and try other permutations
        # Perhaps raising an Exception is better? But sometimes it's ok...
        warning='Warning: guess does not satisfy both constraints and bounds.\n\
Constraints function may be faulty.'
        if warn: print warning
        if details:
            info = classify_variables(string, ndim, variables=varname)
            return [cf, info]
        return cf

    else:
        # If no bounds and guess, no need for permutations.
        # Just form code and get whatever solution sympy comes up with.
        if suggestedorder:
            xlist = ','.join(xorder)
        code += 'soln = symsol([' + eqlist.rstrip(',') + '], [' + xlist.rstrip(',') + '])'
        exec code in globals(), locals()
        if soln == None and warn:
            print "Warning: constraints seem to be inconsistent."

        solvedstring = ""
        for key, value in soln.iteritems():
            solvedstring += str(key) + ' = ' + str(value) + '\n'

        cf = parse_simplified(solvedstring, variables=varname)
        if details:
            info = classify_variables(solvedstring, ndim, variables=varname)
            return [cf, info]
        return cf

def parse(constraints, variables='x', suggestedorder=None, **kwds):
    """Build a constraints function given a constraints string. 
Returns a constraints function.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    FIXME: 'math' throws an 'AttributeError' -- try 'parse_simplified'.

    For example:
        >>> constraints = '''x1 = x2 + 2.
        ...     x2 = x3*2.'''
        >>> f = parse(constraints)
        >>> f([1.0, 0.0, 1.0])
        [4.0, 2.0, 1.0]

Additional Inputs:
    suggestedorder -- tuple containing the order in which the variables should
        be solved for. The first 'neqns' variables will be independent, and
        the rest will be dependent.

    NOTE: For example, if suggestedorder=(3, 1, 2) and there are two
        constraints equations, x3 and x1 will be constrained in terms of x2.
        By default, increasing order (i.e. 1, 2, ...) is used. suggestedorder
        must enumerate all variables, hence len(sugestedorder) == nvars.

Further Inputs:
    details -- boolean for whether or not to also return 'variable_info'.
    guess -- list of parameter values proposed to solve the constraints.
    lower_bounds -- list of lower bounds on solution values.
    upper_bounds -- list of upper bounds on solution values.

    NOTE: For optimization problems with initial values and either lower or
        upper bounds, the constraints function must be formulated in a manner
        such that it does not immediately violate the given bounds.
"""
    # FIXME: exec seems to fail on 'math' functions...
    #        workaound might be to catch AttributeError to 'parse_simplified'
    try:
        return parse_linear(constraints, variables=variables, \
                            suggestedorder=suggestedorder, **kwds)
    except:
        return parse_nonlinear(constraints, variables=variables, \
                               suggestedorder=suggestedorder, **kwds)

def parse_nonlinear(constraints, variables='x', suggestedorder=None, **kwds):
    """Build a constraints function given a string of nonlinear constraints.
Returns a constraints function. 

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality, inequality, mean,
        and/or range constraints. Standard python syntax rules should be
        followed (with the math module already imported).

    FIXME: 'math' throws an 'AttributeError' -- try 'parse_simplified'.

    For example:
        >>> constraints = '''x2 = x4*3. + (x1*x3)**x1'''
        >>> f = parse_nonlinear(constraints)
        >>> f([1.0, 1.0, 1.0, 1.0])
        [1.0, 4.0, 1.0, 1.0]

Additional Inputs:
    suggestedorder -- tuple containing the order in which the variables should
        be solved for. The first 'neqns' variables will be independent, and
        the rest will be dependent.

    NOTE: For example, if suggestedorder=(3, 1, 2) and there are two
        constraints equations, x3 and x1 will be constrained in terms of x2.
        By default, increasing order (i.e. 1, 2, ...) is used. suggestedorder
        must enumerate all variables, hence len(sugestedorder) == nvars.

Further Inputs:
    details -- boolean for whether or not to also return 'variable_info'.
    guess -- list of parameter values proposed to solve the constraints.
    lower_bounds -- list of lower bounds on solution values.
    upper_bounds -- list of upper bounds on solution values.

    NOTE: For optimization problems with initial values and either lower or
        upper bounds, the constraints function must be formulated in a manner
        such that it does not immediately violate the given bounds.
"""
    # FIXME: exec seems to fail on 'math' functions...
    #        workaound might be to catch AttributeError to 'parse_simplified'
    nvars = None # number of variables. Should be determined automatically
    strict = False # if True, force to use 'permutations' code
    warn = True  # if True, don't supress warning about old versions of sympy
    verbose = False # if True, return all permutations
    details = False # if True, print details from classify_variables
    guess = None
    upper_bounds = None
    lower_bounds = None
    #-----------------------undocumented-------------------------------
    if kwds.has_key('nvars'): nvars = kwds['nvars']
    if kwds.has_key('strict'): strict = kwds['strict']
    if kwds.has_key('warn'): warn = kwds['warn']
    if kwds.has_key('verbose'): verbose = kwds['verbose']
    #------------------------------------------------------------------
    if kwds.has_key('details'): details = kwds['details']
    if kwds.has_key('guess'): guess = kwds['guess']
    if kwds.has_key('upper_bounds'): upper_bounds = kwds['upper_bounds']
    if kwds.has_key('lower_bounds'): lower_bounds = kwds['lower_bounds']

    if list_or_tuple_or_ndarray(variables):
        constraints = substitute_symbolic(constraints, variables, '_')
        ndim = len(variables)
        varname = '_'
    else:
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(variables)) for v in myvar])
        else: ndim = 0
        varname = variables
    if nvars: ndim = nvars
    elif guess: ndim = len(guess)
    elif lower_bounds: ndim = len(lower_bounds)
    elif upper_bounds: ndim = len(upper_bounds)

    # Figure out if trying various permutations is necessary
    if (guess and lower_bounds) or (guess and upper_bounds):
        strict = True

    eqns = constraints.splitlines()
    # Remove empty strings:
    actual_eqns = []
    for j in range(len(eqns)):
        if eqns[j].strip():
           actual_eqns.append(eqns[j].strip())
    orig_eqns = actual_eqns[:]

    neqns = len(actual_eqns)

    # Getting all permutations will take a really really long time
    # for something like nvars >= 10.
    perms = list(permutations(range(ndim)))
    if suggestedorder: # Try the suggested order first.
        suggestedorder = tuple(asarray(suggestedorder) - 1)
        perms.remove(suggestedorder)
        perms.insert(0, suggestedorder)

    complete_list = []

    constraints_function_list = []
    # Some of the permutations will give the same answer;
    # look into reducing the number of repeats?
    for p in perms:
        thisorder = p
        # Sort the list actual_eqns so any equation containing x1 is first, etc.
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
            # Trying to use xi as a pivot. Loop through the equations
            # looking for one containing xi.
            target = usedvars[i]
            for eqn in actual_eqns[i:]:
                invertedstring = simplify_symbolic(eqn, variables=varname, target=target, warn=warn)
                if invertedstring:
                    warn = False
                    break
            # substitute into the remaining equations. the equations' order
            # in the list newsystem is like in a linear coefficient matrix.
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
            
        # Invert so that it can be fed properly to parse_simplified
        simplified = []
        for eqn in actual_eqns:
            target = usedvars[actual_eqns.index(eqn)]
            simplified.append(simplify_symbolic(eqn, variables=varname, target=target, warn=warn))

        cf = parse_simplified('\n'.join(simplified), variables=varname) 

        if verbose:
            complete_list.append(cf)
            continue

        if strict and guess:
            try: # catches trying to square root a negative number, for example.
                cf(guess)
            except ValueError:
                continue
            compatible = isbounded(cf, guess, lower_bounds=\
                 lower_bounds, upper_bounds=upper_bounds)
            satisfied = issolution(constraints, cf(guess), verbose=False)
            if compatible and satisfied:
                if details:
                    info = classify_variables('\n'.join(simplified), ndim, variables=varname)
                    return [cf, info]
                else:
                    return cf
        else: #not strict or not guess
            if details:
                info = classify_variables('\n'.join(simplified), ndim, variables=varname)
                return [cf, info]
            else:
                return cf

    warning='Warning: guess does not satisfy both constraints and bounds.\n\
Constraints may not be enforced correctly. Constraints function may be faulty.'
    if warn: print warning
    if details:
        info = classify_variables('\n'.join(simplified), ndim, variables=varname)
        return [cf, info]

    if verbose:
        return complete_list
    return cf

#-------------------------------------------------------------------
# Method for successive `helper' optimizations
 
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
    varname -- base variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...

    NOTE: For example, a constraints_string = '''length = height**2 - 3*width'''
        will have a varnamelist = ['length', 'height', 'width'] which specifies
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
    #FIXME: interface needs some cleaning.
    from mystic.math import approx_equal
    disp = False # to match the defaults in the solvers
    iterated_penalty = False
    randomstate = None
    if kwds.has_key('disp'): disp = kwds['disp']
    if kwds.has_key('iterated_penalty'): iterated_penalty = kwds['iterated_penalty']
    if kwds.has_key('randomstate'): randomstate = kwds['randomstate']

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
        constraints_string = substitute_symbolic(constraints_string, varnamelist)
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
    varname -- base variable name. Default is 'x'.
    varnamelist -- list of variable name strings. Use this keyword if variable
        names in the constraints string are not of the form x1, x2, x3, ...
    constraints_method -- string name of constraints method. Valid method
        names are ['direct', 'barrier', 'penalty', 'auglag'].
"""
    #XXX: probably needs redesign -- should be property of a constraints method?

    #XXX: Should the eqcon_funcs/ineqcon_funcs interface have the user input a
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
                            strict_con = substitute_symbolic(strict_con, \
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
with 'penalty' method."
        method = 'penalty'

    # If direct and symbolic string input, pass to parse
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

            constraints = parse(constraints, ndim, \
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

            constraints_func = parse(new_constraints, \
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
            
    if varnamelist: vars = varnamelist  #XXX: hack to merge varnamelist
    else: vars = varname                #     and varname to variables

    if method == 'barrier':
        # Check if x0 satisfies constraints.
        ineqcon_funcs = []
        eqcon_funcs = []
        if kwds.has_key('ineqcon_funcs'): ineqcon_funcs = kwds['ineqcon_funcs']
        if kwds.has_key('eqcon_funcs'): eqcon_funcs = kwds['eqcon_funcs']
        if not issolution(constraints, x0, variables=vars, verbose=False, \
                          feq=eqcon_funcs, fineq=ineqcon_funcs):
            # If constraints are not satisfied, do an optimization to find
            # a feasible point. If one is found, continue optimizing with that.
            # Should print a message to say a preliminary optimization is being
            # performed?
            feasible_point = solve(constraints_beforestrict, guess=x0, \
                                   feq=eqcon_funcs, fineq=ineqcon_funcs,\
                                   variables=vars, \
                                   lower_bounds=solver_instance._strictMin, \
                                   upper_bounds=solver_instance._strictMax)
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
        costfunc = wrap_constraints(constraints, costfunc, \
                                    strict=constraints_strict, \
                                    variables=vars, **kwds)
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


if __name__ == '__main__':
    pass

#EOF
