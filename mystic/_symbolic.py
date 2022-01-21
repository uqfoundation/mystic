#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

# The following code attempts to construct something like:
# >>> from sympy import Eq, Symbol
# >>> from sympy import solve as symsol
# >>> x0 = Symbol('x0')
# >>> x1 = Symbol('x1')
# >>> x2 = Symbol('x2')
# >>> eq1 = Eq(x1, x0 - 2.)
# >>> eq2 = Eq(x1, x2*2.)
# >>> soln = symsol([eq2, eq1], [x0, x1, x2])

from mystic.tools import permutations
from mystic.tools import list_or_tuple_or_ndarray
import sys
if (sys.hexversion >= 0x30000f0):
    exec_locals_ = 'exec(code, _locals)'
    exec_globals_ = 'exec(_code, globals(), _locals)'
else:
    exec_locals_ = 'exec code in _locals'
    exec_globals_ = 'exec _code in globals(), _locals'
NL = '\n'
#FIXME: remove this head-standing to workaround python2.6 exec bug

def _classify_variables(constraints, variables='x', nvars=None): 
    """Takes a string of constraint equations and determines which variables
are dependent, independent, and unconstrained. Assumes there are no duplicate
equations. Returns a dictionary with keys: 'dependent', 'independent', and
'unconstrained', and with values that enumerate the variables that match
each variable type.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints must be equality constraints only.
        Standard python syntax should be followed (with the math and numpy
        modules already imported).

    For example:
        >>> constraints = '''
        ...     x0 = x4**2
        ...     x2 = x3 + x4'''
        >>> _classify_variables(constraints, nvars=5)
        {'dependent':['x0','x2'], 'independent':['x3','x4'], 'unconstrained':['x1']}
        >>> constraints = '''
        ...     x0 = x4**2
        ...     x4 - x3 = 0.
        ...     x4 - x0 = x2'''
        >>> _classify_variables(constraints, nvars=5)
        {'dependent': ['x0','x2','x4'], 'independent': ['x3'], 'unconstrained': ['x1']}

Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x1' in the example above).
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
"""
    if ">" in constraints or "<" in constraints:
        raise NotImplementedError("cannot classify inequalities") 

    from mystic.symbolic import replace_variables, get_variables
    #XXX: use solve? or first if not in form xi = ... ?
    if list_or_tuple_or_ndarray(variables):
        if nvars is not None: variables = variables[:nvars]
        constraints = replace_variables(constraints, variables)
        varname = '$'
        ndim = len(variables)
    else:
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar]) + 1
        else: ndim = 0
    if nvars is not None: ndim = nvars

    eqns = constraints.splitlines()
    indices = list(range(ndim))
    dep = []
    indep = []
    for eqn in eqns: # find which variables are used
        if eqn:
            for var in range(ndim):
                if indices.count(var) != 0:
                    if eqn.find(varname + str(var)) != -1:
                        indep.append(var)
                        indices.remove(var)
    indep.sort()
    _dep = []
    for eqn in eqns: # find which variables are on the LHS
        if eqn:
            split = eqn.split('=')
            for var in indep:
                if split[0].find(varname + str(var)) != -1:
                    _dep.append(var)
                    indep.remove(var)
                    break
    _dep.sort()
    indep = _dep + indep # prefer variables found on LHS
    for eqn in eqns: # find one dependent variable per equation
        _dep = []
        _indep = indep[:]
        if eqn:
            for var in _indep:
                if eqn.find(varname + str(var)) != -1:
                    _dep.append(var)
                    _indep.remove(var)
        if _dep:
            dep.append(_dep[0])
            indep.remove(_dep[0])
    #FIXME: 'equivalent' equations not ignored (e.g. x2=x2; or x2=1, 2*x2=2)
    """These are good:
    >>> constraints = '''
    ...     x0 = x4**2
    ...     x2 - x4 - x3 = 0.'''
    >>> _classify_variables(constraints, nvars=5)
    {'dependent': ['x0','x2'], 'independent': ['x3','x4'], 'unconstrained': ['x1']}
    >>> constraints = '''
    ...     x0 + x2 = 0.
    ...     x0 + 2*x2 = 0.'''
    >>> _classify_variables(constraints, nvars=5)
    {'dependent': ['x0','x2'], 'independent': [], 'unconstrained': ['x1','x3','x4']}

    This is a bug:
    >>> constraints = '''
    ...     x0 + x2 = 0.
    ...     2*x0 + 2*x2 = 0.'''
    >>> _classify_variables(constraints, nvars=5)
    {'dependent': ['x0','x2'], 'independent': [], 'unconstrained': ['x1','x3','x4']}
    """ #XXX: should simplify first?
    dep.sort()
    indep.sort()
    # return the actual variable names (not the indices)
    if varname == variables: # then was single variable
        variables = [varname+str(i) for i in range(ndim)]
    dep = [variables[i] for i in dep]
    indep = [variables[i] for i in indep]
    indices = [variables[i] for i in indices]
    d = {'dependent':dep, 'independent':indep, 'unconstrained':indices}
    return d


def _prepare_sympy(constraints, variables='x', nvars=None):
    """Parse an equation string and prepare input for sympy. Returns a tuple
of sympy-specific input: (code for variable declaration, left side of equation
string, right side of equation string, list of variables, and the number of
sympy equations).

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints must be equality constraints only.
        Standard python syntax should be followed (with the math and numpy
        modules already imported).

    For example:
        >>> constraints = '''
        ...     x0 = x4**2
        ...     x4 - x3 = 0.
        ...     x4 - x0 = x2'''
        >>> code, lhs, rhs, vars, neqn = _prepare_sympy(constraints, nvars=5)
        >>> print(code)
        x0=Symbol('x0')
        x1=Symbol('x1')
        x2=Symbol('x2')
        x3=Symbol('x3')
        x4=Symbol('x4')
        rand = Symbol('rand')
        >>> print("%s %s" % (lhs, rhs))
        ['x0 ', 'x4 - x3 ', 'x4 - x0 '] [' x4**2', ' 0.', ' x2']
        >>> print("%s in %s eqns" % (vars, neqn))
        x0,x1,x2,x3,x4, in 3 eqns

Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x1' in the example above).
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
"""
    if ">" in constraints or "<" in constraints:
        raise NotImplementedError("cannot simplify inequalities") 

    from mystic.symbolic import replace_variables, get_variables
    #XXX: if constraints contain x0,x1,x3 for 'x', should x2 be in code,xlist?
    if list_or_tuple_or_ndarray(variables):
        if nvars is not None: variables = variables[:nvars]
        constraints = replace_variables(constraints, variables, markers='_')
        varname = '_'
        ndim = len(variables)
    else:
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar]) + 1
        else: ndim = 0
    if nvars is not None: ndim = nvars

    # split constraints_str into lists of left hand sides and right hand sides
    eacheqn = constraints.splitlines()
    neqns = 0
    left = []
    right = []
    for eq in eacheqn: #XXX: Le/Ge instead of Eq; Max/Min... (NotImplemented ?)
        splitlist = eq.replace('==','=').split('=') #FIXME: no inequalities
        if len(splitlist) == 2:   #FIXME: first convert >/< to min/max ?

            # If equation is blank on one side, raise error.
            if len(splitlist[0].strip()) == 0 or len(splitlist[1].strip()) == 0:
                print("%r is not an equation!" % eq) # Raise exception?
            else:
                left.append(splitlist[0])
                right.append(splitlist[1])
                neqns += 1

        # If equation doesn't have one equal sign, raise error.
        if len(splitlist) != 2 and len(splitlist) != 1:
            print("%r is not an equation!" % eq) # Raise exception?

    # First create list of x variables
    xlist = ""
    for i in range(ndim):
        xn = varname + str(i)
        xlist += xn + ","

    # Start constructing the code string
    code = ""
    for i in range(ndim):
        xn = varname + str(i)
        code += xn + '=' + "Symbol('" + xn + "')\n"

    code += "rand = Symbol('rand')\n"
    return code, left, right, xlist, neqns


def_solve_single = '''
def _solve_single(constraint, variables='x', target=None, **kwds):
    """Solve a symbolic constraints equation for a single variable.

Inputs:
    constraint -- a string of symbolic constraints. Only a single constraint
        equation should be provided, and must be an equality constraint. 
        Standard python syntax should be followed (with the math and numpy
        modules already imported).

    For example:
        >>> equation = "x1 - 3. = x0*x2"
        >>> print(_solve_single(equation))
        x0 = -(3.0 - x1)/x2

Additional Inputs:
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    target -- list providing the order for which the variables will be solved.
        By default, increasing order is used.

    For example:
        >>> equation = "x1 - 3. = x0*x2"
        >>> print(_solve_single(equation, target='x1'))
        x1 = 3.0 + x0*x2

Further Inputs:
    locals -- a dictionary of additional variables used in the symbolic
        constraints equations, and their desired values.
    simplest -- if True, simplify all but polynomials order >= 3 [False]
    rational -- if True, recast floats as rationals during solve [False]
    sequence -- if True, solve sequentially and not as a matrix [False]
    implicit -- if True, solve implicitly (with sin, cos, ...) [False]
    check -- if False, skip minimal testing (divide_by_zero, ...) [True]
    permute -- if True, return all permutations [False]
    warn -- if True, don't suppress warnings [True]
    verbose -- if True, print debug information [False]
""" #XXX: an very similar version of this code is found in _solve_linear XXX#
    # for now, we abort on multi-line equations or inequalities
    if len(constraint.replace('==','=').split('=')) != 2:
        raise NotImplementedError("requires a single valid equation") 
    if ">" in constraint or "<" in constraint:
        raise NotImplementedError("cannot simplify inequalities") 

    nvars = None
    permute = False # if True, return all permutations
    warn = True  # if True, don't suppress warnings
    verbose = False  # if True, print debug info
    simplest = False # if True, simplify all but polynomials order >= 3
    rational = False # if True, recast floats as rationals during solve
    sequence = False # if True, solve sequentially and not as a matrix
    implicit = False # if True, solve implicitly (contains sin, ...)
    check = True # if False, skip minimal testing (divide_by_zero, ...)
    kwarg = {} # keywords for sympy's symbolic solve

    permute = kwds['permute'] if 'permute' in kwds else permute
    warn = kwds['warn'] if 'warn' in kwds else warn
    verbose = kwds['verbose'] if 'verbose' in kwds else verbose
    kwarg['simplify'] = kwds['simplest'] if 'simplest' in kwds else simplest
    kwarg['rational'] = kwds['rational'] if 'rational' in kwds else rational
    kwarg['manual'] = kwds['sequence'] if 'sequence' in kwds else sequence
    kwarg['implicit'] = kwds['implicit'] if 'implicit' in kwds else implicit
    kwarg['check'] = kwds['check'] if 'check' in kwds else check

    if target in [None, False]:
        target = []
    elif isinstance(target, str):
        target = target.split(',')
    else:
        target = list(target) # not the best for ndarray, but should work

    from mystic.symbolic import replace_variables, get_variables
    if list_or_tuple_or_ndarray(variables):
        if nvars is not None: variables = variables[:nvars]
        constraints = replace_variables(constraint, variables, markers='_')
        varname = '_'
        ndim = len(variables)
        for i in range(len(target)):
            if variables.count(target[i]):
                target[i] = replace_variables(target[i],variables,markers='_')
    else:
        constraints = constraint # constraints used below
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraint, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar]) + 1
        else: ndim = 0
    if nvars is not None: ndim = nvars

    # create function to replace "_" with original variables
    def restore(variables, mystring):
        if list_or_tuple_or_ndarray(variables):
            vars = get_variables(mystring,'_')
            indices = [int(v.strip('_')) for v in vars]
            for i in range(len(vars)):
                mystring = mystring.replace(vars[i],variables[indices[i]])
        return mystring

    # default is _locals with sympy imported
    _locals = {}
    locals = kwds['locals'] if 'locals' in kwds else None
    if locals is None: locals = {}
    try:
        import imp
        imp.find_module('sympy')
        code = """from sympy import Eq, Symbol;"""
        code += """from sympy import solve as symsol;"""
        code = compile(code, '<string>', 'exec')
        %(exec_locals_)s
    except ImportError: # Equation will not be simplified."
        if warn: print("Warning: sympy not installed.")
        return constraint

    # default is _locals with numpy and math imported
    # numpy throws an 'AttributeError', but math passes error to sympy
    code = """from numpy import *; from math import *;""" # prefer math
    code += """from numpy import mean as average;""" # use np.mean not average
    code += """from numpy import var as variance;""" # look like mystic.math
    code += """from numpy import ptp as spread;"""   # look like mystic.math
    code = compile(code, '<string>', 'exec')
    %(exec_locals_)s
    _locals.update(dict(symsol_kwds=kwarg))
    _locals.update(locals) #XXX: allow this?

    code,left,right,xlist,neqns = _prepare_sympy(constraints, varname, ndim)

    eqlist = ""
    for i in range(1, neqns+1):
        eqn = 'eq' + str(i)
        eqlist += eqn + ","
        code += eqn + '= Eq(' + left[i-1] + ',' + right[i-1].replace(' 0.00', ' .00').replace(' 0.01', ' .01').replace(' 0.02', ' .02').replace(' 0.03', ' .03').replace(' 0.04', ' .04').replace(' 0.05', ' .05').replace(' 0.06', '.06').replace(' 0.07', ' .07').replace(' 0.08', ' .08').replace(' 0.09', ' .09').replace(' 0.0',' 0') + ')'+NL # sympy bug for 0.0
    eqlist = eqlist.rstrip(',')

    # get full list of variables in 'targeted' order
    xperms = xlist.split(',')[:-1]
    targeted = target[:]
    [targeted.remove(i) for i in targeted if i not in xperms]
    [targeted.append(i) for i in xperms if i not in targeted]
    _target = []
    [_target.append(i) for i in targeted if i not in _target]
    targeted = _target
    targeted = tuple(targeted)
    com = ',' if targeted else ''

    ########################################################################
    # solve each xi: symsol(single_equation, [x0,x1,...,xi,...,xn])
    # returns: {x0: f(xn,...), x1: f(xn,...), ..., xn: f(...,x0)}
    if permute or not target: #XXX: the goal is solving *only one* equation
        code += '_xlist = ({0}{1})'.format(','.join(targeted), com) + NL
        code += '_elist = [symsol(['+eqlist+'], [i], **symsol_kwds) for i in _xlist]' + NL
        code += '_elist = [i if isinstance(i, dict) else {j:i[-1][-1]} for j,i in zip(_xlist,_elist) if i]' + NL
        code += 'soln = dict()' + NL
        code += '[soln.update(i) for i in _elist if i]' + NL
    else:
        code += 'soln = symsol(['+eqlist+'], ['+target[0]+'], **symsol_kwds)' + NL
       #code += 'soln = symsol(['+eqlist+'], ['+targeted[0]+'], **symsol_kwds)' + NL
        code += 'soln = soln if isinstance(soln, dict) else {'+target[0]+': soln[-1][-1]} if soln else ""' + NL
    #code += 'print(soln)' + NL
    ########################################################################

    if verbose: print(code)
    _code = compile(code, '<string>', 'exec')
    try: 
        %(exec_globals_)s
        soln = _locals['soln'] if 'soln' in _locals else None
        if not soln:
            if warn: print("Warning: target variable is not valid")
            soln = {}
    except NotImplementedError: # catch 'multivariate' error for older sympy
        if warn: print("Warning: could not simplify equation.")
        return constraint      #FIXME: resolve diff with _solve_linear
    except NameError as error: # catch when variable is not defined
        if warn: print("Warning: {0}".format(error))
        soln = {}
    if verbose: print(soln)

    #XXX handles multiple solutions?
    soln = getattr(soln, 'iteritems', soln.items)()
    soln = dict([(str(key),str(value)) for key, value in soln])
    soln = [(i,soln[i]) for i in targeted if i in soln] #XXX: order as targeted?
    
    solns = []; solved = ""
    for key,value in soln:
        solved = str(key) + ' = ' + str(value) + NL
        if solved: solns.append( restore(variables, solved.rstrip()) )

    if not permute:
        sol = None if not solns[:1] else solns[0]
        return '' if sol and not sol.strip() else sol
    return tuple(solns)
''' % dict(exec_locals_=exec_locals_, exec_globals_=exec_globals_)
exec(def_solve_single)
del def_solve_single

doc_solve_linear = """Solve a system of symbolic linear constraints equations.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints must be equality constraints only.
        Standard python syntax should be followed (with the math and numpy
        modules already imported).

    For example:
        >>> constraints = '''
        ...     x0 - x2 = 2.
        ...     x2 = x3*2.'''
        >>> print(_solve_linear(constraints))
        x2 = 2.0*x3
        x0 = 2.0 + 2.0*x3

Additional Inputs:
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    target -- list providing the order for which the variables will be solved.
        If there are "N" constraint equations, the first "N" variables given
        will be selected as the dependent variables. By default, increasing
        order is used.

    For example:
        >>> constraints = '''
        ...     x0 - x2 = 2.
        ...     x2 = x3*2.'''
        >>> print(_solve_linear(constraints, target=['x3','x2']))
        x3 = -1.0 + 0.5*x0
        x2 = -2.0 + x0

Further Inputs:
    locals -- a dictionary of additional variables used in the symbolic
        constraints equations, and their desired values.
    simplest -- if True, simplify all but polynomials order >= 3 [False]
    rational -- if True, recast floats as rationals during solve [False]
    sequence -- if True, solve sequentially and not as a matrix [False]
    implicit -- if True, solve implicitly (with sin, cos, ...) [False]
    check -- if False, skip minimal testing (divide_by_zero, ...) [True]
    permute -- if True, return all permutations [False]
    warn -- if True, don't suppress warnings [True]
    verbose -- if True, print debug information [False]
"""
def_solve_linear = '''
def _solve_linear(constraints, variables='x', target=None, **kwds):
    nvars = None
    permute = False # if True, return all permutations
    warn = True  # if True, don't suppress warnings
    verbose = False  # if True, print debug info
    simplest = False # if True, simplify all but polynomials order >= 3
    rational = False # if True, recast floats as rationals during solve
    sequence = False # if True, solve sequentially and not as a matrix
    implicit = False # if True, solve implicitly (contains sin, ...)
    check = True # if False, skip minimal testing (divide_by_zero, ...)
    kwarg = {} # keywords for sympy's symbolic solve

    permute = kwds['permute'] if 'permute' in kwds else permute
    warn = kwds['warn'] if 'warn' in kwds else warn
    verbose = kwds['verbose'] if 'verbose' in kwds else verbose
    kwarg['simplify'] = kwds['simplest'] if 'simplest' in kwds else simplest
    kwarg['rational'] = kwds['rational'] if 'rational' in kwds else rational
    kwarg['manual'] = kwds['sequence'] if 'sequence' in kwds else sequence
    kwarg['implicit'] = kwds['implicit'] if 'implicit' in kwds else implicit
    kwarg['check'] = kwds['check'] if 'check' in kwds else check

    if target in [None, False]:
        target = []
    elif isinstance(target, str):
        target = target.split(',')
    else:
        target = list(target) # not the best for ndarray, but should work

    from mystic.symbolic import replace_variables, get_variables
    if list_or_tuple_or_ndarray(variables):
        if nvars is not None: variables = variables[:nvars]
        _constraints = replace_variables(constraints, variables, '_')
        varname = '_'
        ndim = len(variables)
        for i in range(len(target)):
            if variables.count(target[i]):
                target[i] = replace_variables(target[i],variables,markers='_')
    else:
        _constraints = constraints
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar]) + 1
        else: ndim = 0
    if nvars is not None: ndim = nvars

    # create function to replace "_" with original variables
    def restore(variables, mystring):
        if list_or_tuple_or_ndarray(variables):
            vars = get_variables(mystring,'_')
            indices = [int(v.strip('_')) for v in vars]
            for i in range(len(vars)):
                mystring = mystring.replace(vars[i],variables[indices[i]])
        return mystring

    # default is _locals with sympy imported
    _locals = {}
    locals = kwds['locals'] if 'locals' in kwds else None
    if locals is None: locals = {}
    # if sympy not installed, return original constraints
    try:
        import imp
        imp.find_module('sympy')
        code = """from sympy import Eq, Symbol;"""
        code += """from sympy import solve as symsol;"""
        code = compile(code, '<string>', 'exec')
        %(exec_locals_)s
    except ImportError: # Equation will not be simplified."
        if warn: print("Warning: sympy not installed.")
        return constraints

    # default is _locals with numpy and math imported
    # numpy throws an 'AttributeError', but math passes error to sympy
    code = """from numpy import *; from math import *;""" # prefer math
    code += """from numpy import mean as average;""" # use np.mean not average
    code += """from numpy import var as variance;""" # look like mystic.math
    code += """from numpy import ptp as spread;"""   # look like mystic.math
    code = compile(code, '<string>', 'exec')
    %(exec_locals_)s
    _locals.update(dict(symsol_kwds=kwarg))
    _locals.update(locals) #XXX: allow this?

    code,left,right,xlist,neqns = _prepare_sympy(_constraints, varname, ndim)

    eqlist = ""
    for i in range(1, neqns+1):
        eqn = 'eq' + str(i)
        eqlist += eqn + ","
        code += eqn + '= Eq(' + left[i-1] + ',' + right[i-1].replace(' 0.00', ' .00').replace(' 0.01', ' .01').replace(' 0.02', ' .02').replace(' 0.03', ' .03').replace(' 0.04', ' .04').replace(' 0.05', ' .05').replace(' 0.06', '.06').replace(' 0.07', ' .07').replace(' 0.08', ' .08').replace(' 0.09', ' .09').replace(' 0.0',' 0') + ')' + NL # sympy bug for 0.0
    eqlist = eqlist.rstrip(',')

    # get full list of variables in 'targeted' order
    xperms = xlist.split(',')[:-1]
    targeted = target[:]
    [targeted.remove(i) for i in targeted if i not in xperms]
    [targeted.append(i) for i in xperms if i not in targeted]
    _target = []
    [_target.append(i) for i in targeted if i not in _target]
    targeted = _target
    targeted = tuple(targeted)

    if permute:
        # Form constraints equations for each permutation.
        # This will change the order of the x variables passed to symsol()
        # to get different variables solved for.
        xperms = list(permutations(xperms)) #XXX: takes a while if nvars is ~10
        if target: # put the tuple with the 'targeted' order first
            xperms.remove(targeted)
            xperms.insert(0, targeted)
    else:
        xperms = [tuple(targeted)]

    solns = []
    for perm in xperms: 
        _code = code
        xlist = ','.join(perm).rstrip(',') #XXX: if not all, use target ?
        # solve dependent xi: symsol([linear_system], [x0,x1,...,xi,...,xn])
        # returns: {x0: f(xn,...), x1: f(xn,...), ...}
        _code += 'soln = symsol(['+eqlist+'], ['+xlist+'], **symsol_kwds)'
        #XXX: need to convert/check soln similarly as in _solve_single ?
        if verbose: print(_code)
        _code = compile(_code, '<string>', 'exec')
        try: 
            %(exec_globals_)s
            soln = _locals['soln'] if 'soln' in _locals else None
            if not soln:
                if warn: print("Warning: could not simplify equation.")
                soln = {}
        except NotImplementedError: # catch 'multivariate' error
            if warn: print("Warning: could not simplify equation.")
            soln = {}
        except NameError as error: # catch when variable is not defined
            if warn: print("Warning: {0}".format(error))
            soln = {}
        if verbose: print(soln)

        solved = ""
        for key, value in getattr(soln, 'iteritems', soln.items)():
            solved += str(key) + ' = ' + str(value) + NL
        if solved: solns.append( restore(variables, solved.rstrip()) )

    if not permute:
        return None if not solns[:1] else solns[0]

    # Remove duplicates
    filter = []; results = []
    for i in solns:
        _eqs = NL.join(sorted(i.split(NL)))
        if _eqs not in filter:
            filter.append(_eqs)
            results.append(i)
    return tuple(results)

_solve_linear.__doc__ = doc_solve_linear
''' % dict(exec_locals_=exec_locals_, exec_globals_=exec_globals_)
exec(def_solve_linear)
del def_solve_linear, doc_solve_linear, exec_locals_, exec_globals_

    # Create strings of all permutations of the solved equations.
    # Remove duplicates, then take permutations of the lines of equations
    # to create equations in different orders.
#   noduplicates = []
#   [noduplicates.append(i) for i in solns if i not in noduplicates]
#   stringperms = []
#   for item in noduplicates:
#       spl = item.splitlines()
#       for perm in permutations(spl):
#           permstring = ""
#           for line in perm:
#               permstring += line + '\n'
#           stringperms.append(permstring.rstrip())
#   return tuple(stringperms)


def solve(constraints, variables='x', target=None, **kwds):
    """Solve a system of symbolic constraints equations.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints must be equality constraints only.
        Standard python syntax should be followed (with the math and numpy
        modules already imported).

    For example:
        >>> constraints = '''
        ...     x0 - x2 = 2.
        ...     x2 = x3*2.'''
        >>> print(solve(constraints))
        x2 = 2.0*x3
        x0 = 2.0 + 2.0*x3
        >>> constraints = '''
        ...     spread([x0,x1]) - 1.0 = mean([x0,x1])   
        ...     mean([x0,x1,x2]) = x2'''
        >>> print(solve(constraints))
        x0 = -0.5 + 0.5*x2
        x1 = 0.5 + 1.5*x2

Additional Inputs:
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    target -- list providing the order for which the variables will be solved.
        If there are "N" constraint equations, the first "N" variables given
        will be selected as the dependent variables. By default, increasing
        order is used.

    For example:
        >>> constraints = '''
        ...     x0 - x2 = 2.
        ...     x2 = x3*2.'''
        >>> print(solve(constraints, target=['x3','x2']))
        x3 = -1.0 + 0.5*x0
        x2 = -2.0 + x0

Further Inputs:
    locals -- a dictionary of additional variables used in the symbolic
        constraints equations, and their desired values.
    simplest -- if True, simplify all but polynomials order >= 3 [False]
    rational -- if True, recast floats as rationals during solve [False]
    sequence -- if True, solve sequentially and not as a matrix [False]
    implicit -- if True, solve implicitly (with sin, cos, ...) [False]
    check -- if False, skip minimal testing (divide_by_zero, ...) [True]
    permute -- if True, return all permutations [False]
    warn -- if True, don't suppress warnings [False]
    verbose -- if True, print debug information [False]
"""
    kwds['warn'] = kwds.get('warn', False)
    try:
        if len(constraints.replace('==','=').split('=')) <= 2:
            soln = _solve_single(constraints, variables=variables, \
                                 target=target, **kwds)
            # for corner case where has something like: '0*xN'
            if not soln or not soln.strip():
                if target in [None, False]: target = []
                elif isinstance(target, str): target = target.split(',')
                else: target = list(target)
                soln = _solve_single(constraints, variables=variables, \
                                     target=target[1:], **kwds)
        else:
            soln = _solve_linear(constraints, variables=variables, \
                                 target=target, **kwds)
        if not soln: raise ValueError
    except:
        soln = _solve_nonlinear(constraints, variables=variables, \
                                target=target, **kwds)
    return soln


def _solve_nonlinear(constraints, variables='x', target=None, **kwds):
    """Build a constraints function given a string of nonlinear constraints.
Returns a constraints function. 

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints must be equality constraints only.
        Standard python syntax should be followed (with the math and numpy
        modules already imported).

    For example:
        >>> constraints = '''x1 = x3*3. + x0*x2'''
        >>> print(_solve_nonlinear(constraints))
        x0 = (x1 - 3.0*x3)/x2
        >>> constraints = '''
        ...     spread([x0,x1]) - 1.0 = mean([x0,x1])   
        ...     mean([x0,x1,x2]) = x2'''
        >>> print(_solve_nonlinear(constraints))
        x0 = -0.5 + 0.5*x2
        x1 = 0.5 + 1.5*x2

Additional Inputs:
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    target -- list providing the order for which the variables will be solved.
        If there are "N" constraint equations, the first "N" variables given
        will be selected as the dependent variables. By default, increasing
        order is used.

    For example:
        >>> constraints = '''
        ...     spread([x0,x1]) - 1.0 = mean([x0,x1])   
        ...     mean([x0,x1,x2]) = x2'''
        >>> print(_solve_nonlinear(constraints, target=['x1']))
        x1 = -0.833333333333333 + 0.166666666666667*x2
        x0 = -0.5 + 0.5*x2

Further Inputs:
    locals -- a dictionary of additional variables used in the symbolic
        constraints equations, and their desired values.
    simplest -- if True, simplify all but polynomials order >= 3 [False]
    rational -- if True, recast floats as rationals during solve [False]
    sequence -- if True, solve sequentially and not as a matrix [False]
    implicit -- if True, solve implicitly (with sin, cos, ...) [False]
    check -- if False, skip minimal testing (divide_by_zero, ...) [True]
    permute -- if True, return all permutations [False]
    warn -- if True, don't suppress warnings [True]
    verbose -- if True, print debug information [False]
"""
    nvars = None
    permute = False # if True, return all permutations
    warn = True  # if True, don't suppress warnings
    verbose = False # if True, print details from _classify_variables
    simplest = False # if True, simplify all but polynomials order >= 3
    rational = False # if True, recast floats as rationals during solve
    sequence = False # if True, solve sequentially and not as a matrix
    implicit = False # if True, solve implicitly (contains sin, ...)
    check = True # if False, skip minimal testing (divide_by_zero, ...)
    kwarg = {} # keywords for sympy's symbolic solve

    permute = kwds['permute'] if 'permute' in kwds else permute
    warn = kwds['warn'] if 'warn' in kwds else warn
    verbose = kwds['verbose'] if 'verbose' in kwds else verbose
    kwarg['simplest'] = kwds['simplest'] if 'simplest' in kwds else simplest
    kwarg['rational'] = kwds['rational'] if 'rational' in kwds else rational
    kwarg['sequence'] = kwds['sequence'] if 'sequence' in kwds else sequence
    kwarg['implicit'] = kwds['implicit'] if 'implicit' in kwds else implicit
    kwarg['check'] = kwds['check'] if 'check' in kwds else check

    if target in [None, False]:
        target = []
    elif isinstance(target, str):
        target = target.split(',')
    else:
        target = list(target) # not the best for ndarray, but should work

    from mystic.symbolic import replace_variables, get_variables
    if list_or_tuple_or_ndarray(variables):
        if nvars is not None: variables = variables[:nvars]
        constraints = replace_variables(constraints, variables, '_')
        varname = '_'
        ndim = len(variables)
    else:
        varname = variables # varname used below instead of variables
        myvar = get_variables(constraints, variables)
        if myvar: ndim = max([int(v.strip(varname)) for v in myvar]) + 1
        else: ndim = 0
    if nvars is not None: ndim = nvars

    # create function to replace "_" with original variables
    def restore(variables, mystring):
        if list_or_tuple_or_ndarray(variables):
            vars = get_variables(mystring,'_')
            indices = [int(v.strip('_')) for v in vars]
            for i in range(len(vars)):
                mystring = mystring.replace(vars[i],variables[indices[i]])
        return mystring

    locals = kwds['locals'] if 'locals' in kwds else None
    if locals is None: locals = {}

    eqns = constraints.splitlines()
    # Remove empty strings:
    actual_eqns = []
    for j in range(len(eqns)):
        if eqns[j].strip():
           actual_eqns.append(eqns[j].strip())
    orig_eqns = actual_eqns[:]

    neqns = len(actual_eqns)

    xperms = [varname+str(i) for i in range(ndim)]
    if target:
        [target.remove(i) for i in target if i not in xperms]
        [target.append(i) for i in xperms if i not in target]
        _target = []
        [_target.append(i) for i in target if i not in _target]
        target = _target
    target = tuple(target)

    xperms = list(permutations(xperms)) #XXX: takes a while if nvars is ~10
    if target: # Try the suggested order first.
        xperms.remove(target)
        xperms.insert(0, target)

    complete_list = []

    constraints_function_list = []
    # Some of the permutations will give the same answer;
    # look into reducing the number of repeats?
    for perm in xperms:
        # Sort the list actual_eqns so any equation containing x0 is first, etc.
        sorted_eqns = []
        actual_eqns_copy = orig_eqns[:]
        usedvars = []
        for variable in perm: # range(ndim):
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
        for m in range(nmissing):
            usedvars.append(varname + str(len(tempusedvar) + m))

        #FIXME: not sure if the code below should be totally trusted...
        for i in range(neqns):
            # Trying to use xi as a pivot. Loop through the equations
            # looking for one containing xi.
            _target = usedvars[i%len(usedvars)] #XXX: ...to make it len of neqns
            for eqn in actual_eqns[i:]:
                invertedstring = _solve_single(eqn, variables=varname, target=_target, warn=warn, locals=locals, **kwarg)
                if invertedstring:
                    warn = False
                    break
            if invertedstring is None: continue #XXX: ...when _solve_single fails
            # substitute into the remaining equations. the equations' order
            # in the list newsystem is like in a linear coefficient matrix.
            newsystem = ['']*neqns
            j = actual_eqns.index(eqn)
            newsystem[j] = invertedstring #XXX: ...was eqn. I think correct now
            othereqns = actual_eqns[:j] + actual_eqns[j+1:]
            for othereqn in othereqns:
                expression = invertedstring.split("=")[1]
                fixed = othereqn.replace(_target, '(' + expression + ')')
                k = actual_eqns.index(othereqn)
                newsystem[k] = fixed
            actual_eqns = newsystem #XXX: potentially carrying too many eqns

        # Invert so that it can be fed properly to generate_constraint
        simplified = []
        for eqn in actual_eqns[:len(usedvars)]: #XXX: ...needs to be same len
            _target = usedvars[actual_eqns.index(eqn)]
            mysoln = _solve_single(eqn, variables=varname, target=_target, warn=warn, locals=locals, **kwarg)
            if mysoln: simplified.append(mysoln)
        simplified = restore(variables, '\n'.join(simplified).rstrip())

        if permute:
            complete_list.append(simplified)
            continue

        if verbose:
            print(_classify_variables(simplified, variables, ndim))
        return simplified

    warning='Warning: an error occurred in building the constraints.'
    if warn: print(warning)
    if verbose:
        print(_classify_variables(simplified, variables, ndim))
    if permute: #FIXME: target='x3,x1' may order correct, while 'x1,x3' doesn't
        filter = []; results = []
        for i in complete_list:
            _eqs = '\n'.join(sorted(i.split('\n')))
            if _eqs and (_eqs not in filter):
                filter.append(_eqs)
                results.append(i)
        return tuple(results) #FIXME: somehow 'rhs = xi' can be in results
    return simplified


# EOF
