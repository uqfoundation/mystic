#!/usr/bin/env python
#
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# originally coded by Alta Fang, 2010
# refactored by Mike McKerns, 2012
"""Tools for working with symbolic constraints.
"""
__all__ = ['linear_symbolic','replace_variables','get_variables','denominator',
           'simplify','comparator','flip','_flip','condense','flat','equals',
           'penalty_parser','constraints_parser','generate_conditions','solve',
           'generate_solvers','generate_penalty','generate_constraint','merge',
           '_simplify','absval','symbolic_bounds']

from numpy import ndarray, asarray, any as _any
from mystic._symbolic import solve
from mystic.tools import list_or_tuple_or_ndarray, flatten
import sys
if (sys.hexversion >= 0x30000f0):
    exec_locals_ = 'exec(code, _locals)'
    exec_locals = 'exec(code, locals)'
    exec_globals = 'exec(code, globals)'
    exec_results = 'exec(code, globals, results)'
else:
    exec_locals_ = 'exec code in _locals'
    exec_locals = 'exec code in locals'
    exec_globals = 'exec code in globals'
    exec_results = 'exec code in globals, results'
NL = '\n'
#FIXME: remove this head-standing to workaround python2.6 exec bug


# XXX: another function for the inverse... symbolic to matrix? (good for scipy)
def linear_symbolic(A=None, b=None, G=None, h=None, variables=None):
    """convert linear equality and inequality constraints from matrices to a 
symbolic string of the form required by mystic's constraint parser.

Inputs:
    A -- (ndarray) matrix of coefficients of linear equality constraints
    b -- (ndarray) vector of solutions of linear equality constraints
    G -- (ndarray) matrix of coefficients of linear inequality constraints
    h -- (ndarray) vector of solutions of linear inequality constraints
    variables -- (list[str]) list of variable names

    NOTE: if variables=None, then variables = ['x0', 'x1', ...];
          if variables='y', then variables = ['y0', 'y1', ...];
          otherwise use the explicit list of variables provided.

    NOTE: Must provide A and b; G and h; or A, b, G, and h;
          where Ax = b and Gx <= h. 

    For example:
    >>> A = [[3., 4., 5.],
    ...      [1., 6., -9.]]
    >>> b = [0., 0.]
    >>> G = [1., 0., 0.]
    >>> h = [5.]
    >>> print(linear_symbolic(A,b,G,h))
    1.0*x0 + 0.0*x1 + 0.0*x2 <= 5.0
    3.0*x0 + 4.0*x1 + 5.0*x2 = 0.0
    1.0*x0 + 6.0*x1 + -9.0*x2 = 0.0
"""
    if variables is None: variables = 'x'
    try:
        basestring
    except NameError:
        basestring = str
    has_base = isinstance(variables, basestring)

    eqstring = ""
    # Equality constraints
    if A is not None and b is not None:
        # If one-dimensional and not in a nested list, add a list layer
        try:
            ndim = len(A[0])
        except:
            ndim = len(A)
            A = [A]

        # Flatten b, in case it's in the form [[0, 1, 2]] for example.
        if len(b) == 1:
            b = ndarray.flatten(asarray(b)).tolist()

        # Check dimensions and give errors if incorrect.
        if len(A) != len(b): #NOTE: changed from Exception 11/15/21
            raise ValueError("dimensions of A and b are not consistent")

        if has_base:
            names = [variables+str(j) for j in range(ndim)]
        else:
            if len(variables) != ndim:
                raise ValueError("variables is not consistent with A")
            names = variables

        # 'matrix multiply' and form the string
        for i in range(len(b)):
            Asum = ""
            for j in range(ndim):
                Asum += str(A[i][j]) + '*' + names[j] + ' + '
            eqstring += Asum.rstrip(' + ') + ' = ' + str(b[i]) + '\n'

    # Inequality constraints
    ineqstring = ""
    if G is not None and h is not None:
        # If one-dimensional and not in a nested list, add a list layer
        try:
            ndim = len(G[0])
        except:
            ndim = len(G)
            G = [G]

        # Flatten h, in case it's in the form [[0, 1, 2]] for example.
        if len(h) == 1:
            h = ndarray.flatten(asarray(h)).tolist()

        # Check dimensions and give errors if incorrect.
        if len(G) != len(h): #NOTE: changed from Exception 11/15/21
            raise ValueError("dimensions of G and h are not consistent")

        if has_base:
            names = [variables+str(j) for j in range(ndim)]
        else:
            if len(variables) != ndim:
                raise ValueError("variables is not consistent with G")
            names = variables

        # 'matrix multiply' and form the string
        for i in range(len(h)):
            Gsum = ""
            for j in range(ndim):
                Gsum += str(G[i][j]) + '*' + names[j] + ' + '
            ineqstring += Gsum.rstrip(' + ') + ' <= ' + str(h[i]) + '\n'
    totalconstraints = ineqstring + eqstring
    return totalconstraints 


def symbolic_bounds(min, max, variables=None):
    """convert min,max to symbolic string for use in mystic's constraint parser

Inputs:
    min -- (list[float]) list of lower bounds
    max -- (list[float]) list of upper bounds
    variables -- (list[str]) list of variable names

    NOTE: if variables=None, then variables = ['x0', 'x1', ...];
          if variables='y', then variables = ['y0', 'y1', ...];
          otherwise use the explicit list of variables provided.

    For example:
    >>> eqn = symbolic_bounds(min=[-10,None], max=[10,69], variables='y')
    >>> print(eqn)
    y0 >= -10.0
    y0 <= 10.0
    y1 <= 69.0

    >>> eqn = symbolic_bounds(min=[-1,-2], max=[1,2], variables=list('AB'))
    >>> print(eqn)
    A >= -1.0
    B >= -2.0
    A <= 1.0
    B <= 2.0
"""
    inf = float('inf')
    if len(min) != len(max):
        raise ValueError("length of min and max are not consistent")
    # when 'some' of the bounds are given as 'None', replace with default
    for i in range(len(min)):
        if min[i] is None: min[i] = -inf
        if max[i] is None: max[i] = inf
    min = asarray(min); max = asarray(max)
    if _any(( min > max ),0):
        raise ValueError("each min[i] must be <= the corresponding max[i]")

    if variables is None: variables = 'x'
    try:
        basestring
    except NameError:
        basestring = str
    has_base = isinstance(variables, basestring)
    lo = '%s >= %s'
    hi = '%s <= %s'
    if has_base:
        lo = variables + lo
        hi = variables + hi
        imin = enumerate(min)
        imax = enumerate(max)
    else:
        if len(min) != len(variables):
            raise ValueError("variables is not consistent with bounds")
        imin = zip(variables, min)
        imax = zip(variables, max)
    #NOTE: we are stripping off leading zeros
    lo = '\n'.join(lo % (i,str(float(j)).lstrip('0')) for (i,j) in imin if j != -inf)
    hi = '\n'.join(hi % (i,str(float(j)).lstrip('0')) for (i,j) in imax if j != inf)
    return '\n'.join([lo, hi]).strip()


def comparator(equation):
    "identify the comparator (e.g. '<', '=', ...) in a constraints equation"
    if '\n' in equation.strip(): #XXX: failure throws error or returns ''?
        return [comparator(eqn) for eqn in equation.strip().split('\n') if eqn]
    return '<=' if equation.count('<=') else '<' if equation.count('<') else \
           '>=' if equation.count('>=') else '>' if equation.count('>') else \
           '!=' if equation.count('!=') else \
           '==' if equation.count('==') else '=' if equation.count('=') else ''


def _flip(cmp, bounds=False): # to invert sign if dividing by negative value
    "flip the comparator (i.e. '<' to '>', or  '<' to '>=' if bounds=True)"
    if bounds:
        return '<' if cmp == '>=' else '<=' if cmp == '>' else \
               '>' if cmp == '<=' else '>=' if cmp == '<' else cmp
    return '<=' if cmp == '>=' else '<' if cmp == '>' else \
           '>=' if cmp == '<=' else '>' if cmp == '<' else cmp


def flip(equation, bounds=False):
    """flip the inequality in the equation (i.e. '<' to '>'), if one exists

Inputs:
    equation -- an equation string; can be an equality or inequality
    bounds -- if True, ensure set boundaries are respected (i.e. '<' to '>=')
"""
    cmp = comparator(equation)
    return _flip(cmp, bounds).join(equation.split(cmp)) if cmp else equation


#FIXME: if 'cycle=True', do all perumtations (and select shortest)?
#FIXME: should be better, currenlty only condenses exact matches
def condense(*equations, **kwds):
    """condense tuples of equations to the simplest representation

Inputs:
    equations -- tuples of inequalities or equalities

    For example:
    >>> condense(('C <= 0', 'B <= 0'), ('C <= 0', 'B >= 0'))
    [('C <= 0',)]
    >>> condense(('C <= 0', 'B <= 0'), ('C >= 0', 'B <= 0'))
    [('B <= 0',)]
    >>> condense(('C <= 0', 'B <= 0'), ('C >= 0', 'B >= 0'))
    [('C <= 0', 'B <= 0'), ('C >= 0', 'B >= 0')]

Additional Inputs:
    verbose -- if True, print diagnostic information. Default is False.
"""
    verbose = kwds['verbose'] if 'verbose' in kwds else False
    result, miss = [],[]
    skip = set()
    found = False
    for i,u in enumerate(equations):
        if i in skip: continue
        for j,v in enumerate(equations[i+1:],i+1):
            if verbose: print("try: {0} {1}".format(u,v))
            left = []
            same = tuple(k for k in u if k in v or left.append(flip(k)))
            if len(same) == len(u) - 1 and all(k in v for k in left):
                if same: result.append(same)
                skip.add(i); skip.add(j)
                found = True
                break
        if not found: miss.append(u)
        else: found = False
    if verbose:
        print("matches: {0}".format(result))
        print("misses: {0}".format(miss))
    return condense(*result, **kwds) + miss if result else miss


def merge(*equations, **kwds):
    """merge bounds in a sequence of equations (e.g. ``[A<0, A>0] --> [A!=0]``)

Args:
    equations (tuple(str)): a sequence of equations
    inclusive (bool, default=True): if False, bounds are exclusive

Returns:
    tuple sequence of equations, where the bounds have been merged

Notes:
    if bounds are invalid, returns ``None``

Examples:
    >>> merge(*['A > 0', 'A > 0', 'B >= 0', 'B <= 0'], inclusive=False)
    ('A > 0', 'B = 0')

    >>> merge(*['A > 0', 'A > 0', 'B >= 0', 'B <= 0'], inclusive=True)
    ('A > 0',)
"""
    inclusive = kwds['inclusive'] if 'inclusive' in kwds else True
    if inclusive:
        '''
        if ('X > 0', 'X < 0') then 'X != 0'
        if ('X > 0', 'X <= 0') then ''
        if ('X >= 0', 'X < 0') then ''
        if ('X >= 0', 'X <= 0') then ''
        '''
        # sub >< with !=
        equations = tuple(i.replace(comparator(i),'!=') if (comparator(i) in ('>','<')) and (flip(i) in equations) else i for i in equations)
        # sub other inequality pairs with ''; delete duplicate entries
        equations = set('' if ('>' in i or '<' in i) and (flip(i) in equations or flip(i,True) in equations) else i for i in equations)
        # delete '' entries
        return tuple(i for i in equations if i != '')
    # else exclusive
    '''
    if ('X > 0', 'X < 0') then None
    if ('X > 0', 'X <= 0') then None
    if ('X >= 0', 'X < 0') then None
    if ('X >= 0', 'X <= 0') then 'X = 0'
    '''
    # sub >< with =
    equations = tuple(i.replace(comparator(i),'=') if ('>=' in i or '<=' in i) and (flip(i) in equations) else i for i in equations)
    # sub other inequality pairs with None; delete duplicate entries
    equations = set(None if ('>' in i or '<' in i) and (flip(i) in equations or flip(i,True) in equations) else i for i in equations)
    # if None in entries then return None (i.e. not valid)
    return None if None in equations else tuple(equations)


def _enclosed(equation):
    import re
    "split equation at the close of all enclosing parentheses"
    res = []
    # split equation at the close of the first enclosing parenthesis
    ir = iter(re.findall(r'[^\)]*\)', equation)) #FIXME: misses **2
    n,r = 0,''
    for i in ir:
        r += i; n += i.count('(')-1
        if n <= 0:
            break
    if r:
        res.append(r)
    # repeat if necessary
    i = ''.join(ir)
    if i:
        res.extend(_enclosed(i))
    return res


def _noparen(variable):
    "remove enclosing parenthesis for a single-variable expression"
    if variable.startswith('(') and variable.endswith(')'):
        return variable[1:-1]
    return variable


def flat(equation, subs=None):
    """flatten equation by replacing expressions in parenthesis with a marker

Inputs:
    equation -- a symbolic equation string, with no more than one line, and
        following standard python syntax.
    subs -- a dict of {marker: sub-string} of replacements made, where marker
        will be of the form '$0$', '$1$', etc.
    """
    eqn = _enclosed(equation)
    for i,e in enumerate(eqn):
        marker = '$%s$' % i
        # find the enclosed expression
        _e = e[e.find('(')+1:-1]
        if not subs is None:
            subs[marker] = _e
        # make the substitution in the original
        equation = equation.replace(e, e.replace('(%s)' % _e, '(%s)' % marker))
    return equation


def _denominator(equation):
    """find valid denominators for placeholder-simplified equations"""
    # placeholder-simplified: parenthesis-enclosed text is replaced by markers
    import re
    return [''.join(i).strip('/') for i in re.findall(r'(/\s*[\(\$\w\)]+)(\*\*[\(\$\w\)]+)?', equation)]


def denominator(equation, variables=None):
    """find denominators containing the given variables in an equation

Inputs:
    equation -- a symbolic equation string, with no more than one line.
        Equation can be an equality or inequality, and must follow standard
        python syntax (with the math and numpy modules already imported).
    variables -- a variable base string (e.g. 'x' = 'x0','x1',...), or
        a list of variable name strings (e.g. ['x','y','z0']). Default is 'x'.
    """
    if variables is None: variables = 'x'
    # deal with the lhs and rhs separately
    cmp = comparator(equation)
    if cmp:
        res,equation = equation.split(cmp,1)
        res = set(denominator(res, variables))
    else: res = set()
    # flatten and find all statements enclosed in parenthesis
    subs = {}
    equation = flat(equation, subs)
    # find denominators in flattened equation
    denom = set(_denominator(equation))
    res.update(_noparen(e) for e in denom if get_variables(e, variables))
    # find denominators within the enclosed parentheses
    for k,v in subs.items():
        # replace tmp-names in denominators as necessary
        den = (e.replace(k,v) for e in denom if k in e)
        # exclude replaced k's that don't have variables
        res.update(d for d in den if get_variables(d, variables))
        # recurse into new statements
        res.update(denominator(v, variables))
    return list(res)


#XXX: add target=None to kwds?
def _solve_zeros(equation, variables=None, implicit=True):
    '''symbolic solve the equation for when produces a ZeroDivisionError'''
    # if implicit = True, can solve to functions of a variable (i.e. sin(A)=1)
    res = denominator(equation, variables)#XXX: w/o this, is a general solve
    x = variables or 'x'
    _res = []
    for i,eqn in enumerate(res):
        _eqn = eqn+' = 0'
        try:
            eqn_ = solve(_eqn, target=variables, variables=x)
            if not eqn_:
                msg = "cannot simplify '%s'" % _eqn
                raise ValueError(msg)
            vars = set(get_variables(eqn,x)).difference(get_variables(eqn_,x))
            while vars: # solve for missing variables
                alt = solve(_eqn, target=vars.pop(), variables=x)
                if alt: _res.append(alt) 
            _eqn = eqn_
        except ValueError:
            if not implicit:
                msg = "cannot simplify '%s'" % _eqn
                raise ValueError(msg)
            #else: pass
        res[i] = _eqn
    return res + _res


#XXX: if error=False, should return None? or ???
def equals(before, after, vals=None, **kwds):
    """check if equations before and after are equal at the given vals

Inputs:
    before -- an equation string
    after -- an equation string
    vals -- a dict with variable names as keys and floats as values

Additional Inputs:
    variables -- a list of variable names
    locals -- a dict with variable names as keys and 'fixed' values
    error -- if False, ZeroDivisionError evaluates as None
    variants -- a list of ints to use as variants for fractional powers
""" #verbose -- print debug messages
    errors = kwds['error'] if 'error' in kwds else True#XXX: default of False?
    variants = kwds['variants'] if 'variants' in kwds else None
    #verbose = kwds['verbose'] if 'verbose' in kwds else False
    vars = kwds['variables'] if 'variables' in kwds else 'x'
    _vars = get_variables(after, vars)
    locals = kwds['locals'] if 'locals' in kwds else None
    if locals is None: locals = {}
    if vals is None: vals = {}
    locals.update(vals)
    #if verbose: print(locals)
    locals_ = locals.copy() #XXX: HACK _locals
    while variants:
        try:
            after, before = eval(after,{},locals_), eval(before,{},locals_)
            break
        except (ValueError,TypeError) as error:  #FIXME: python2.5
            if (error.args[0].startswith('negative number') and \
               error.args[0].endswith('raised to a fractional power')) or \
               (error.args[0].find('not supported') and \
               error.args[0].rfind("'complex'")):
                val = variants.pop()
                [locals_.update({k:v+val}) for k,v in getattr(locals_, 'iteritems', locals_.items)() if k in _vars]
            else:
                raise error
        except ZeroDivisionError as error:
            if errors: raise error
            try:
                eval(after,{},locals_)
            except ZeroDivisionError:
                try:
                    eval(before,{},locals_)
                    return False
                except ZeroDivisionError:
                    return True
            return False
    else: #END HACK
        try:
            after, before = eval(after,{},locals_), eval(before,{},locals_)
        except ZeroDivisionError as error:
            if errors: raise error
            try:
                eval(after,{},locals_)
            except ZeroDivisionError:
                try:
                    eval(before,{},locals_)
                    return False
                except ZeroDivisionError:
                    return True
            return False
    return before == after


def _absval(equation, **kwds):
    verbose = kwds['verbose'] if 'verbose' in kwds else False
    # find each top-level 'abs(' and the contents in the '()'
    #FIXME: should search for 'Xabs(' where X cannot be used in a name
    q = {}
    abs_ = 'abs('
    fe0 = flat(equation, q)
    z = []
    for e in _enclosed(fe0):
        if abs_ in e:
            z.append(e[e.find(abs_)+4:-1])

    # prepare q.values as lists
    p = {} # 'conditions'
    for k,v in q.items():
        if abs_ in v:
            q[k],p[k] = _absval(v, **kwds)
        else:
            q[k] = [v]
            p[k] = ['']

    if verbose:
        print('org: {0}'.format(equation))
        print('sub: {0}'.format(fe0))
        print('eqn: {0}'.format(q))
        print('con: {0}'.format(p))

    # build equations out of innards of abs() #NOTE: order is same as 'z'
    qz0 = [[(c.replace(NL,' >= 0\n',1) if NL in c else c+' >= 0') for c in q[zi]] for zi in z]

    # generate all cases by replacing 'abs' with combinations of '' and '-'
    s = ['','-']
    re0 = []
    ze0 = []

    import itertools as it
    for si in it.product(s, repeat=fe0.count(abs_)):
        ze = [] #NOTE: order is same as 'z'
        re = fe0
        for i,j in enumerate(si):
            ze.append([NL.join(flip(ii) for ii in jj.split(NL)) for jj in qz0[i]] if j else qz0[i]) #XXX: flip(ii) or flip(ii,True)?
            re = re.replace('abs',j,1)
        ze0.append([(NL+NL.join(i)).rstrip() for i in it.product(*ze,repeat=1)])
        re0.append(re)

    # replace stubs with inner (sort to ensure same order as ze0)
    rz0 = []
    sorter = lambda i:(int(k.strip('$')),v)
    pk,pv = tuple(zip(*sorted(p.items(), key=sorter )))
    qk,qv = tuple(zip(*sorted(q.items(), key=sorter )))

    for re,ze in zip(re0,ze0):
        qkv = []
        for i,(qvi,pvi) in enumerate(zip(it.product(*qv),it.product(*pv))):
            r_ = re
            for ki,vi in zip(qk, qvi):
                r_ = r_.replace(ki,vi) # replace stubs
            qkv.append(r_)
            ze[i] += ''.join(pvi) # include inner conditions
        rz0.append(qkv)

    if verbose:
        print('eqns: {0}'.format(re0))
        print('cond: {0}'.format(ze0))
    ze0 = list(it.chain.from_iterable(ze0))
    re0 = list(it.chain.from_iterable(rz0))
    return re0,ze0

def absval(constraints, **kwds):
    """rewrite a system of symbolic constraints without using absolute value.

Returns a system of equations where 'abs' has been replaced with the
equivalent conditional algebraic expressions.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Standard python syntax should be followed (with
        the math and numpy modules already imported).

Additional Inputs:
    all -- boolean to return all simplifications due to absolute values.
        If all is True, return all possible simplifications due to absolute
        value being used in one or more of the equations. The default is
        False, returning only one possible simplification.
    verbose -- if True, print debug information [False]
"""
    import random
    import itertools as it
    all = kwds['all'] if 'all' in kwds else False
    verbose = kwds['verbose'] if 'verbose' in kwds else False
    abs_ = 'abs('
    eqns = [([''.join((i,j)) for i,j in zip(*_absval(e.strip(), **kwds))] if abs_ in e else [e.strip()]) for e in constraints.strip().split(NL)]
    # combine each eqn, and simplify the conditionals
    eqns = tuple(NL.join(merge(*(NL.join(i).split(NL)), inclusive=True)) for i in it.product(*eqns)) #FIXME: inclusive=True, or False ???
    return (eqns if all else eqns[random.randint(0,len(eqns)-1)]) if len(eqns) > 1 else (eqns[0] if len(eqns) else '') #FIXME: len(eqns) = 0 --> Error, '', ???


doc_simplify = """simplify a system of symbolic constraints equations.

Returns a system of equations where a single variable has been isolated on
the left-hand side of each constraints equation, thus all constraints are
of the form "x_i = f(x)".

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Standard python syntax should be followed (with
        the math and numpy modules already imported).

    For example:
        >>> constraints = '''
        ...     x0 - x2 <= 2.
        ...     x2 = x3*2.'''
        >>> print(simplify(constraints))
        x0 <= x2 + 2.0
        x2 = 2.0*x3
        >>> constraints = '''
        ...     x0 - x1 - 1.0 = mean([x0,x1])   
        ...     mean([x0,x1,x2]) >= x2'''
        >>> print(simplify(constraints))
        x0 = 3.0*x1 + 2.0
        x0 >= -x1 + 2*x2

Additional Inputs:
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    target -- list providing the order for which the variables will be solved.
        If there are "N" constraint equations, the first "N" variables given
        will be selected as the dependent variables. By default, increasing
        order is used.

Further Inputs:
    locals -- a dictionary of additional variables used in the symbolic
        constraints equations, and their desired values.
    cycle -- boolean to cycle the order for which the variables are solved.
        If cycle is True, there should be more variety on the left-hand side
        of the simplified equations. By default, the variables do not cycle.
    all -- boolean to return all simplifications due to negative values.
        When dividing by a possibly negative variable, an inequality may flip,
        thus creating alternate simplifications. If all is True, return all
        possible simplifications due to negative values in an inequalty.
        The default is False, returning only one possible simplification.
    simplest -- if True, simplify all but polynomials order >= 3 [False]
    rational -- if True, recast floats as rationals during solve [False]
    sequence -- if True, solve sequentially and not as a matrix [False]
    implicit -- if True, solve implicitly (with sin, cos, ...) [False]
    check -- if False, skip minimal testing (divide_by_zero, ...) [True]
    permute -- if True, return all permutations [False]
    warn -- if True, don't suppress warnings [False]
    verbose -- if True, print debug information [False]
"""
#FIXME: should minimize number of times LHS is reused; (or use 'and_')?
#FIXME: should not fail at ZeroDivisionError (what should it do there?)
#FIXME: should order better (e.g. C > 0; B == C - 5; A > B + 2)
def_simplify = '''
def _simplify(constraints, variables='x', target=None, **kwds):
    ### undocumented ###
   #rand -- random number generator [default: random.random]
   #error -- if False, ZeroDivisionError evaluates as None [default: True]
    ####################
    all = kwds['all'] if 'all' in kwds else False
    import random
    import itertools as it
    locals = kwds['locals'] if 'locals' in kwds else {} #XXX: HACK _locals
    _locals = {}
    # default is _locals with numpy and math imported
    # numpy throws an 'AttributeError', but math passes error to sympy
    code = """from numpy import *; from math import *;""" # prefer math
    code += """from numpy import mean as average;""" # use np.mean not average
    code += """from numpy import var as variance;""" # look like mystic.math
    code += """from numpy import ptp as spread;"""   # look like mystic.math
    code += """_sqrt = lambda x:x**.5;""" # 'domain error' to 'negative power'
    code = compile(code, '<string>', 'exec')
    %(exec_locals_)s
    _locals.update(locals)
    kwds['locals'] = _locals
    del locals

    def _simplify1(eqn, rand=random.random, target=None, **kwds):
        'isolate one variable on the lhs'
        verbose = kwds['verbose'] if 'verbose' in kwds else False
        vars = kwds['variables'] if 'variables' in kwds else 'x'
        cmp = comparator(eqn)
        # get all variables used
        allvars = get_variables(eqn, vars)
        # find where the sign flips might occur (from before)
        res = eqn.replace(cmp,'=')
        zro = _solve_zeros(res, allvars)
        # check which variables have been used
        lhs = lambda zro: tuple(z.split('=')[0].strip() for z in zro)
        used = lhs(zro) #XXX: better as iterator?
        # cycle used variables to the rear
        _allvars = []
        _allvars = [i for i in allvars if i not in used or (_allvars.append(i) if i not in _allvars else False)] + _allvars
        # simplify so lhs has only one variable (and replace sympy's Abs)
        res = solve(res, target=target, **kwds)
        _eqn = res.replace('=',cmp).replace('Abs(','abs(')
        # find where the sign flips might occur (from after)
        zro += _solve_zeros(res, get_variables(res.split('=')[-1],_allvars))
        _zro = [z.replace('=','!=') for z in zro]
        if verbose:
            print('in: {0}'.format(eqn))
            print('out: {0}'.format(_eqn))
            print('zero: {0}'.format(_zro))
        # if no inequalities, then return
        if not cmp.count('<')+cmp.count('>'):
            return NL.join([_eqn]+_zro) if _zro else _eqn
        del _zro

        # make sure '=' is '==' so works in eval
        before,after = (eqn,_eqn) if cmp != '=' else (eqn.replace(cmp,'=='),_eqn.replace(cmp,'=='))
        #HACK: avoid (rand-M)**(1/N) w/ (rand-M) negative; sqrt(x) to x**.5
        before = before.replace('sqrt(','_sqrt(')
        after = after.replace('sqrt(','_sqrt(')

        # sort zeros so equations with least variables are first
        zro.sort(key=lambda z: len(get_variables(z, vars))) #XXX: best order?
        # build dicts of test variables, with +/- epsilon at solved zeros
        testvars = dict((i,2*rand()-1) for i in allvars)
        eps = str(.01 * rand()) #XXX: better epsilon?
        #FIXME: following not sufficient w/multiple 'zs' (A != 0, A != -B)
        testvals = it.product(*((z+'+'+eps,z+'-'+eps) for z in zro))
        # build tuple of corresponding comparators for testvals
        signs = it.product(*(('>','<') for z in zro))

        def _testvals(testcode):
            'generate dict of test values as directed by the testcode'
            locals = _locals.copy()
            locals.update(testvars)
            code = ';'.join(i for i in testcode)
            code = compile(code, '<string>', 'exec')
            try:
                %(exec_locals)s
            except SyntaxError as error:
                msg = "cannot simplify '{0}'".format(testcode)
                raise SyntaxError(msg,)
            return dict((i,locals[i]) for i in allvars)

        # iterator of dicts of test values
        testvals = getattr(it, 'imap', map)(_testvals, testvals)

        # evaluate expression to see if comparator needs to be flipped
        results = []
        variants = (100000,-200000,100100,-200,110,-20,11,-2,1) #HACK
        kwds['variants'] = list(variants)
        for sign in signs:
            if equals(before,after,next(testvals),**kwds):
                new = [after]
            else:
                new = [after.replace(cmp,flip(cmp))] #XXX: or flip(cmp,True)?
            new.extend(z.replace('=',i) for (z,i) in getattr(it, 'izip', zip)(zro,sign))
            results.append(new)

        # reduce the results to the simplest representation
       #results = condense(*results, **kwds) #XXX: remove depends on testvals
        # convert results to a tuple of multiline strings
        results = tuple(NL.join(i).replace('_sqrt(','sqrt(') for i in results)
        if len(results) == 1: results = results[0]
        return results

    #### ...the rest is _simplify()... ###
    cycle = kwds['cycle'] if 'cycle' in kwds else False
    verbose = kwds['verbose'] if 'verbose' in kwds else False
    eqns = []
    used = []
    for eqn in constraints.strip().split(NL):
        # get least used, as they are likely to be simpler
        vars = get_variables(eqn, variables)
        vars.sort(key=eqn.count) #XXX: better to sort by count(var+'**')?
        vars = target[:] if target else vars
        if cycle: vars = [var for var in vars if var not in used] + used
        while vars:
            try: # cycle through variables trying 'simplest' first
                res = _simplify1(eqn, variables=variables, target=vars, **kwds)
                if verbose: print('#: {0}'.format(res))
                res = res if type(res) is tuple else (res,)
                eqns.append(res)
                r = res[0] #XXX: only add the 'primary' variable to used
                used.append(r.split(comparator(r.split(NL)[0]),1)[0].strip())
                #print("v,u: {0} {1}".format(vars, used))
                break
            except ValueError:
                try:
                    basestring
                except NameError:
                    basestring = str
                if isinstance(vars, basestring): vars = []
                else: vars.pop(0)
                if verbose: print('PASS')
                #print("v,u: {0} {1}".format(vars, used))
        else: # failure... so re-raise error
            res = _simplify1(eqn, variables=variables, target=target, **kwds)
            if verbose: print('X: {0}'.format(res))
            res = res if type(res) is tuple else (res,)
            eqns.append(res)
    _eqns = it.product(*eqns)
    eqns = tuple(NL.join(i) for i in _eqns)
    # "merge" the multiple equations to find simplest bounds
    eqns = tuple(merge(*e.split(NL), inclusive=False) for e in eqns)
    if eqns.count(None) == len(eqns): return None
    #   msg = 'No solution'
    #   raise ValueError(msg) #XXX: return None? throw Error? or ???
    eqns = tuple(NL.join(e) for e in eqns if e != None)
    #XXX: if all=False, is possible to return "most True" (smallest penalty)?
    return (eqns if all else eqns[random.randint(0,len(eqns)-1)]) if len(eqns) > 1 else (eqns[0] if len(eqns) else '')

_simplify.__doc__ = doc_simplify
''' % dict(exec_locals_=exec_locals_, exec_locals=exec_locals)
exec(def_simplify)
del def_simplify, doc_simplify, exec_locals_, exec_locals


def simplify(constraints, variables='x', target=None, **kwds):
    import random
    import itertools as it
    all = kwds['all'] if 'all' in kwds else False
    cons = absval(constraints, **kwds) #NOTE: only uses all,verbose
    kwds['variables'] = variables
    kwds['target'] = target
    #import klepto as kl
    """
    @kl.inf_cache(keymap=kl.keymaps.stringmap(flat=False), ignore=('**','kwds'))
    def simple(eqn, **kwds):
        return _simplify(eqn, **kwds)
    """
    simple = _simplify
    cons = [simple(ci, **kwds) for ci in cons] if type(cons) is tuple else simple(cons, **kwds)
    #simple.__cache__().clear() #NOTE: clear stored entries
    eqns = tuple(it.chain.from_iterable(i if type(i) is tuple else (i,) for i in cons)) if type(cons) is list else (cons if type(cons) is tuple else (cons,))
    return (eqns if all else eqns[random.randint(0,len(eqns)-1)]) if len(eqns) > 1 else (eqns[0] if len(eqns) else '') #FIXME: len(eqns) = 0 --> Error, '', ???
simplify.__doc__ = _simplify.__doc__


def replace_variables(constraints, variables=None, markers='$'):
    """replace variables in constraints string with a marker.
Returns a modified constraints string.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality and/or inequality
        constraints. Standard python syntax should be followed (with the
        math and numpy modules already imported).
    variables -- list of variable name strings. The variable names will
        be replaced in the order that they are provided, where if the
        default marker "$i" is used, the first variable will be replaced
        with "$0", the second with "$1", and so on.

    For example:
        >>> variables = ['spam', 'eggs']
        >>> constraints = '''spam + eggs - 42'''
        >>> print(replace_variables(constraints, variables, 'x'))
        'x0 + x1 - 42'

Additional Inputs:
    markers -- desired variable name. Default is '$'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base.

    For example:
        >>> variables = ['x1','x2','x3']
        >>> constraints = "min(x1*x2) - sin(x3)"
        >>> print(replace_variables(constraints, variables, ['x','y','z']))
        'min(x*y) - sin(z)'
"""
    if variables is None: variables = []
    elif isinstance(variables, str): variables = list((variables,))

    # substitite one list of strings for another
    if list_or_tuple_or_ndarray(markers):
        equations = replace_variables(constraints,variables,'_')
        vars = get_variables(equations,'_')
        indices = [int(v.strip('_')) for v in vars]
        for i in range(len(vars)):
            equations = equations.replace(vars[i],markers[indices[i]])
        return equations

    # Sort by decreasing length of variable name, so that if one variable name 
    # is a substring of another, that won't be a problem. 
    variablescopy = variables[:]
    variablescopy.sort(key=lambda x: -len(x))

    # Figure out which index goes with which variable.
    indices = []
    for item in variablescopy:
        indices.append(variables.index(item))

    # Default is markers='$', as '$' is not a special symbol in Python,
    # and it is unlikely a user will choose it for a variable name.
    if markers in variables:
        marker = '_$$$$$$$$$$' # even less likely...
    else:
        marker = markers

    '''Bug demonstrated here:
    >>> equation = """x3 = max(y,x) + x"""
    >>> vars = ['x','y','z','x3']
    >>> print(replace_variables(equation,vars))
    $4 = ma$1($2,$1) + $1
    ''' #FIXME: don't parse if __name__ in __builtins__, globals, or locals?
    for i in indices: #FIXME: or better, use 're' pattern matching
        constraints = constraints.replace(variables[i], marker + str(i))
    return constraints.replace(marker, markers)


def get_variables(constraints, variables='x'):
    """extract a list of the string variable names from constraints string

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality and/or inequality
        constraints. Standard python syntax should be followed (with the
        math and numpy modules already imported).

    For example:
        >>> constraints = '''
        ...     x1 + x2 = x3*4
        ...     x3 = x2*x4'''
        >>> get_variables(constraints)
        ['x1', 'x2', 'x3', 'x4'] 

Additional Inputs:
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.

    For example:
        >>> constraints = '''              
        ...     y = min(u,v) - z*sin(x)
        ...     z = x**2 + 1.0 
        ...     u = v*z'''
        >>> get_variables(constraints, list('pqrstuvwxyz'))
        ['u', 'v', 'x', 'y', 'z']
"""
    if list_or_tuple_or_ndarray(variables):
        equations = replace_variables(constraints,variables,'_')
        vars = get_variables(equations,'_')
        indices = [int(v.strip('_')) for v in vars]
        varnamelist = []
        from numpy import sort
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


def penalty_parser(constraints, variables='x', nvars=None):
    """parse symbolic constraints into penalty constraints.
Returns a tuple of inequality constraints and a tuple of equality constraints.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality and/or inequality
        constraints. Standard python syntax should be followed (with the
        math and numpy modules already imported).

    For example:
        >>> constraints = '''
        ...     x2 = x0/2.
        ...     x0 >= 0.'''
        >>> penalty_parser(constraints, nvars=3)
        (('-(x[0] - (0.))',), ('x[2] - (x[0]/2.)',))

Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x2' in the example above).
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    """
   #from mystic.tools import src
   #ndim = len(get_variables(src(func), variables))
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

    # Parse the constraints string
    lines = constraints.splitlines()
    eqconstraints = []
    ineqconstraints = []
    for line in lines:
        if line.strip():
            fixed = line
            # Iterate in reverse in case ndim > 9.
            indices = list(range(ndim))
            indices.reverse()
            for i in indices:
                fixed = fixed.replace(varname + str(i), 'x[' + str(i) + ']') 
            constraint = fixed.strip()

            # Replace 'spread', 'mean', and 'variance' (uses numpy, not mystic)
            if constraint.find('spread(') != -1:
                constraint = constraint.replace('spread(', 'ptp(')
            if constraint.find('mean(') != -1:
                constraint = constraint.replace('mean(', 'average(')
            if constraint.find('variance(') != -1:
                constraint = constraint.replace('variance(', 'var(')

            # Sorting into equality and inequality constraints, and making all
            # inequality constraints in the form expression <= 0. and all 
            # equality constraints of the form expression = 0.
            split = constraint.split('>')
            direction = '>'
            if len(split) == 1:
                split = constraint.split('<')
                direction = '<'
            if len(split) == 1:
                split = constraint.split('!=')
                direction = '!='
            if len(split) == 1:
                split = constraint.split('=')
                direction = '='
            if len(split) == 1:
                print("Invalid constraint: %s" % constraint)
            # Use epsilon whenever '<' or '>' is comparator
            eps = comparator(constraint)
            eps = ' + e_ ' if eps == '>' else (' - e_ ' if eps == '<' else '')
            eqn = {'lhs':split[0].rstrip('=').strip(), \
                   'rhs':split[-1].lstrip('=').strip()}
            eqn['rhs'] += eps.replace('e_', '_tol(%s,tol,rel)' % eqn['rhs'])
            expression = '%(lhs)s - (%(rhs)s)' % eqn
            if direction == '=':
                eqconstraints.append(expression)
            elif direction == '<':
                ineqconstraints.append(expression)
            elif direction == '>':
                ineqconstraints.append('-('+ expression +')')
            else: #XXX: better value than 1 for when '==' is True?
                eqconstraints.append('('+ expression +') == 0')

    return tuple(ineqconstraints), tuple(eqconstraints)


def constraints_parser(constraints, variables='x', nvars=None):
    """parse symbolic constraints into a tuple of constraints solver equations.
The left-hand side of each constraint must be simplified to support assignment.

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality and/or inequality
        constraints. Standard python syntax should be followed (with the
        math and numpy modules already imported).

    For example:
        >>> constraints = '''
        ...     x2 = x0/2.
        ...     x0 >= 0.'''
        >>> constraints_parser(constraints, nvars=3)
        ('x[2] = x[0]/2.', 'x[0] = max(0., x[0])')

Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x2' in the example above).
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    """
   #from mystic.tools import src
   #ndim = len(get_variables(src(func), variables))
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

    def _process_line(line):
        'string processing from line to expression'
        # Iterate in reverse in case ndim > 9.
        indices = list(range(ndim))
        indices.reverse()
        for i in indices:
            line = line.replace(varname + str(i), 'x[' + str(i) + ']') 
        constraint = line.strip()

        # Replace 'ptp', 'average', and 'var' (uses mystic, not numpy)
        if constraint.find('ptp(') != -1:
            constraint = constraint.replace('ptp(', 'spread(')
        if constraint.find('average(') != -1:
            constraint = constraint.replace('average(', 'mean(')
        if constraint.find('var(') != -1:
            constraint = constraint.replace('var(', 'variance(')
        if constraint.find('prod(') != -1:
            constraint = constraint.replace('prod(', 'product(')
        return constraint

    def _process_expression(expression):
        ' allow mystic.math.measures impose_* on LHS '
        lhs,rhs = expression.split('=')
        if lhs.find('spread(') != -1:
          lhs = lhs.split('spread')[-1]
          rhs = ' impose_spread( (' + rhs.lstrip() + '),' + lhs + ')'
        if lhs.find('mean(') != -1:
          lhs = lhs.split('mean')[-1]
          rhs = ' impose_mean( (' + rhs.lstrip() + '),' + lhs + ')'
        if lhs.find('variance(') != -1:
          lhs = lhs.split('variance')[-1]
          rhs = ' impose_variance( (' + rhs.lstrip() + '),' + lhs + ')'
        if lhs.find('sum(') != -1:
          lhs = lhs.split('sum')[-1]
          rhs = ' impose_sum( (' + rhs.lstrip() + '),' + lhs + ')'
        if lhs.find('product(') != -1:
          lhs = lhs.split('product')[-1]
          rhs = ' impose_product( (' + rhs.lstrip() + '),' + lhs + ')'
        return lhs,rhs

    # Parse the constraints string
    lines = constraints.splitlines()
    parsed = [] #XXX: in penalty_parser is eqconstraints, ineqconstraints
    xLHS, xRHS = [],[]
    for line in lines:
        if line.strip():
            constraint = _process_line(line)
            # Skip whenever '!=' is not the comparator
            if '!=' != comparator(constraint):
                continue
            eta = ' * (_tol(%(rhs)s,tol,rel) * 1.1) '#XXX: better 1.1*e_ or ???
            # collect the LHS and RHS of all != cases, to use later.
            split = constraint.split('!=') #XXX: better to use 1? eta? or ???
            expression = '%(lhs)s = %(lhs)s + equal(%(lhs)s,%(rhs)s)' + eta
            eqn = {'lhs':split[0].rstrip('=').strip(), \
                   'rhs':split[-1].lstrip('=').strip()}
            xLHS.append(eqn['lhs'])
            xRHS.append(eqn['rhs'])
            lhs, rhs = _process_expression(expression % eqn)
            parsed.append("=".join((lhs,rhs)))

    # iterate again, actually processing strings knowing where the '!=' are.
    for line in lines:
        if line.strip():
            constraint = _process_line(line)
            # Skip whenever '!=' is the comparator
            eps = comparator(constraint)
            if eps == '!=':
                continue  #XXX: use 1.1? or ???
            eta = '(_tol(%(rhs)s,tol,rel) * any(equal(%(rhs)s,%(neq)s)))'
            # Use eta whenever '<=' or '>=' is comparator
            eta = (' + ' + eta) if '>=' == eps else ((' - ' + eta) if '<=' == eps else '') #XXX: '>' in eps, or '>=' == eps?
            # Use epsilon whenever '<' or '>' is comparator
            eps = ' + e_ ' if eps == '>' else (' - e_ ' if eps == '<' else '')

            # convert "<" to min(LHS, RHS) and ">" to max(LHS,RHS)
            split = constraint.split('>')
            expression = '%(lhs)s = max(%(rhs)s, %(lhs)s)'
            if len(split) == 1: # didn't contain '>' or '!='
                split = constraint.split('<')
                expression = '%(lhs)s = min(%(rhs)s, %(lhs)s)'
            if len(split) == 1: # didn't contain '>', '<', or '!='
                split = constraint.split('=')
                expression = '%(lhs)s = %(rhs)s'
            if len(split) == 1: # didn't contain '>', '<', '!=', or '='
                print("Invalid constraint: %s" % constraint)
            eqn = {'lhs':split[0].rstrip('=').strip(), \
                   'rhs':split[-1].lstrip('=').strip()}
            # get list of LHS,RHS that != forces not to appear
            eqn['neq'] = '[' + ','.join(j for (i,j) in zip(xLHS+xRHS,xRHS+xLHS) if eqn['lhs'] == i) + ']'
            eqn['rhs'] += eps.replace('e_', '_tol(%(rhs)s,tol,rel)' % eqn) \
                          or eta % eqn
            expression = "=".join(_process_expression(expression % eqn))

            parsed.append(expression)

    return tuple(reversed(parsed))


doc_generate_conditions = """generate penalty condition functions from a set of constraint strings

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality and/or inequality
        constraints. Standard python syntax should be followed (with the
        math and numpy modules already imported).

    NOTE: Alternately, constraints may be a tuple of strings of symbolic
          constraints. Will return a tuple of penalty condition functions.

    For example:
        >>> constraints = '''
        ...     x0**2 = 2.5*x3 - 5.0
        ...     exp(x2/x0) >= 7.0'''
        >>> ineqf,eqf = generate_conditions(constraints, nvars=4)
        >>> print(ineqf[0].__doc__)
        '-(exp(x[2]/x[0]) - (7.0))'
        >>> ineqf[0]([1,0,1,0])
        4.2817181715409554
        >>> print(eqf[0].__doc__)
        'x[0]**2 - (2.5*x[3] - 5.0)'
        >>> eqf[0]([1,0,1,0])
        6.0

Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x2' in the example above).
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    locals -- a dictionary of additional variables used in the symbolic
        constraints equations, and their desired values.  Default is
        {'tol': 1e-15, 'rel': 1e-15}, where 'tol' and 'rel' are the absolute
        and relative difference from the extremal value in a given inequality.
        For more details, see `mystic.math.tolerance`.
"""
def_generate_conditions = '''
def generate_conditions(constraints, variables='x', nvars=None, locals=None):
    try:
        basestring
    except NameError:
        basestring = str
    if not isinstance(constraints, basestring):
        return tuple(generate_conditions(constraint, variables, nvars, locals) for constraint in constraints)

    ineqconstraints, eqconstraints = penalty_parser(constraints, \
                                      variables=variables, nvars=nvars)

    # parse epsilon
    if locals is None: locals = {}
    locals['tol'] = tol = locals['tol'] if 'tol' in locals else 1e-15
    locals['rel'] = rel = locals['rel'] if 'rel' in locals else 1e-15
    if tol < 0 or rel < 0:
        msg = 'math domain error'
        raise ValueError(msg)
    # default is globals with numpy and math imported
    globals = {}
    code = """from math import *; from numpy import *;"""
    code += """from numpy import mean as average;""" # use np.mean not average
   #code += """from mystic.math.measures import spread, variance, mean;"""
    code += """from mystic.math import tolerance as _tol;"""
    code = compile(code, '<string>', 'exec')
    %(exec_globals)s
    globals.update(locals) #XXX: allow this?
    
    # build an empty local scope to exec the code and build the functions
    results = {'equality':[], 'inequality':[]}
    for funcs, conditions in zip(['equality','inequality'], \
                                 [eqconstraints, ineqconstraints]):
      for func in conditions:
        fid = str(id(func))
        fdict = {'name':fid, 'equation':func, 'container':funcs}
        # build the condition function
        code = """
def {container}_{name}(x): return eval('{equation}')
{container}_{name}.__name__ = '{container}'
{container}_{name}.__doc__ = '{equation}'""".format(**fdict)
        #XXX: should locals just be the above dict of functions, or should we...
        # add the condition to container then delete the condition
        code += """
{container}.append({container}_{name})
del {container}_{name}""".format(**fdict)
        code = compile(code, '<string>', 'exec')
        %(exec_results)s

    #XXX: what's best form to return?  will couple these with ptypes
    return tuple(results['inequality']), tuple(results['equality'])
   #return results

generate_conditions.__doc__ = doc_generate_conditions
''' % dict(exec_globals=exec_globals, exec_results=exec_results)
exec(def_generate_conditions)
del def_generate_conditions, doc_generate_conditions

doc_generate_solvers = """generate constraints solver functions from a set of constraint strings

Inputs:
    constraints -- a string of symbolic constraints, with one constraint
        equation per line. Constraints can be equality and/or inequality
        constraints. Standard python syntax should be followed (with the
        math and numpy modules already imported). The left-hand side of
        each equation must be simplified to support assignment.

    NOTE: Alternately, constraints may be a tuple of strings of symbolic
          constraints. Will return a tuple of constraint solver functions.

    For example:
        >>> constraints = '''
        ...     x2 = x0/2.
        ...     x0 >= 0.'''
        >>> solv = generate_solvers(constraints, nvars=3)
        >>> print(solv[0].__doc__)
        'x[2] = x[0]/2.'
        >>> solv[0]([1,2,3])
        [1, 2, 0.5]
        >>> print(solv[1].__doc__)
        'x[0] = max(0., x[0])'
        >>> solv[1]([-1,2,3])
        [0.0, 2, 3]

Additional Inputs:
    nvars -- number of variables. Includes variables not explicitly
        given by the constraint equations (e.g. 'x2' in the example above).
    variables -- desired variable name. Default is 'x'. A list of variable
        name strings is also accepted for when desired variable names
        don't have the same base, and can include variables that are not
        found in the constraints equation string.
    locals -- a dictionary of additional variables used in the symbolic
        constraints equations, and their desired values.  Default is
        {'tol': 1e-15, 'rel': 1e-15}, where 'tol' and 'rel' are the absolute
        and relative difference from the extremal value in a given inequality.
        For more details, see `mystic.math.tolerance`.
"""
def_generate_solvers = '''
def generate_solvers(constraints, variables='x', nvars=None, locals=None):
    try:
        basestring
    except NameError:
        basestring = str
    if not isinstance(constraints, basestring):
        return tuple(generate_solvers(constraint, variables, nvars, locals) for constraint in constraints)

    _constraints = constraints_parser(constraints, \
                                      variables=variables, nvars=nvars)

    # parse epsilon
    if locals is None: locals = {}
    locals['tol'] = tol = locals['tol'] if 'tol' in locals else 1e-15
    locals['rel'] = rel = locals['rel'] if 'rel' in locals else 1e-15
    if tol < 0 or rel < 0:
        msg = 'math domain error'
        raise ValueError(msg)
    # default is globals with numpy and math imported
    globals = {}
    code = """from math import *; from numpy import *;"""
    code += """from numpy import mean as average;""" # use np.mean not average
    code += """from mystic.math.measures import spread, variance, mean;"""
    code += """from mystic.math.measures import impose_spread, impose_mean;"""
    code += """from mystic.math.measures import impose_sum, impose_product;"""
    code += """from mystic.math.measures import impose_variance;"""
    code += """from mystic.math import tolerance as _tol;"""
    code = compile(code, '<string>', 'exec')
    %(exec_globals)s
    globals.update(locals) #XXX: allow this?
    
    # build an empty local scope to exec the code and build the functions
    results = {'solver':[]}
    for func in _constraints:
        fid = str(id(func))
        fdict = {'name':fid, 'equation':func, 'container':'solver'}
        # build the condition function
        code = """
def {container}_{name}(x):
    '{equation}'
    exec('{equation}')
    return x
{container}_{name}.__name__ = '{container}'
""".format(**fdict)#XXX: better, check if constraint satisfied; if not, solve
        #XXX: should locals just be the above dict of functions, or should we
        #     add the condition to container then delete the condition?
        code += """
{container}.append({container}_{name})
del {container}_{name}""".format(**fdict)
        code = compile(code, '<string>', 'exec')
        %(exec_results)s

    #XXX: what's best form to return?  will couple these with ctypes ?
    return tuple(results['solver'])
   #return results

generate_solvers.__doc__ = doc_generate_solvers
''' % dict(exec_globals=exec_globals, exec_results=exec_results)
exec(def_generate_solvers)
del def_generate_solvers, doc_generate_solvers, exec_globals, exec_results


def generate_penalty(conditions, ptype=None, join=None, **kwds):
    """converts a penalty constraint function to a ``mystic.penalty`` function.

Args:
    conditions (object): a penalty contraint function, or list of penalty
        constraint functions.
    ptype (object, default=None): a ``mystic.penalty`` type, or a list of
        ``mystic.penalty`` types of the same length as *conditions*.
    join (object, default=None): ``and_`` or ``or_`` from ``mystic.coupler``.
    k (int, default=None): penalty multiplier.
    h (int, default=None): iterative multiplier.

Returns:
    a ``mystic.penalty`` function built from the given constraints

Notes:
    If ``join=None``, then apply the given penalty constraints sequentially.
    Otherwise, apply the penalty constraints concurrently using a coupler.

Examples:
    >>> constraints = '''
    ...     x2 = x0/2.
    ...     x0 >= 0.'''
    >>> ineqf,eqf = generate_conditions(constraints, nvars=3)
    >>> penalty = generate_penalty((ineqf,eqf))
    >>> penalty([1.,2.,0.])
    25.0
    >>> penalty([1.,2.,0.5])
    0.0
"""
    # allow for single condition, list of conditions, or nested list
    if not list_or_tuple_or_ndarray(conditions):
        conditions = list((conditions,))
    else: pass #XXX: should be fine...

    # discover the nested structure of conditions and ptype
    nc = nt = 0
    if ptype is None or not list_or_tuple_or_ndarray(ptype):
        nt = -1
    else:
        while tuple(flatten(conditions, nc)) != tuple(flatten(conditions)):
            nc += 1
        while tuple(flatten(ptype, nt)) != tuple(flatten(ptype)):
            nt += 1

    if join is None: pass  # don't use 'and/or' to join the conditions
    #elif nc >= 2: # join when is tuple of tuples of conditions
    else: # always use join, if given (instead of only if nc >= 2)
        if nt >= nc: # there as many or more nested ptypes than conditions
            p = iter(ptype)
            return join(*(generate_penalty(c, next(p), **kwds) for c in conditions))
        return join(*(generate_penalty(c, ptype, **kwds) for c in conditions))
    # flatten everything and produce the penalty
    conditions = list(flatten(conditions))

    # allow for single ptype, list of ptypes, or nested list
    if ptype is None:
        ptype = []
        from mystic.penalty import quadratic_equality, quadratic_inequality
        for condition in conditions:
            if 'inequality' in condition.__name__: 
                ptype.append(quadratic_inequality)
            else:
                ptype.append(quadratic_equality)
    elif not list_or_tuple_or_ndarray(ptype):
        ptype = list((ptype,))*len(conditions)
    else: pass #XXX: is already a list, should be the same len as conditions
    ptype = list(flatten(ptype))

    # iterate through penalties, building a compound penalty function
    pf = lambda x:0.0
    pfdoc = ""
    for penalty, condition in zip(ptype, conditions):
        pfdoc += "%s: %s\n" % (penalty.__name__, condition.__doc__)
        apply = penalty(condition, **kwds)
        pf = apply(pf)
    pf.__doc__ = pfdoc.rstrip('\n')
    pf.__name__ = 'penalty'
    return pf


def generate_constraint(conditions, ctype=None, join=None, **kwds):
    """converts a constraint solver to a ``mystic.constraints`` function.

Args:
    conditions (object): a constraint solver, or list of constraint solvers.
    ctype (object, default=None): a ``mystic.coupler`` type, or a list of
        ``mystic.coupler`` types of the same length as *conditions*.
    join (object, default=None): ``and_`` or ``or_`` from ``mystic.constraints``.

Returns:
    a ``mystic.constaints`` function built from the given constraints

Notes:
    If ``join=None``, then apply the given constraints sequentially.
    Otherwise, apply the constraints concurrently using a constraints coupler.

Warning:
    This constraint generator doesn't check for conflicts in conditions, but
    simply applies conditions in the given order. This constraint generator
    assumes that a single variable has been isolated on the left-hand side
    of each constraints equation, thus all constraints are of the form
    "x_i = f(x)". This solver picks speed over robustness, and relies on
    the user to formulate the constraints so that they do not conflict.

Examples:
    >>> constraints = '''
    ...     x0 = cos(x1) + 2.
    ...     x1 = x2*2.'''
    >>> solv = generate_solvers(constraints)
    >>> constraint = generate_constraint(solv)
    >>> constraint([1.0, 0.0, 1.0])
    [1.5838531634528576, 2.0, 1.0]

    Standard python math conventions are used. For example, if an ``int``
    is used in a constraint equation, one or more variable may be evaluate
    to an ``int`` -- this can affect solved values for the variables.

    >>> constraints = '''
    ...     x2 = x0/2.
    ...     x0 >= 0.'''
    >>> solv = generate_solvers(constraints, nvars=3)
    >>> print(solv[0].__doc__)
    'x[2] = x[0]/2.'
    >>> print(solv[1].__doc__)
    'x[0] = max(0., x[0])'
    >>> constraint = generate_constraint(solv)
    >>> constraint([1,2,3])
    [1, 2, 0.5]
    >>> constraint([-1,2,-3])
    [0.0, 2, 0.0]
"""
    # allow for single condition, list of conditions, or nested list
    if not list_or_tuple_or_ndarray(conditions):
        conditions = list((conditions,))
    else: pass #XXX: should be fine...

    # discover the nested structure of conditions and ctype
    nc = nt = 0
    if ctype is None or not list_or_tuple_or_ndarray(ctype):
        nt = -1
    else:
        while tuple(flatten(conditions, nc)) != tuple(flatten(conditions)):
            nc += 1
        while tuple(flatten(ctype, nt)) != tuple(flatten(ctype)):
            nt += 1

    if join is None: pass  # don't use 'and/or' to join the conditions
    #elif nc >= 2: # join when is tuple of tuples of conditions
    else: # always use join, if given (instead of only if nc >= 2)
        if nt >= nc: # there as many or more nested ctypes than conditions
            p = iter(ctype)
            return join(*(generate_constraint(c, next(p), **kwds) for c in conditions))
        return join(*(generate_constraint(c, ctype, **kwds) for c in conditions))
    # flatten everything and produce the constraint
    conditions = list(flatten(conditions))

    # allow for single ctype, list of ctypes, or nested list
    if ctype is None:
        from mystic.coupler import inner #XXX: outer ?
        ctype = list((inner,))*len(conditions)
    elif not list_or_tuple_or_ndarray(ctype):
        ctype = list((ctype,))*len(conditions)
    else: pass #XXX: is already a list, should be the same len as conditions
    ctype = list(flatten(ctype))

    # iterate through solvers, building a compound constraints solver
    cf = lambda x:x
    cfdoc = ""
    for wrapper, condition in zip(ctype, conditions):
        cfdoc += "%s: %s\n" % (wrapper.__name__, condition.__doc__)
        apply = wrapper(condition, **kwds)
        cf = apply(cf)
    cf.__doc__ = cfdoc.rstrip('\n')
    cf.__name__ = 'constraint'
    return cf


# EOF
