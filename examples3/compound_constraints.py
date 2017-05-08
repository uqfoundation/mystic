# constraints
def _and(*constraints, **settings):
    """combine several constraints into a single constraint

Inputs:
    constraints -- constraint functions (or constraint solvers)

Additional Inputs:
    maxiter -- maximum number of iterations to attempt to solve [default: 100]

NOTE: 
    If a repeating cycle is detected, some of the inputs may be randomized.
    """
    import itertools as it
    import random as rnd
    n = len(constraints)
    maxiter = settings.pop('maxiter', 100) * n
    def _constraint(x): #XXX: inefficient, rewrite without append
        x = [x]
        # apply all constaints once
        for c in constraints:
            x.append(c(x[-1]))
        if all(xi == x[-1] for xi in x[1:]): return x[-1]
        # cycle constraints until there's no change
        _constraints = it.cycle(constraints) 
        for j in range(n,maxiter):
            x.append(next(_constraints)(x[-1]))
            if all(xi == x[-1] for xi in x[-n:]): return x[-1]
            # may be trapped in a cycle... randomize
            if x[-1] == x[-(n+1)]:
                x[-1] = [(i+rnd.randint(-1,1))*rnd.random() for i in x[-1]]
            if not j%(2*n):
                del x[:n]
        # give up
        return x[-1] #XXX: or fail by throwing Error?
    return lambda x: _constraint(x)


def _or(*constraints, **settings):
    """create a constraint that is satisfied if any constraints are satisfied

Inputs:
    constraints -- constraint functions (or constraint solvers)

Additional Inputs:
    maxiter -- maximum number of iterations to attempt to solve [default: 100]

NOTE: 
    If a repeating cycle is detected, some of the inputs may be randomized.
    """
    import itertools as it
    import random as rnd
    n = len(constraints)
    maxiter = settings.pop('maxiter', 100) * n
    def _constraint(x): #XXX: inefficient, rewrite without append
        x = [x]
        # check if initial input is valid
        for c in constraints:
            x.append(c(x[0]))
            if x[-1] == x[0]: return x[-1]
        # cycle constraints until there's no change
        _constraints = it.cycle(constraints) 
        for j in range(n,maxiter):
            x.append(next(_constraints)(x[-n]))
            if x[-1] == x[-(n+1)]: return x[-1]
            else: # may be trapped in a rut... randomize
                x[-1] = x[-rnd.randint(1,n)]
            if not j%(2*n):
                del x[:n]
        # give up
        return x[-1] #XXX: or fail by throwing Error?
    return lambda x: _constraint(x)


def _not(constraint, **settings): #FIXME: not a decorator; use constraint API
    """invert the region where the given constraints are valid, then solve

Inputs:
    constraint -- constraint function (or constraint solver)

Additional Inputs:
    maxiter -- maximum number of iterations to attempt to solve [default: 100]

NOTE: 
    If a repeating cycle is detected, some of the inputs may be randomized.
    """
    import random as rnd
    maxiter = settings.pop('maxiter', 100)
    def _constraint(x):
        # check if initial input is valid, else randomize and try again
        for j in range(0,maxiter):
            if constraint(x) != x: return x
            x = [(i+rnd.randint(-1,1))*rnd.random() for i in x]
        # give up
        return x #XXX: or fail by throwing Error?
    return lambda x: _constraint(x)


# penalties   #XXX: the following was initially 'constraints.combined'
def and_(*penalties, **settings): #FIXME: isn't a decorator; use penalty API
    """combine several penalties into a single penalty function by summation

Inputs:
    penalties -- penalty functions (or penalty conditions)

Additional Inputs:
    ptype -- penalty function type [default: linear_equality]
    args -- arguments for the penalty function [default: ()]
    kwds -- keyword arguments for the penalty function [default: {}]
    k -- penalty multiplier [default: 1]
    h -- iterative multiplier [default: 5]

NOTE: The defaults provide a linear combination of the individual penalties
    without any scaling. A different ptype (from 'mystic.penalty') will
    apply a nonlinear scaling to the combined penalty, while a different
    k will apply a linear scaling.

NOTE: This function is also useful for combining constraints solvers
    into a single constraints solver, however can not do so directly.  
    Constraints solvers must first be converted to penalty functions
    (i.e. with 'as_penalty'), then combined, then can be converted to
    a constraints solver (i.e. with 'as_constraint'). The resulting
    constraints will likely be more expensive to evaluate and less
    accurate than writing the constraints solver from scratch.
    """
    k = settings.setdefault('k', 1)
    if k is None: del settings['k']
    ptype = settings.pop('ptype', None)
    if ptype is None:
        from mystic.penalty import linear_equality as ptype
    penalty = lambda x: sum(p(x) for p in penalties)
    return ptype(penalty, **settings)(lambda x:0.)


def or_(*penalties, **settings): #FIXME: isn't a decorator; use penalty API
    """create a single penalty that selects the minimum of several penalties

Inputs:
    penalties -- penalty functions (or penalty conditions)

Additional Inputs:
    ptype -- penalty function type [default: linear_equality]
    args -- arguments for the penalty function [default: ()]
    kwds -- keyword arguments for the penalty function [default: {}]
    k -- penalty multiplier [default: 1]
    h -- iterative multiplier [default: 5]

NOTE: The defaults provide a linear combination of the individual penalties
    without any scaling. A different ptype (from 'mystic.penalty') will
    apply a nonlinear scaling to the combined penalty, while a different
    k will apply a linear scaling.

NOTE: This function is also useful for combining constraints solvers
    into a single constraints solver, however can not do so directly.  
    Constraints solvers must first be converted to penalty functions
    (i.e. with 'as_penalty'), then combined, then can be converted to
    a constraints solver (i.e. with 'as_constraint'). The resulting
    constraints will likely be more expensive to evaluate and less
    accurate than writing the constraints solver from scratch.
    """
    k = settings.setdefault('k', 1)
    if k is None: del settings['k']
    ptype = settings.pop('ptype', None)
    if ptype is None:
        from mystic.penalty import linear_equality as ptype
    penalty = lambda x: min(p(x) for p in penalties)
    return ptype(penalty, **settings)(lambda x:0.)


def not_(penalty, **settings): #FIXME: isn't a decorator; use penalty API
    """invert, so penalizes the region where the given penalty is valid

Inputs:
    penalty -- a penalty function (or penalty condition)

Additional Inputs:
    ptype -- penalty function type [default: linear_equality]
    args -- arguments for the penalty function [default: ()]
    kwds -- keyword arguments for the penalty function [default: {}]
    k -- penalty multiplier [default: 1]
    h -- iterative multiplier [default: 5]
    """
    k = settings.setdefault('k', 1)
    if k is None: del settings['k']
    ptype = settings.pop('ptype', None)
    if ptype is None:
        import mystic.penalty as mp
        try:
            ptype = getattr(mp, penalty.ptype)
        except AttributeError: 
            ptype = mp.linear_equality
    try:
        condition = penalty.func # is a penalty
    except AttributeError:
        condition = penalty # is a raw condition
    if ptype.__name__.endswith('_inequality'):
        _penalty = lambda x: 0 - condition(x)
    else:
        _penalty = lambda x: not condition(x)
    return ptype(_penalty, **settings)(lambda x:0.)


if __name__ == '__main__':
    from mystic.penalty import quadratic_inequality, quadratic_equality

    p1 = lambda x: sum(x) - 5
    p2 = lambda x: min(i**2 for i in x)
    p = p1,p2

    p = [quadratic_inequality(pi)(lambda x:0.) for pi in p]
    p1,p2 = p
    penalty = and_(*p)

    x = [[1,2],[-2,-1],[5,-5]]
    for xi in x:
        assert p1(xi) + p2(xi) == penalty(xi)

    penalty = or_(*p)
    for xi in x:
        assert min(p1(xi),p2(xi)) == penalty(xi)

    penalty = not_(p1)
    for xi in x:
        assert bool(p1(xi)) != bool(penalty(xi))
    penalty = not_(p2)
    for xi in x:
        assert bool(p2(xi)) != bool(penalty(xi))

    # constraints
    from mystic.constraints import with_mean, discrete
    @with_mean(5.0)
    def meanie(x):
      return x
    
    @discrete(range(11))
    def integers(x):
      return x
    
    c = _and(integers, meanie)
    x = c([1,2,3])
    assert x == integers(x) == meanie(x)
    x = c([9,2,3])
    assert x == integers(x) == meanie(x)
    x = c([0,-2,3])
    assert x == integers(x) == meanie(x)
    x = c([9,-200,344])
    assert x == integers(x) == meanie(x)

    c = _or(meanie, integers)
    x = c([1.1234, 4.23412, -9])
    assert x == meanie(x) and x != integers(x)
    x = c([7.0, 10.0, 0.0])
    assert x == integers(x) and x != meanie(x)
    x = c([6.0, 9.0, 0.0])
    assert x == integers(x) == meanie(x)
    x = c([3,4,5])
    assert x == integers(x) and x != meanie(x)
    x = c([3,4,5.5])
    assert x == meanie(x) and x != integers(x)

    c = _not(integers)
    x = c([1,2,3])
    assert x != integers(x) and x != [1,2,3] and x == c(x)
    x = c([1.1,2,3])
    assert x != integers(x) and x == [1.1,2,3] and x == c(x)
    c = _not(meanie)
    x = c([1,2,3])
    assert x != meanie(x) and x == [1,2,3] and x == c(x)
    x = c([4,5,6])
    assert x != meanie(x) and x != [4,5,6] and x == c(x)
    c = _not(_and(meanie, integers))
    x = c([4,5,6])
    assert x != meanie(x) and x != integers(x) and x != [4,5,6] and x == c(x)


