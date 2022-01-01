#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2021-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import mystic.symbolic as ms
from mystic.constraints import and_ as _and, or_ as _or
from mystic.coupler import and_, or_


if __name__ == '__main__':
    eps = 1e-16

    equations = """
    A*B + C > 1
    B < 0
    """
    print(equations)

    var = list('ABC')
    eqns = ms.simplify(equations, variables=var, all=True)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([1,-2,1])

    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))

    equations = """
    A*(B-C) + 2*C > 1
    B + C = 2*D
    D < 0
    """
    print(equations)
    var = list('ABCD')
    eqns = ms.simplify(equations, variables=var, all=True)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([1,2,1,-1])

    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))
    #XXX: worry about ZeroDivisionError ?

    equations = """
    A*B*C > 1
    A*C < 3
    -A*B > 4
    """
    print(equations)
    vars = list('ABC')
    eqns = ms.simplify(equations, variables=var, all=True)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([1,2,-1])

    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))

    equations = """
    A*B*C > 1
    A + C < 3
    A - B > 4
    """
    print(equations)
    vars = list('ABC')
    eqns = ms.simplify(equations, variables=var, all=True)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([1,2,-1])

    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))

    equations = """
    A*B + 2*C > 1
    B < 0
    C > 0
    """
    print(equations)
    vars = list('ABC')
    eqns = ms.simplify(equations, variables=var, all=True)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([1,2,-1])

    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))

    equations = """
    A*B > 1
    B > 0
    """
    print(equations)
    vars = list('AB')
    eqns = ms.simplify(equations, variables=var, all=True)
    print(eqns)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([1,2])

    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))


    equations = '''
    B == 2
    P == 2
    T < 5
    M > 0
    T == B + P + M
    '''
    print(equations)
    var = list('MTBP')
    eqns = ms.simplify(equations, variables=var, all=True)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([4,4,4,4])
    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))


    equations = """
    2*A + 3*B >= C
    -C + D == -B
    A*B > D
    D > 0
    """

    var = list('ABCD')
    eqns = ms.simplify(equations, variables=var, all=True)
    if isinstance(eqns, str):
      _join = join_ = None
      print(eqns)
    else:
      _join,join_ = _or,or_
      for eqn in eqns:
        print(eqn + '\n----------------------')

    constrain = ms.generate_constraint(ms.generate_solvers(eqns, var, locals=dict(e_=eps)), join=_join)
    solution = constrain([4,4,4,4])
    print('solved: %s' % dict(zip(var, solution)))

    penalty = ms.generate_penalty(ms.generate_conditions(eqns, var, locals=dict(e_=eps)), join=join_)
    print('penalty: %s' % penalty(solution))


# EOF
