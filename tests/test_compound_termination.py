#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

"""
Tests of factories that enable verbose and compound termination conditions
"""
disp = 0

def __and(*funcs):
  "factory for compounding termination conditions with *and*"
  def term(solver, info=False): #FIXME: 'self' doesn't flatten
    stop = {}
    [stop.update({f : f(solver, info)}) for f in funcs]
    _all = all(stop.values())
    if not info: return _all
    if info == 'self': return tuple(stop.keys()) if _all else ()
    return "; ".join(set("; ".join(stop.values()).split("; "))) if _all else ""
  return term

def __or(*funcs):
  "factory for compounding termination conditions with *or*"
  def term(solver, info=False): #FIXME: 'self' doesn't flatten
    stop = {}
    [stop.update({f : f(solver, info)}) for f in funcs]
    _any = any(stop.values())
    if not info: return _any
    [stop.pop(cond) for (cond,met) in stop.items() if not met]
    if info == 'self': return tuple(stop.keys())
    return "; ".join(set("; ".join(stop.values()).split("; ")))
  return term

def __when(func):
  "factory for single termination condition cast as a compound condition"
  return __and(func)


from mystic.termination import And, Or, When


def test_me(info='self'):
    from mystic.solvers import DifferentialEvolutionSolver
    s = DifferentialEvolutionSolver(4,4)

    from mystic.termination import VTR
    from mystic.termination import ChangeOverGeneration
    from mystic.termination import NormalizedChangeOverGeneration
    v = VTR()
    c = ChangeOverGeneration()
    n = NormalizedChangeOverGeneration()

    if disp: print("define conditions...")
    _v = When(v)
    _c = When(c)
    _n = When(n)
    v_and_c = And(v,c)
    v_and_n = And(v,n)
    v_or_c = Or(v,c)
    v_or_n = Or(v,n)
    v_and_c__or_n = Or(v_and_c,_n)
    v_and_c__or_c = Or(v_and_c,_c)
    v_and_n__or_n = Or(v_and_n,_n)
    c_and_v = And(c,v)
    c_or_v = Or(c,v)
    c_or__c_and_v = Or(_c,c_and_v)

    assert len(_v) == len(_c) == len(_n) == 1
    assert list(_v).index(v) == list(_c).index(c) == list(_n).index(n) == 0
    assert list(_v).count(v) == list(_c).count(c) == list(_n).count(n) == 1
    assert v in _v and c in _c and n in _n
    assert v in v_and_c and c in v_and_c
    assert v in v_and_n and n in v_and_n
    assert v in v_or_c and c in v_or_c
    assert v in v_or_n and n in v_or_n
    assert len(v_and_c) == len(v_and_n) == len(v_or_c) == len(v_or_n) == 2
    assert len(v_and_c__or_n) == len(v_and_c__or_c) == len(v_and_n__or_n) == 2
    assert v not in v_and_c__or_n and c not in v_and_c__or_n and n not in v_and_c__or_n
    assert v_and_c in v_and_c__or_n and _n in v_and_c__or_n
    assert v_and_c in v_and_c__or_c and _c in v_and_c__or_c
    assert v_and_n in v_and_n__or_n and _n in v_and_n__or_n
    assert c in c_and_v and v in c_and_v
    assert c in c_or_v and v in c_or_v
    assert list(c_and_v).index(c) == list(v_and_c).index(v)
    assert list(c_and_v).index(v) == list(v_and_c).index(c)
    assert list(c_or_v).index(c) == list(v_or_c).index(v)
    assert list(c_or_v).index(v) == list(v_or_c).index(c)
    assert c_and_v in c_or__c_and_v and _c in c_or__c_and_v

    if disp:
        print("_v:%s" % _v)
        print("_c:%s" % _c)
        print("_n:%s" % _n)
        print("v_and_c:%s" % v_and_c)
        print("v_and_n:%s" % v_and_n)
        print("v_or_c:%s" % v_or_c)
        print("v_or_n:%s" % v_or_n)
        print("v_and_c__or_n:%s" % v_and_c__or_n)
        print("v_and_c__or_c:%s" % v_and_c__or_c)
        print("v_and_n__or_n:%s" % v_and_n__or_n)
        print("c_and_v:%s" % c_and_v)
        print("c_or_v:%s" % c_or_v)
        print("c_or__c_and_v:%s" % c_or__c_and_v)

    if disp: print("initial conditions...")
    _v_and_c = v_and_c(s, info)
    _v_and_n = v_and_n(s, info)
    _v_or_c = v_or_c(s, info)
    _v_or_n = v_or_n(s, info)

    assert not _v_and_c
    assert not _v_and_n
    assert not _v_or_c
    assert not _v_or_n

    if disp:
        print("v_and_c:%s" % _v_and_c)
        print("v_and_n:%s" % _v_and_n)
        print("v_or_c:%s" % _v_or_c)
        print("v_or_n:%s" % _v_or_n)

    if disp: print("after convergence toward zero...")
    s.energy_history = [0,0,0,0,0,0,0,0,0,0,0,0]
    # _v and _n are True, _c is False
    _v_and_c = v_and_c(s, info)
    _v_and_n = v_and_n(s, info)
    _v_or_c = v_or_c(s, info)
    _v_or_n = v_or_n(s, info)

    assert not _v_and_c
    assert _v_and_n
    assert _v_or_c
    assert _v_or_n
    if info == 'self':
        assert v in _v_and_n and n in _v_and_n
        assert v in _v_or_c and c not in _v_or_c
        assert v in _v_or_n and n in _v_or_n

    if disp:
        print("v_and_c:%s" % _v_and_c)
        print("v_and_n:%s" % _v_and_n)
        print("v_or_c:%s" % _v_or_c)
        print("v_or_n:%s" % _v_or_n)

    if disp: print("nested compound termination...")
    # _v and _n are True, _c is False
    _v_and_c__or_n = v_and_c__or_n(s, info)
    _v_and_c__or_c = v_and_c__or_c(s, info)
    _v_and_n__or_n = v_and_n__or_n(s, info)

    assert _v_and_c__or_n
    assert not _v_and_c__or_c
    assert _v_and_n__or_n
    if info == 'self':
        assert _n in _v_and_c__or_n and v_and_c not in _v_and_c__or_n
        assert v not in _v_and_c__or_n and c not in _v_and_c__or_n
        assert v_and_n in _v_and_n__or_n and _n in _v_and_n__or_n
        assert v not in _v_and_n__or_n and n not in _v_and_n__or_n

    if disp:
        print("v_and_c__or_n:%s" % _v_and_c__or_n)
        print("v_and_c__or_c:%s" % _v_and_c__or_c)
        print("v_and_n__or_n:%s" % _v_and_n__or_n)

    if disp: print("individual conditions...")
    __v = _v(s, info)
    __c = _c(s, info)
    __n = _n(s, info)

    assert __v and __n
    assert not __c
    if info == 'self':
        assert v in __v and n in __n

    if disp:
        print("v:%s" % __v)
        print("c:%s" % __c)
        print("n:%s" % __n)

    if disp: print("conditions with false first...")
    _c_and_v = c_and_v(s, info)
    _c_or_v = c_or_v(s, info)
    _c_or__c_and_v = c_or__c_and_v(s, info)

    assert not _c_and_v
    assert _c_or_v
    assert not _c_or__c_and_v
    if info == 'self':
        assert v in _c_or_v and c not in _c_or_v

    if disp:
        print("c_and_v:%s" % _c_and_v)
        print("c_or_v:%s" % _c_or_v)
        print("c_or__c_and_v:%s" % _c_or__c_and_v)


if __name__ == '__main__':
    test_me(False)
    test_me(True)
    test_me()


# EOF
