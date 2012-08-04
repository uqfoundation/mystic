#! /usr/bin/env python

"""
Tests of factories that enable verbose and compound termination conditions
"""

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


if __name__ == '__main__':
    info = 'self' #False #True
    _all = And #__and
    _any = Or #__or
    _one = When #__when

    from mystic.solvers import DifferentialEvolutionSolver
    s = DifferentialEvolutionSolver(4,4)

    from mystic.termination import VTR
    from mystic.termination import ChangeOverGeneration
    from mystic.termination import NormalizedChangeOverGeneration
    v = VTR()
    c = ChangeOverGeneration()
    n = NormalizedChangeOverGeneration()

    print "define conditions..."
    _v = _one(v)
    _c = _one(c)
    _n = _one(n)
    vAc = _all(v,c)
    vAn = _all(v,n)
    vOc = _any(v,c)
    vOn = _any(v,n)
    vAc_On = _any(vAc,_n)
    vAc_Oc = _any(vAc,_c)
    vAn_On = _any(vAn,_n)
    print "_v:", _v
    print "_c:", _c
    print "_n:", _n
    print "vAc:", vAc
    print "vAn:", vAn
    print "vOc:", vOc
    print "vOn:", vOn
    print "vAc_On:", vAc_On
    print "vAc_Oc:", vAc_Oc
    print "vAn_On:", vAn_On

    print "initial conditions..."
    print "vAc:", vAc(s, info)
    print "vAn:", vAn(s, info)
    print "vOc:", vOc(s, info)
    print "vOn:", vOn(s, info)

    print "after convergence toward zero..."
    s.energy_history = [0,0,0,0,0,0,0,0,0,0,0,0]
    print "vAc:", vAc(s, info)
    print "vAn:", vAn(s, info)
    print "vOc:", vOc(s, info)
    print "vOn:", vOn(s, info)

    print "nested compound termination..."
    print "vAc_On:", vAc_On(s, info)
    print "vAc_Oc:", vAc_Oc(s, info)
    print "vAn_On:", vAn_On(s, info)

    print "individual conditions..."
    print "v:", _v(s, info)
    print "c:", _c(s, info)
    print "n:", _n(s, info)

# EOF
