#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2025-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import os
import numpy as np
from mystic.samplers import SparsitySampler
from mystic.monitors import Monitor, LoggingMonitor
from mystic.solvers import PowellDirectionalSolver
from mystic.termination import NormalizedChangeOverGeneration as NCOG
from mystic.tools import listify as tolist
from mystic.cache.archive import file_archive, read as get_db
from mystic.models import rosen as cost
from mystic.bounds import Bounds
bounds = Bounds(0,5,n=4)


def test_reset(s):
    si0 = s.iters()
    se0 = s.evals()
    ai0 = s._sampler._all_iters
    ae0 = s._sampler._all_evals
    ls0 = len(s._sampler._stepmon)
    st0 = s._sampler.Terminated(all=True)
    #print(st0)
    assert not any(st0)
    assert not si0
    assert not se0
    assert not any(ai0)
    assert not any(ae0)
    assert not ls0

    # reset non-terminated solvers [None]
    s.sample(if_terminated=False, reset_all=True)
    st1 = s._sampler.Terminated(all=True)
    si1 = s.iters()
    se1 = s.evals()
    ai1 = s._sampler._all_iters
    ae1 = s._sampler._all_evals
    ls1 = len(s._sampler._stepmon)
    #print(st1)
    assert not any(st1)
    assert si1
    assert se1
    assert not any(ai1)
    assert sum(ae1) == N
    assert ls1 == 1

    # reset non-terminated solvers [None]
    s.sample(if_terminated=False, reset_all=True)
    st2 = s._sampler.Terminated(all=True)
    si2 = s.iters()
    se2 = s.evals()
    ai2 = s._sampler._all_iters
    ae2 = s._sampler._all_evals
    ls2 = len(s._sampler._stepmon)
    #print(st1)
    assert not any(st2)
    assert si2 > si1
    assert se2 > se1
    assert not any(ai2)
    assert sum(ae2) == N
    assert ls2 == 1

    # reset regardless
    s.sample(if_terminated=None, reset_all=True)
    st3 = s._sampler.Terminated(all=True)
    si3 = s.iters()
    se3 = s.evals()
    ai3 = s._sampler._all_iters
    ae3 = s._sampler._all_evals
    ls3 = len(s._sampler._stepmon)
    #print(st3)
    assert not any(st3)
    assert si3 > si2
    assert se3 > se2
    assert not any(ai3)
    assert sum(ae3) == N
    assert ls3 == 1

    # never reset
    s.sample(if_terminated=all, reset_all=None)
    st4 = s._sampler.Terminated(all=True)
    si4 = s.iters()
    se4 = s.evals()
    ai4 = s._sampler._all_iters
    ae4 = s._sampler._all_evals
    ls4 = len(s._sampler._stepmon)
    #print(st4)
    assert not any(st4)
    assert si4 > si3
    assert se4 > se3
    assert sum(ai4) > sum(ai3)
    assert sum(ae4) > sum(ae3)
    assert ls4 > ls3

    # never reset
    s.sample(if_terminated=all, reset_all=None)
    st5 = s._sampler.Terminated(all=True)
    si5 = s.iters()
    se5 = s.evals()
    ai5 = s._sampler._all_iters
    ae5 = s._sampler._all_evals
    ls5 = len(s._sampler._stepmon)
    #print(st5)
    assert not any(st5)
    assert si5 > si4
    assert se5 > se4
    assert sum(ai5) > sum(ai4)
    assert sum(ae5) > sum(ae4)
    assert ls5 > ls4

    # sample until any terminated, never reset
    while not any(s._sampler.Terminated(all=True)):
        s.sample(if_terminated=any, reset_all=None)

    st6 = s._sampler.Terminated(all=True)
    si6 = s.iters()
    se6 = s.evals()
    ai6 = s._sampler._all_iters
    ae6 = s._sampler._all_evals
    ls6 = len(s._sampler._stepmon)
    #print(st6)
    assert any(st6)
    assert si6 > si5
    assert se6 > se5
    assert sum(ai6) > sum(ai5)
    assert sum(ae6) > sum(ae5)
    assert ls6 > ls5

    # terminated, but don't reset
    s.sample(if_terminated=any, reset_all=None)
    st7 = s._sampler.Terminated(all=True)
    si7 = s.iters()
    se7 = s.evals()
    ai7 = s._sampler._all_iters
    ae7 = s._sampler._all_evals
    ls7 = len(s._sampler._stepmon)
    #print(st7)
    assert any(st7)
    assert si7 > si6
    assert se7 > se6
    assert sum(ai7) == sum(ai6) + (N-sum(st6))
    assert sum(ae7) == sum(ae6) if all(st6) else sum(ae7) > sum(ae6)
    assert ls7 >= ls6 #XXX: better

    # terminated, but don't reset
    s.sample(if_terminated=all, reset_all=None)
    st8 = s._sampler.Terminated(all=True)
    si8 = s.iters()
    se8 = s.evals()
    ai8 = s._sampler._all_iters
    ae8 = s._sampler._all_evals
    ls8 = len(s._sampler._stepmon)
    #print(st8)
    assert any(st8)
    assert si8 > si7
    assert se8 > se7
    assert sum(ai8) == sum(ai7) + (N-sum(st7))
    assert sum(ae8) == sum(ae7) if all(st7) else sum(ae8) > sum(ae7)
    assert ls8 >= ls7 #XXX: better

    # terminated, and reset all
    s.sample(if_terminated=all, reset_all=True)
    st9 = s._sampler.Terminated(all=True)
    si9 = s.iters()
    se9 = s.evals()
    ai9 = s._sampler._all_iters
    ae9 = s._sampler._all_evals
    ls9 = len(s._sampler._stepmon)
    #print(st9)
    assert (not any(st9)) if all(st8) else any(st9)
    assert si9 > si8
    assert se9 > se8
    assert not any(ai9) if all(st8) else sum(ai9) > sum(ai8)
    assert sum(ae9) == N if all(st8) else sum(ae9) > sum(ae8)
    assert ls9 == 1 if all(st8) else ls9 >= ls8

    # sample until any terminated, reset all
    while not any(s._sampler.Terminated(all=True)):
        s.sample(if_terminated=any, reset_all=True)

    st10 = s._sampler.Terminated(all=True)
    si10 = s.iters()
    se10 = s.evals()
    ai10 = s._sampler._all_iters
    ae10 = s._sampler._all_evals
    ls10 = len(s._sampler._stepmon)
    #print(st10)
    assert any(st10)
    assert si10 >= si9 #XXX: better
    assert se10 >= se9 #XXX: better
    assert all(ai10)
    assert sum(ae10) > sum(ae9) if sum(st10) > sum(st9) else sum(ae10) == sum(ae9)
    assert ls10 > ls9 if sum(st10) > sum(st9) else ls10 == ls9

    # reset all regardless
    s.sample(if_terminated=None, reset_all=True)
    st11 = s._sampler.Terminated(all=True)
    si11 = s.iters()
    se11 = s.evals()
    ai11 = s._sampler._all_iters
    ae11 = s._sampler._all_evals
    ls11 = len(s._sampler._stepmon)
    #print(st11)
    assert not any(st11)
    assert si11 > si10
    assert se11 > se10
    assert not all(ai11)
    assert sum(ae11) == N
    assert ls11 == 1

    # reset all regardless
    s.sample(if_terminated=None, reset_all=True)
    st12 = s._sampler.Terminated(all=True)
    si12 = s.iters()
    se12 = s.evals()
    ai12 = s._sampler._all_iters
    ae12 = s._sampler._all_evals
    ls12 = len(s._sampler._stepmon)
    #print(st12)
    assert not any(st12)
    assert si12 > si11
    assert se12 > se11
    assert not all(ai12)
    assert sum(ae12) == N
    assert ls12 == 1


def test_init(s):
    s._reset_sampler()
    s.sample_until(terminated=any)
    t0 = s._sampler.Terminated(all=True)
    assert np.any(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    s.sample(None, True) # reset all solvers regardless
    tN = s._sampler.Terminated(all=True)
    assert not np.any(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 != xN, regardless
    assert np.all([j if i else j for i,j in zip(tN, np.any(np.array(x0) - xN, axis=1))]) 
    s._reset_sampler()
    s.sample_until(terminated=any)
    t0 = s._sampler.Terminated(all=True)
    assert np.any(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    s.sample(None, None) # never reset regardless [NOTE: does nothing]
    tN = s._sampler.Terminated(all=True)
    assert np.any(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 == xN, regardless
    assert np.all([~j if i else ~j for i,j in zip(tN, np.any(np.array(x0) - xN, axis=1))]) 
    t0 = s._sampler.Terminated(all=True)
    assert np.any(t0)
    s.sample(None, False) # reset terminated solvers regardless
    tN = s._sampler.Terminated(all=True)
    assert not np.any(tN)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    ## x0 != xN, if terminated
    assert np.all([j if i else ~j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])
    s.sample(False, True) # if none terminated, reset all solvers
    t0 = s._sampler.Terminated(all=True)
    assert not np.any(t0)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 != xN, if none terminated
    assert np.all([j if i else j for i,j in zip(tN, np.any(np.array(x0) - xN, axis=1))])
    s.sample(False, None) # if none terminated, never reset [NOTE: does nothing]
    tN = s._sampler.Terminated(all=True)
    assert not np.any(tN)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    ## x0 == xN, regardless
    assert np.all([~j if i else ~j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])
    s.sample(False, False) # if none terminated, reset terminated solvers (irrelevant)
    t0 = s._sampler.Terminated(all=True)
    assert not np.any(t0)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 == xN, regardless
    assert np.all([~j if i else ~j for i,j in zip(tN, np.any(np.array(x0) - xN, axis=1))])
    s._reset_sampler()
    s.sample_until(terminated=any)
    t0 = s._sampler.Terminated(all=True)
    assert np.any(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    s.sample(any, None) # if any terminated, never reset [NOTE: does nothing]
    tN = s._sampler.Terminated(all=True)
    assert np.any(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 == xN, regardless
    assert np.all([~j if i else ~j for i,j in zip(tN, np.any(np.array(x0) - xN, axis=1))])
    s.sample(any, False) # if any terminated, reset terminated solvers
    t0 = s._sampler.Terminated(all=True)
    assert not np.any(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    ## x0 != xN, if any terminated
    assert np.all([j if i else ~j for i,j in zip(tN, np.any(np.array(x0) - xN, axis=1))])
    s._reset_sampler()
    s.sample_until(terminated=any)
    t0 = s._sampler.Terminated(all=True)
    assert np.any(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    s.sample(any, True) # if any terminated, reset all solvers
    tN = s._sampler.Terminated(all=True)
    assert not np.any(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 != xN, if any terminated
    assert np.all([j if i else j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])
    s._reset_sampler()
    s.sample_until(terminated=all)
    t0 = s._sampler.Terminated(all=True)
    assert np.all(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    s.sample(all, None) # if all terminated, never reset [NOTE: does nothing]
    tN = s._sampler.Terminated(all=True)
    assert np.all(tN) 
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 == xN, regardless
    assert np.all([~j if i else ~j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])
    s.sample(all, True) # if all terminated, reset all solvers
    t0 = s._sampler.Terminated(all=True)
    assert not np.any(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    ## x0 != xN, if all terminated
    assert np.all([j if i else j for i,j in zip(tN, np.any(np.array(x0) - xN, axis=1))])
    s.sample(all, False) # if all terminated, reset terminated solvers
    tN = s._sampler.Terminated(all=True)
    assert not np.any(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 != xN, if all terminated
    assert np.all([~j if i else ~j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])
    s._reset_sampler()
    s.sample_until(terminated=all)
    t0 = s._sampler.Terminated(all=True)
    assert np.all(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    s.sample(True, None) # if best terminated, never reset [NOTE: does nothing]
    tN = s._sampler.Terminated(all=True)
    assert np.all(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 == xN, regardless
    assert np.all([~j if i else ~j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])
    s.sample(True, True) # if best terminated, reset all solvers
    tN = s._sampler.Terminated(all=True)
    assert not np.any(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 != xN, if best terminated
    assert np.all([j if i else j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])
    s._reset_sampler()
    s.sample_until(terminated=all)
    t0 = s._sampler.Terminated(all=True)
    assert np.all(t0)
    x0 = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert x0 == s._init_solution
    s.sample(True, False) # if best terminated, reset terminated solvers
    tN = s._sampler.Terminated(all=True)
    assert not np.any(tN)
    xN = [si._stepmon._x[0] for si in s._sampler._allSolvers]
    if hasattr(s, '_init_solution'):
        assert xN == s._init_solution
    ## x0 != xN, if best terminated
    assert np.all([j if i else j for i,j in zip(t0, np.any(np.array(x0) - xN, axis=1))])


if __name__ == '__main__':
    nx = 4; ny = None
    N = 4
    kwds = dict(npts=N, maxiter=100, maxfun=1000, id=0)
    s = SparsitySampler(bounds, cost, **kwds)
    test_reset(s)

    kwds = dict(npts=N, maxiter=10, maxfun=1000, id=0)
    s = SparsitySampler(bounds, cost, **kwds)
    test_reset(s)

    test_init(s)
