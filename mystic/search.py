#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
a global searcher
"""
#FIXME: refactor to use mystic.samplers (and code in _workflow)

class Searcher(object):
   #searcher has:
   #    sprayer - an ensemble algorithm
   #    archive - a sampled point archive(s)
   #    cache - a trajectory cache(s)
   #    retry - max consectutive retries w/o an archive 'miss'
   #    repeat - number of times to repeat the search
   #    tol - minima comparator rounding precision
   #    _allSolvers - collection of sprayers (for all solver trajectories)
   #
   #searcher (or archive) has:
   #    memtol - memoization rounding precision
   #
   #searcher (or sprayer) has:
   #    seeker - an optimiztion algorithm
   #    map - an enselble map
   #    traj - trajectory configuration (bool)
   #    disp - verbosity configuration (bool)
   #    npts - number of solvers
   #
   #searcher can:
   #    _memoize - apply caching archive to sprayer
   #    _search - perform a single search iteration (the search algorithm ?)
   #    Search - keep performing _search until retry condition is met
   #    UseTrajectories - save all sprayers, thus save all trajectories
   #    Reset - clear caching archive and saved sprayers (thus trajectories)
   #    Trajectories - fetch (step, param, cost) for all solver trajectories
   #    Samples - all sampled points as a n-D array (output is array[-1])
   #    Values - fetch values (model(x)) for sampled points on surface (model)
   #    Coordinates - fetch inputs (x) for sampled points on surface (model)
   #    Minima - fetch {x: model(x)} for all x that minimize model
   #    _summarize - print diagnostics on the value/quantity of solved values
   #
   # searcher (or sprayer) can:
   #    _print - print bestSolution and bestEnergy for each sprayer
   #    _configure (model, bounds, stop, monitor) - configure sprayer
   #    _solve - spray multiple seekers

    def __init__(self, npts=4, retry=1, tol=8, memtol=1, memsize=None,
                       map=None, archive=None, cache=None, sprayer=None,
                       seeker=None, traj=False, disp=False, repeat=0):
        """searcher, which searches for all minima of a response surface

        Input:
          npts - number of solvers in the ensemble
          retry - max consectutive retries w/o a cache 'miss'
          tol - rounding precision for the minima comparator
          memtol - rounding precision for memoization
          memsize - maximum size of cache to hold in memory
          map - map used for spawning solvers in the ensemble
          archive - the sampled point archive(s)
          cache - the trajectory cache(s)
          sprayer - the mystic.ensemble instance
          seeker - the mystic.solvers instance
          traj - if True, save the parameter trajectories
          disp - if True, be verbose
          repeat - number of times to repeat the search
        """
        #XXX: better not to use klepto as default? just dict and false cache?
        from klepto.archives import dict_archive as _archive
        from mystic.solvers import BuckshotSolver #XXX: or SparsitySolver?
        from mystic.solvers import PowellDirectionalSolver
        from mystic.pools import SerialPool as Pool
        import sys
        if (sys.hexversion >= 0x30000f0):
            from builtins import map as _map
        else:
            from __builtin__ import map as _map
        del sys
        self.archive = _archive(cached=False) if archive is None else archive
        self.cache = _archive(cached=False) if cache is None else cache
        self.sprayer = BuckshotSolver if sprayer is None else sprayer
        self.seeker = PowellDirectionalSolver if seeker is None else seeker
        self.map = Pool().map if map in (None, _map) else map 
        self.traj = traj
        self.disp = disp

        self.npts = npts # number of solvers
        self.retry = retry   # max consectutive retries w/o a cache 'miss'
        self.repeat = repeat # number of times to repeat the search
        self.tol = tol       # rounding precision
        self.memtol = memtol # memoization rounding precision
        self.memsize = memsize # max in-memory cache size
        self._allSolvers = []
        self._inv = False    # self-awareness: am I inverted (maximizing)?
        return

    def _print(self, solver, tol=8):
        """print bestSolution and bestEnergy for each sprayer"""
        l = -1 if self._inv else 1
        for _solver in solver._allSolvers:
            bestSol = tuple(round(s, tol) for s in _solver.bestSolution)
            bestRes = round(_solver.bestEnergy, tol)
            print("%s %s" % (bestSol, l*bestRes))

    def _memoize(self, solver, tol=1, all=False, size=None):
        """apply caching to ensemble solver instance"""
        archive = self.archive if all else self.cache

        from klepto import lru_cache as _cache #XXX: or lru_? or ?
        from klepto.keymaps import keymap

        km = keymap()
        ca = _cache(maxsize=size, ignore=('**','out'),
                    tol=tol, cache=archive, keymap=km)

        @ca
        def memo(*args, **kwds):
            return kwds['out']

        l = -1 if self._inv else 1
        if all: #FIXME: applied *after* _solve; should be *during* _solve
            cache = memo.__cache__()
            for _solver in solver._allSolvers:
                if _solver._evalmon:
                    param = _solver._evalmon._x
                    cost = _solver._evalmon._y             #FIXME: python2.5
                    cache.update((tuple(x),l*y) for (x,y) in zip(param,cost))
        else:
            for _solver in solver._allSolvers:
                bestSol = tuple(_solver.bestSolution)
                bestRes = float(_solver.bestEnergy)
                memo(*bestSol, out=l*bestRes)  #FIXME: python2.5
        return memo

    #FIXME: instead, take a configured solver?
    def _configure(self, model, bounds, stop=None, **kwds):
        """configure ensemble solver from objective and other solver options"""
        monitor = kwds.get('monitor', None)
        evalmon = kwds.get('evalmon', None)
        penalty = kwds.get('penalty', None)
        constraints = kwds.get('constraints', None)
        tight = kwds.get('tightrange', None)
        clip = kwds.get('cliprange', None)

        from mystic.monitors import Monitor
        # configure monitor
        monitor = Monitor() if monitor is None else monitor

        # get dimensions
        ndim = len(bounds)
        _min, _max = zip(*bounds)

        # configure ensemble solver
        self.solver = self.sprayer(ndim, self.npts)
        self.solver.SetNestedSolver(self.seeker)
        self.solver.SetMapper(self.map)
        self.solver.SetObjective(model)
        self.solver.SetConstraints(constraints)
        self.solver.SetPenalty(penalty)
        if stop: self.solver.SetTermination(stop)
        if evalmon is not None: self.solver.SetEvaluationMonitor(evalmon)
        if monitor is not None: self.solver.SetGenerationMonitor(monitor) #NOTE: may be xy,-z
        self.solver.SetStrictRanges(min=_min, max=_max, tight=tight, clip=clip)
        return

    def _solve(self, id=None, disp=None):
        """run the solver (i.e. search for the minima)"""
        from copy import deepcopy as _copy
        solver = _copy(self.solver) #FIXME: python2.6
        # configure solver
        solver.id = id
        model = solver._cost[1] #FIXME: HACK b/c Solve(model) is required
        # solve
        disp = self.disp if disp is None else disp
#       import time
#       start = time.time()
        solver.Solve(model, disp=disp)
#       print("TOOK: %s" % (time.time() - start))
        return solver

    def _search(self, sid): #FIXME: load and leverage archived evaluations
        """run the solver, store the trajectory, and cache to the archive"""
        solver = self._solve(sid, self.disp)
        if self.traj: self._allSolvers.append(solver)
        sid += len(solver._allSolvers)
#       self._print(solver, tol=self.tol)
        size = self.memsize
        # write to evaluation cache
        memo = self._memoize(solver, tol=None, all=True, size=size).info()
        # write to trajectory archive (extrema only)
        info = self._memoize(solver, tol=self.memtol).info()
        if self.disp: print(info)
        size = info.size
        return sid, size

    def UseTrajectories(self, traj=True):
        """save all sprayers, thus save all trajectories
        """
        self.traj = bool(traj)
        return

    def Verbose(self, disp=True):
        """be verbose
        """
        self.disp = bool(disp)
        return

    #FIXME: instead, take a configured solver?
    def Search(self, model, bounds, stop=None, traj=None, disp=None, **kwds):
        """use an ensemble of optimizers to search for all minima

        Inputs:
          model - function z=f(x) to be used as the objective of the Searcher
          bounds - tuple of floats (min,max), bounds on the search region
          stop - termination condition
          traj - klepto.archive to store sampled points
          disp - if True, be verbose
          monitor - mystic.monitor instance to store parameter trajectories
          evalmon - mystic.monitor instance to store parameter evaluations
          penalty - mystic.penalty instance of the form y' = k*p(x)
          constraints - mystic.constraints instance of the form x' = c(x)
          tightrange - if True, apply bounds concurrent with other constraints
          cliprange - if True, bounding constraints will clip exterior values
        """
        self.traj = self.traj if traj is None else traj
        self.disp = self.disp if disp is None else disp
        self._configure(model, bounds, stop, **kwds)
        sid = 0  # keep track of which solver is which across multiple runs
        run = -1
        while run < self.repeat: # stop after repeat 'runs'
            #print('run: {}'.format(run))
            count = 0 if self.retry else -1 
            while self.retry > count: # stop after retry consecutive no new hits
                #print('count: {}'.format(count))
                _size = -1
                size = osize = len(self.cache) #XXX: 'size' or 'len(vals)'?
                while size > _size: # stop if no new hits
                    #print('size: {}, _: {}'.format(size,_size))
                    _size = size
                    sid, size = self._search(sid) # uses self.traj and self.disp
                if size == osize: count = count + 1
                else: count = 0
            run = run + 1

        #NOTE: traj & disp are sticky
        return

    def Reset(self, cache=None, inv=None):
        """clear the trajectory cache of sampled points

        Input:
          cache - the trajectory cache(s)
          inv - if True, reset the cache for the inverse of the objective
        """
        if cache is None: self.cache.clear() #XXX: clear the archive?
        self.cache = self.cache if cache is None else cache
        [self._allSolvers.pop() for i in range(len(self._allSolvers))]
        if inv is not None: self._inv = inv

    def Values(self, unique=False, all=False):
        """return the sequence of stored response surface outputs

        Input:
          unique: if True, only return unique values
          all: if True, return all sampled values (not just trajectory values)

        Output:
          a list of stored response surface outputs
        """
        archive = self.archive if all else self.cache
        vals = getattr(archive, 'itervalues', archive.values)()
        new = set()
        return [v for v in vals if v not in new and not new.add(v)] if unique else list(vals)

    def Coordinates(self, unique=False, all=False):
        """return the sequence of stored model parameter input values

        Input:
          unique: if True, only return unique values
          all: if True, return all sampled inputs (not just trajectory inputs)

        Output:
          a list of parameter trajectories
        """
        archive = self.archive if all else self.cache
        keys = getattr(archive, 'iterkeys', archive.keys)()
        new = set()
        return [k for k in keys if k not in new and not new.add(k)] if unique else list(keys)

    def Minima(self, tol=None): #XXX: unique?
        """return a dict of (coordinates,values) of all discovered minima

        Input:
          tol: tolerance within which to consider a point a minima

        Output:
          a dict of (coordinates,values) of all discovered minima
        """
        if tol is None: tol=self.tol
        data = self.cache
        _min = max if self._inv else min
        _min = _min(getattr(data, 'itervalues', data.values)())
        return dict((k,v) for (k,v) in getattr(data, 'iteritems', data.items)() if round(v, tol) == round(_min, tol))

    def _summarize(self):
        """provide a summary of the search results"""
        import sys
        if (sys.hexversion >= 0x30000f0):
            from builtins import min as _min
        else:
            from __builtin__ import min as _min
        del sys
        #NOTE: len(size) = # of dirs; len(vals) = # unique dirs
        keys = self.Coordinates()
        vals = self.Values()
        mins = self.Minima()
        name = 'max' if self._inv else 'min'
        min = max if self._inv else _min
        # print the minimum and number of times the minimum was found
        print("%s: %s (count=%s)" % (name, min(mins.values()), len(mins)))
        # print number of minima found, number of unique minima, archive size
        print("pts: %s (values=%s, size=%s)" % (len(set(keys)), len(set(vals)), len(keys)))
        #_i = max(dict(i).values())
        return

    def Trajectories(self, all=False):
        """return tuple (iteration, coordinates, cost) of all trajectories
        """
        mon = '_evalmon' if all else '_stepmon'
        from mystic.munge import read_trajectories
        if not self.traj:
            try: #NOTE: FRAGILE (if absolute path is not used)
                filename = getattr(self.solver, mon)._filename
                step, param, cost = read_trajectories(filename)
            except AttributeError:
                msg = "a LoggingMonitor or UseTrajectories is required"
                raise RuntimeError(msg)
        else:
            step = []; cost = []; param = [];
            for sprayer in self._allSolvers:  #XXX: slow? better thread.map?
                for seeker in sprayer._allSolvers:
                    values = read_trajectories(getattr(seeker,mon))
                    step.extend(values[0])
                    param.extend(values[1])
                    cost.extend(values[2])
        #XXX: (not from archive, so) if self._inv: use -cost
        return step, param, cost

    def Samples(self, all=False):
        """return array of (coordinates, cost) for all trajectories
        """
        import numpy as np
        xy,xy,z = self.Trajectories(all=all)
        xy = np.vstack((np.array(xy).T,z))
        if self._inv: xy[-1,:] = -xy[-1]
        return xy #NOTE: actually xyz

    pass



# EOF
