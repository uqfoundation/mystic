#!/usr/bin/env python
#
# Author: Lan Huong Nguyen (lanhuong @stanford)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2012-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

import mystic.collapse as ct
import mystic.mask as ma
import mystic.termination as mt
import mystic.constraints as cn
import mystic.tools as to

##### collapse updater #####
def collapse(solver, verbose=False):
    """if solver has terminated by collapse, apply the collapse"""
    collapses = ct.collapsed(solver.Terminated(info=True)) or dict()
    if collapses:
        if verbose:
            print "#", solver._stepmon._step-1, "::", \
                  solver.bestEnergy, "@\n#", list(solver.bestSolution)

        # get collapse conditions  #XXX: efficient? 4x loops over collapses
        state = mt.state(solver._termination)
        npts = getattr(solver._stepmon, '_npts', None)  #XXX: default?
        conditions = [cn.impose_at(*to.select_params(solver,collapses[k])) if state[k].get('target') is None else cn.impose_at(collapses[k],state[k].get('target')) for k in collapses if k.startswith('CollapseAt')]
        conditions += [cn.impose_as(collapses[k],state[k].get('offset')) for k in collapses if k.startswith('CollapseAs')]
        # get measure collapse conditions
        if npts: #XXX: faster/better if comes first or last?
            conditions += [cn.impose_measure( npts, [collapses[k] for k in collapses if k.startswith('CollapsePosition')], [collapses[k] for k in collapses if k.startswith('CollapseWeight')] )]

        # update termination and constraints in solver
        constraints = to.chain(*conditions)(solver._constraints)
        termination = ma.update_mask(solver._termination, collapses)
        solver.SetConstraints(constraints)
        solver.SetTermination(termination)
        #print mt.state(solver._termination).keys()
    return collapses


# EOF
