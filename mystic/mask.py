#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Lan Huong Nguyen (lanhuong @stanford)
# Copyright (c) 2012-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

import mystic.termination as _term

def get_mask(condition): #FIXME: gets None if is None *and* if no mask
    '''get mask from termination condition'''
    condition = _term.state(condition)
    if not condition: return None
    if isinstance(condition, tuple): return None
    return condition.popitem()[-1].get('mask', None)


#FIXME: get collapses by evaluating condition (not requiring already evaluated)?
def update_mask(condition, collapse, new=False):
    '''update the termination condition with the given collapse (dict)'''
    if collapse is None: return condition
    for kind,mask in collapse.iteritems():
        condition = _update_masks(condition, mask, kind, new)
    return condition


##### mask updater helpers #####
def _update_masks(condition, mask, kind='', new=False):
    '''update the termination condition with the given mask'''
    kind = kind if kind else 'Collapse'
    if isinstance(condition, tuple):# mystic.termination.When is tuple
        conditions = () #XXX: faster using list?
        for term in condition:
            if isinstance(term, tuple) or term.__doc__.startswith(kind):
                term = _update_masks(term, mask, kind, new)
            conditions += (term,)
        return type(condition)(*conditions)
    # else: get dict used to build termination condition
    if new: return _replace_mask(condition, mask)
    return _extend_mask(condition, mask)

def _replace_mask(condition, mask):
    '''replace the mask in the termination condition with the given mask'''
    kwds = _term.state(condition).popitem()[-1]
    if kwds.has_key('mask'): kwds['mask'] = mask
    return _term.type(condition)(**kwds)

def _extend_mask(condition, mask):
    '''extend the mask in the termination condition with the given mask'''
    if mask is None:
        return condition
    kwds = _term.state(condition).popitem()[-1]
    # short-circiut: if no kwds['mask'], then abort
    if not kwds.has_key('mask'): return _term.type(condition)(**kwds)
    # extend current mask
    _mask = kwds['mask']
    if not _mask: # mask is None, {}, set(), or ()
        kwds['mask'] = mask
    elif type(_mask) is set: # assumes mask is set
        kwds['mask'].update(mask)
    elif type(_mask) is dict: # assumes mask is dict
        for k,v in mask.iteritems():
            _mask.setdefault(k,v).update(v)
        kwds['mask'] = _mask
    else: # assumes mask is tuple (or list)
        kwds['mask'] = type(_mask)(_mask[i]+mask[i] for i in range(2))
    return _term.type(condition)(**kwds)


##### deprecated #####
#NOTE: weakness is that it replaces all masks of given type
def update_weight_masks(condition, mask, new=False):
    '''update all weight masks in the given termination condition'''
    return _update_masks(condition, mask, 'CollapseWeight', new)

#NOTE: weakness is that it replaces all masks of given type
def update_position_masks(condition, mask, new=False):
    '''update all position masks in the given termination condition'''
    return _update_masks(condition, mask, 'CollapsePosition', new)



# EOF
