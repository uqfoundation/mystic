#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import mystic.symbolic as ms


if __name__ == '__main__':

  eqn = '''
  abs(1 - x0) - abs(1 + abs(x0) - abs(x1 + x2)) > 4
  x0**.5 = abs(x1 + x2)
  abs(x0) - x1 > 0
  x2**2 > 1
  (x2**2 - 1)**.5 + x1 > 0
  '''

  res = ms.absval(eqn, all=True)
  assert len(res) == 64
  assert not any('abs(' in r for r in res)
  #print(res[0])
  #print('')

  eq_ = '''
  abs(x0) - x1 > 0
  x2/(x0 - x1) > 1
  '''

  res = ms.simplify(eq_, all=True)
  #print(res[0])
  #print('')
  res0 = ms.simplify(eq_, all=False)
  #print(res0)
  #print('')
  assert res0 in res
  assert len(res) == 3

  _eq = '''
  x0**.5 == abs(x1 + x2)
  abs(x0) - x1 > 0
  x2**2 > 1
  '''

  res = ms.simplify(_eq, all=True)
  import mystic.constraints as mc
  c = ms.generate_constraint(ms.generate_solvers(res), join=mc.or_)
  kx = c([1,2,3])
  #print(kx)
  x0,x1,x2 = kx
  assert x0**.5 == abs(x1 + x2)
  assert abs(x0) - x1 > 0
  assert x2**2 > 1

