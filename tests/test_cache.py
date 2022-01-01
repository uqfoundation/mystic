#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import mystic.cache as mc
import mystic.models as mm

# basic interaction with an archive
d = mc.archive.read('test', type=mc.archive.dict_archive)
assert len(d) == 0

mc.archive.write(d, dict(a=1, b=2, c=3))
assert len(d) == 3

# basic pattern to cache an objective
d = mc.archive.read('rosen', type=mc.archive.dict_archive)
model = mc.cached(archive=d)(mm.rosen)
model([1,2,1])
model([1,1,1])
c = model.__cache__()
assert len(c) == 2

model.__inverse__([1,2,3]) == -model([1,2,3])
assert len(c) == 3
