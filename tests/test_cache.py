#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from klepto.archives import dict_archive
from mystic.cache import archive 

d = archive.read('test', type=dict_archive)
assert len(d) == 0

archive.write(d, dict(a=1, b=2, c=3))
assert len(d) == 3

