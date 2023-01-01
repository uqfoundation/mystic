#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
print the size of the database of sampled data from xrd_design*.py
"""

import os
import numpy as np
import mystic.cache.archive as mca
archives = ('ave.db', 'min.db', 'max.db')

for archive in archives:
    a = None
    if os.path.exists(archive):
        try:
            a = mca.read(archive, type=mca.file_archive)
            print('len(%s): %s' % (archive, len(a)))
            #a = np.array(list(a.values()))
            #print('  min=%s, ave=%s, max=%s' % (a.min(), a.mean(), a.max()))
        except Exception:
            pass
