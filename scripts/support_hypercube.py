#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mystic.support import hypercube

__doc__ = hypercube.__doc__

if __name__ == '__main__':

    import sys

    hypercube(sys.argv[1:])


# EOF
