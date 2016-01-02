2016sr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mystic import log_reader

__doc__ = log_reader.__doc__

if __name__ == '__main__':

    import sys

    log_reader(sys.argv[1:])


# EOF
