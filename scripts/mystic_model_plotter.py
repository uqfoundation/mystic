#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Jean-Christophe Fillion-Robin (jchris.fillionr @kitware.com)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mystic.model_plotter import model_plotter

__doc__ = model_plotter.__doc__

if __name__ == '__main__':

    import sys
    model_plotter(cmdargs=sys.argv[1:])


# EOF
