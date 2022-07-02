#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Jean-Christophe Fillion-Robin (jchris.fillionr @kitware.com)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic import model_plotter

__doc__ = model_plotter.__doc__

if __name__ == '__main__':

    import sys

    model_plotter(sys.argv[1:])


# EOF
