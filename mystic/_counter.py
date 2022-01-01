#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# Inspired by https://stackoverflow.com/a/21681534
"""
a multiprocessing-friendly counter
"""

class Counter(object):
    def __init__(self, value=0):
        """a counter where the value can be shared by multiple processes
        """
        try: #XXX: is multiprocess ever useful here vs multiprocessing?
            import multiprocess as mp
        except ImportError:
            import multiprocessing as mp
        self.val = mp.Value('i', value)

    def __repr__(self):
        return "%s(value=%s)" % (self.__class__.__name__, self.value)

    def increment(self, n=1):
        """increase the current value by n

    Inputs:
        n [int]: amount to increase the current value

    Returns:
        the incremented counter
        """
        with self.val.get_lock():
            self.val.value += n
        return self

    def count(self, n=1):
        """return the current value, then increase by n

    Inputs:
        n [int]: amount to increase the current value

    Returns:
        the current value
        """
        return self.increment(n).value - n

    @property
    def value(self):
        "the current value of the counter"
        return self.val.value
