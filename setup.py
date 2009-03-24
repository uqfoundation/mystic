#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2009  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from distutils.core import setup

setup(
    name="mystic",
    version="0.1a1",
    author="Mike McKerns, Patrick Hung"
    maintainer="Mike McKerns",
    maintainer_email="mmckerns@caltech.edu",
    license="BSD",
    platforms=["any"],
    description="Interactive Inversion Analysis Framework",
    classifiers=(
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Physics Programming"),

    packages=['mystic','mystic.models'],
    package_dir={'mystic':'mystic','mystic.models':'models'},

    scripts=[])


if __name__=='__main__':
    pass

# End of file
