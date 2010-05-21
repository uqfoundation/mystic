#!/usr/bin/env python
#
# Michael McKerns
# mmckerns@caltech.edu

# check if easy_install is available
try:
#   import __force_distutils__ #XXX: uncomment to force use of distutills
    from setuptools import setup
    has_setuptools = True
except ImportError:
    from distutils.core import setup
    has_setuptools = False

# build the 'setup' call
setup_code = """
setup(name='mystic',
      version='0.2a1',
      description='a simple interactive inversion analysis framework',
      author = 'Mike McKerns',
      maintainer = 'Mike McKerns',
      maintainer_email = 'mmckerns@caltech.edu',
      license = 'BSD',
      platforms = ['any'],
      url = 'http://www.cacr.caltech.edu/~mmckerns',
      classifiers = ('Intended Audience :: Developers',
                     'Programming Language :: Python',
                     'Topic :: Physics Programming'),

      packages = ['mystic','mystic.models','mystic.math'],
      package_dir = {'mystic':'mystic','mystic.models':'models',
                     'mystic.math':'_math'},
"""

# add dependencies
numpy_version = '>=1.0'
matplotlib_version = '>=0.91'
if has_setuptools:
    setup_code += """
      install_requires = ('numpy%s'),
""" % numpy_version

# close 'setup' call
setup_code += """    
      zip_safe=True,
      scripts=[])
"""

# exec the 'setup' code
exec setup_code

# if dependencies are missing, print a warning
try:
    import numpy
    #import matplotlib #XXX: has issues being zip_safe
except ImportError:
    print "\n***********************************************************"
    print "WARNING: One of the following dependencies is unresolved:"
    print "    numpy %s" % numpy_version
    print "    matplotlib %s (optional)" % matplotlib_version
    print "***********************************************************\n"


if __name__=='__main__':
    pass

# end of file
