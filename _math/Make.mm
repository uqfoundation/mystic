# -*- Makefile -*-

PROJECT = mystic
PACKAGE = math

all: export

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
    __init__.py \
    approx.py \
    grid.py \
    poly.py \
    integrate.py \
    samples.py \
    stats.py \
    measures.py \
    dirac_measure.py \

export:: export-package-python-modules


# End of file

