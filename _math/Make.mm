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

export:: export-package-python-modules


# End of file

