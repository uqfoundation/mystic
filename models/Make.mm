# -*- Makefile -*-

PROJECT = mystic
PACKAGE = models

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
    dejong.py \
    griewangk.py \
    zimmermann.py \
    corana.py \
    fosc3d.py \
    wavy.py \
    poly.py \
    mogi.py \
    br8.py \
    lorentzian.py \
    circle.py \

export:: export-package-python-modules


# End of file

