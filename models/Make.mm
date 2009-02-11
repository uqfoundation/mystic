# -*- Makefile -*-

PROJECT = mystic
PACKAGE = models

all: export

update: $(BUILD_DIRS)

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#
# export

EXPORT_PYTHON_MODULES = \
    __init__.py \
    chebyshev8.py \
    poly2.py \
    mogi.py \
#   br8.py \
#   circle.py \
#   corana.py \

export:: export-package-python-modules


# End of file

