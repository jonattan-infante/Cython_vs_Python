from distutils.core import setup

import numpy
from Cython.Build import cythonize

exts = (cythonize("src/algorithms_cython/*.pyx", annotate=True))
setup(ext_modules=exts,
      include_dirs=[numpy.get_include()])
