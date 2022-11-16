from distutils.core import setup
from Cython.Build import cythonize

exts = (cythonize("src/algorithms_cython/*.pyx", annotate=True))
setup(ext_modules=exts)
