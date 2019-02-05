from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules =cythonize(Extension("block_processing_cython", sources=["block_processing_cython.pyx"]))
)