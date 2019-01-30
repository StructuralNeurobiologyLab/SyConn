from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules =cythonize(Extension("cs_extraction_steps", sources=["cs_extraction_steps.pyx"]))
)