from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
    ext_modules =cythonize(Extension("in_bounding_box", sources=["in_bounding_box.pyx"]))
)