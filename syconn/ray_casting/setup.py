from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules=cythonize("ray_casting_radius.pyx")
)
