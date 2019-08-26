from setuptools import find_packages, setup
from distutils.extension import Extension
import numpy
import os
import glob

# catch ImportError during the readthedocs build.
try:
    from Cython.Build import cythonize
    setup_requires = ["pytest-runner", "cython>=0.23"]
    ext_modules = [Extension("*", [fname],
                             extra_compile_args=["-std=c++11"], language='c++')
                   for fname in glob.glob('syconn/*/*.pyx', recursive=True)]
    cython_out = cythonize(ext_modules, compiler_directives={
                           'language_level': 3, 'boundscheck': False,
                           'wraparound': False, 'initializedcheck': False,
                           'cdivision': False, 'overflowcheck': True})
except ImportError as e:
    print("WARNING: Could not build cython modules. {}".format(e))
    setup_requires = ["pytest-runner"]
    cython_out = None
readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
with open(readme_file) as f:
    readme = f.read()

config = {
    'description': 'Analysis pipeline for EM raw data based on deep and '
                   'supervised learning to extract high level biological'
                   'features and connectivity. ',
    'author': 'Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld',
    'url': 'https://structuralneurobiologylab.github.io/SyConn/',
    'download_url': 'https://github.com/StructuralNeurobiologyLab/SyConn.git',
    'author_email': 'pschubert@neuro.mpg.de',
    'version': '0.2',
    'license': 'GPLv2',
    'install_requires': [
                         'numpy==1.16.4', 'scipy', 'lz4', 'h5py', 'networkx', 'ipython',
                         'configobj', 'fasteners', 'flask', 'coloredlogs',
                         'opencv-python', 'pyopengl', 'scikit-learn>=0.21.3',
                         'scikit-image', 'plyfile', 'termcolor',
                         'pytest', 'tqdm', 'dill', 'zmesh', 'seaborn',
                         'pytest-runner', 'prompt-toolkit', 'numba==0.45.0',
                         'matplotlib', 'vtki', 'joblib', 'yaml'],
    'name': 'SyConn',
    'dependency_links': ['https://github.com/knossos-project/knossos_utils'
                         '/tarball/dev#egg=knossos_utils',
                         'https://github.com/ELEKTRONN/elektronn3/'
                         'tarball/phil/#egg=elektronn3'],
    'packages': find_packages(exclude=['scripts']), 'long_description': readme,
    'setup_requires': setup_requires, 'tests_require': ["pytest", ],
    'ext_modules': cython_out,
    'include_dirs': [numpy.get_include(), ],
}

setup(**config)
