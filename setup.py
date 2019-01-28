from setuptools import setup, find_packages
import os
import sys
from unittest.mock import MagicMock
from distutils.core import setup
import numpy
from Cython.Build import cythonize
readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    # m2r may not be installed in user environment
    with open(readme_file) as f:
        readme = f.read()


# not possible to install C-library libsnappy-dev, TODO: this did not work with readthedocs
# (https://docs.readthedocs.io/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules)
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = ['snappy', 'python-snappy']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

config = {
    'description': 'Analysis pipeline for EM raw data based on deep and '
                   'supervised learning to extract high level biological'
                   'features and connectivity. ',
    'author': 'Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld',
    'url': 'https://structuralneurobiologylab.github.io/SyConn/',
    'download_url': 'https://github.com/StructuralNeurobiologyLab/SyConn.git',
    'author_email': 'pschubert@neuro.mpg.de',
    'version': '0.2',
    'license': 'GPL-2.0',
    'install_requires': ['knossos_utils>=0.1', 'ELEKTRONN2', 'matplotlib',
                         'numpy==1.16', 'scipy', 'lz4', 'h5py', 'networkx', 'ipython <7.0.0',
                         'configobj', 'fasteners', 'flask', 'coloredlogs',
                         'opencv-python', 'pyopengl', 'scikit-learn==0.19.1',
                         'scikit-image==0.14.2', 'm2r', 'plyfile', 'vtkInterface',
                         'sphinx-autodoc-typehints', 'openmesh', 'pytest',
                         'pytest-runner', 'prompt-toolkit<2.0', 'numba==0.41.0',
                         'llvmlite==0.26.0', 'elektronn3'],  # numba/llvmluite requirements due to https://github.com/numba/numba/issues/3666 in @jit compilation of 'id2rgb_array_contiguous' (in multiviews.py)
    'name': 'SyConn',
    'dependency_links': ['https://github.com/knossos-project/knossos_utils'
                         '/tarball/dev#egg=knossos_utils',
                         'https://github.com/ELEKTRONN/ELEKTRONN2'
                         '/tarball/master#egg=ELEKTRONN2',
                         'https://github.com/ELEKTRONN/elektronn3/'
                         'tarball/phil/#egg=elektronn3'],
    'packages': find_packages(exclude=['scripts']), 'long_description': readme,
    'setup_requires': ["pytest-runner", "cython>=0.23"], 'tests_require': ["pytest", ],
    # this will compile all files within directories in syconn/
    # 'ext_modules': cythonize(["syconn/*/*.pyx"], include_path=[numpy.get_include()]),
}

setup(**config)
