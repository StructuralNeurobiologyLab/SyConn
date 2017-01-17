try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
from distutils.extension import Extension
import numpy as np
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    raise ImportError('Cython not find, please install manually beforehand.')
cmdclass = {'build_ext': build_ext}
ext_modules = [Extension("syconn.ray_casting.ray_casting_radius",
                         ["syconn/ray_casting/ray_casting_radius.pyx"],
                         include_dirs=[np.get_include()], language="c++")]
config = {
    'description': 'Analysis pipeline for EM raw data based on deep and '
                   'supervised learning to extract high level biological'
                   'features and connectivity. ',
    'author': 'Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld',
    'url': 'syconn.org',
    'download_url': 'https://github.com/pschubert/SyConn.git',
    'author_email': '',
    'version': '0.1',
    'install_requires': ['cython', 'knossos_utils', 'nose',
                         'matplotlib', 'scikit-learn==0.17.1', 'networkx', 'numpy',
                         'scipy', 'seaborn'],
    'scripts': [],  'cmdclass': cmdclass, 'ext_modules': ext_modules,
    'name': 'SyConn',
    'dependency_links': ['https://github.com/knossos-project/knossos_utils'
                         '/tarball/master#egg=knossos_utils'],
    'include_dirs': [np.get_include()],
    'package_data': {'syconn': ['ray_casting/*.so']},
    'packages': find_packages(), ' include_package_data': True,
}
setup(**config)
# compile ray casting c function
