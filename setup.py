try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import numpy as np

ext_modules = []
cmdclass = {}

config = {
    'description': 'Analysis pipeline for EM raw data based on deep and '
                   'supervised learning to extract high level biological'
                   'features and connectivity. ',
    'author': 'Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld',
    'url': 'syconn.org',
    'download_url': 'https://github.com/StructuralNeurobiologyLab/SyConn.git',
    'author_email': '',
    'version': '0.2',
    'install_requires': ['knossos_utils', 'matplotlib', 'numpy', 'scipy',
                         'lz4', 'h5py', 'networkx', 'configobj', 'fasteners',
                         'flask'],
    'scripts': [],  'cmdclass': cmdclass, 'ext_modules': ext_modules,
    'name': 'SyConn',
    'dependency_links': ['https://github.com/knossos-project/knossos_utils'
                         '/tarball/master#egg=knossos_utils'],
    'include_dirs': [np.get_include()],
    'package_data': {},
    'packages': find_packages(), ' include_package_data': True,
}
setup(**config)
