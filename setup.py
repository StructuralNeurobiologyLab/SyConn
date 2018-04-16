from setuptools import setup, find_packages
# TODO: add vigra and opencv (cv2)
config = {
    'description': 'Analysis pipeline for EM raw data based on deep and '
                   'supervised learning to extract high level biological'
                   'features and connectivity. ',
    'author': 'Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld',
    'url': 'syconn.org',
    'download_url': 'https://github.com/StructuralNeurobiologyLab/SyConn.git',
    'author_email': '',
    'version': '0.2',
    'install_requires': ['knossos_utils', 'ELEKTRONN2', 'matplotlib',
                         'numpy', 'scipy', 'lz4', 'h5py', 'networkx',
                         'configobj', 'fasteners', 'flask'],
    'name': 'SyConn',
    'dependency_links': ['https://github.com/knossos-project/knossos_utils'
                         '/tarball/master#egg=knossos_utils',
                         'https://github.com/ELEKTRONN/ELEKTRONN2'
                         '/tarball/master#egg=ELEKTRONN2'],
    'packages': find_packages(exclude=['scripts']),
}
setup(**config)