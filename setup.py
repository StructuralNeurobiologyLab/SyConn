from setuptools import setup, find_packages
import os
# TODO: add vigra and opencv (cv2)
readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    # m2r may not be installed in user environment
    with open(readme_file) as f:
        readme = f.read()

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
                         'numpy', 'scipy', 'lz4', 'h5py', 'networkx', 'numba',
                         'configobj', 'fasteners', 'flask', 'coloredlogs',
                         'opencv-python', 'pyopengl', 'scikit-learn', ],
    'name': 'SyConn',
    'dependency_links': ['https://github.com/knossos-project/knossos_utils'
                         '/tarball/master#egg=knossos_utils',
                         'https://github.com/ELEKTRONN/ELEKTRONN2'
                         '/tarball/master#egg=ELEKTRONN2'],
    'packages': find_packages(exclude=['scripts']),
    'long_description': readme,
}
setup(**config)
