from distutils.core import setup
from distutils.extension import Extension
import numpy as np
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
    #ImportError("Cython not installed. Please install it manually before "
    #                  "running this setup using 'pip install cython'")
else:
    use_cython = True
cmdclass = { }
if use_cython:
    cmdclass = {'build_ext': build_ext}
    ext_modules = [Extension("syconn.ray_casting",
                             ["syconn/ray_casting/ray_casting_radius.pyx"],
                             include_dirs=[np.get_include()])]
else:
    ext_modules = [
        Extension("syconn.ray_casting", ["syconn/raycasting/ray_casting.c"])]


config = {
    'description': 'Analysis pipeline for EM raw data based on deep and '
                   'supervised learning to extract high level biological'
                   'features and connectivity. ',
    'author': 'Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'My email.',
    'version': '0.1',
    'install_requires': ['nose', 'matplotlib', 'sklearn', 'networkx', 'numpy',
                         'scipy', 'seaborn', 'knossos_python_tools'],
    'packages': ['syconn'],
    'scripts': [],  'cmdclass': cmdclass, 'ext_modules': ext_modules,
    'name': 'SyConn',
    'dependency_links': ['https://github.com/knossos-project/knossos_python_'
                         'tools/tarball/master#egg=knossos_python_tools'],
    'include_dirs': [np.get_include()]
}
setup(**config)
# compile ray casting c function
