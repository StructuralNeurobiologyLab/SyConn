from setuptools import find_packages, setup
from distutils.extension import Extension
import os
import glob

# catch ImportError during the readthedocs build.
try:
    from Cython.Build import cythonize
    ext_modules = [Extension("*", [fname],
                             extra_compile_args=["-std=c++11"], language='c++')
                   for fname in glob.glob('syconn/*/*.pyx', recursive=True)]
    cython_out = cythonize(ext_modules, compiler_directives={
                           'language_level': 3, 'boundscheck': False,
                           'wraparound': False, 'initializedcheck': False,
                           'cdivision': False, 'overflowcheck': True})
except ImportError as e:
    print("WARNING: Could not build cython modules. {}".format(e))
    cython_out = None

VERSION = '0.3'


def read_readme():
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
    with open(readme_file) as f:
        readme = f.read()
    return readme


def write_version_py(filename='syconn/version.py'):
    content = """# THIS FILE IS GENERATED FROM SYCONN SETUP.PY
#
version = '%(version)s'
"""
    with open(filename, 'w') as f:
        f.write(content % {'version': VERSION})


write_version_py()


setup(
    name='SyConn',
    version=VERSION,
    description='Analysis pipeline for EM raw data based on deep and '
                'supervised learning to extract high level biological'
                'features and connectivity.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://structuralneurobiologylab.github.io/SyConn/',
    download_url='https://github.com/StructuralNeurobiologyLab/SyConn.git',
    author='Philipp Schubert et al.',
    author_email='pschubert@neuro.mpg.de',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Connectomics :: Analysis Tools',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    platforms=["Linux", ],
    keywords='connectomics machinelearning imageprocessing',
    packages=find_packages(exclude=['scripts', 'tests']),
    python_requires='>=3.6, <4',
    package_data={'syconn': ['handler/config.yml']},
    include_package_data=True,
    tests_require=['pytest', 'pytest-cov', 'pytest-runner'],
    ext_modules=cython_out,
    entry_points={
        'console_scripts': [
            'syconn.server=syconn.analysis.server:main'
        ],
    },
)
