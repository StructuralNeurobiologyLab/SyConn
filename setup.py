from setuptools import find_packages, setup
from distutils.extension import Extension
import os
import glob

# catch ImportError during the readthedocs build.
# TODO: pytest stuff can probably be removed from `setup_requires`
try:
    from Cython.Build import cythonize
    setup_requires = ['pytest', 'pytest-cov', "pytest-runner", 'pytest-xdist',
                      "cython>=0.23"]
    ext_modules = [Extension("*", [fname],
                             extra_compile_args=["-std=c++11"], language='c++')
                   for fname in glob.glob('syconn/*/*.pyx', recursive=True)]
    cython_out = cythonize(ext_modules, compiler_directives={
                           'language_level': 3, 'boundscheck': False,
                           'wraparound': False, 'initializedcheck': False,
                           'cdivision': False, 'overflowcheck': True})
except ImportError as e:
    print("WARNING: Could not build cython modules. {}".format(e))
    setup_requires = ['pytest', 'pytest-cov', "pytest-runner", 'pytest-xdist']
    cython_out = None
readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
with open(readme_file) as f:
    readme = f.read()

setup(
    name='SyConn',
    version='0.2',
    description='Analysis pipeline for EM raw data based on deep and '
                'supervised learning to extract high level biological'
                'features and connectivity.',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://structuralneurobiologylab.github.io/SyConn/',
    download_url='https://github.com/StructuralNeurobiologyLab/SyConn.git',
    author='Philipp Schubert, Joergen Kornfeld',
    author_email='pschubert@neuro.mpg.de',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Neuroscientists',
        'Topic :: Connectomics :: Analysis Tools',
        'License :: OSI Approved :: GPL-2.0 License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='machinelearning imageprocessing connectomics',
    packages=find_packages(exclude=['scripts']),
    python_requires='>=3.6, <4',
<<<<<<< HEAD
    install_requires=['numpy==1.16.4', 'scipy', 'lz4', 'h5py', 'networkx',
                      'fasteners', 'flask', 'coloredlogs', 'opencv-python',
                      'pyopengl', 'scikit-learn>=0.21.3', 'scikit-image',
                      'plyfile', 'termcolor', 'dill', 'tqdm', 'zmesh',
                      'seaborn', 'pytest-runner', 'prompt-toolkit',
                      'numba==0.45.0', 'matplotlib', 'vtki', 'joblib',
                      'pyyaml', 'cython'],
    setup_requires=setup_requires, tests_require=['pytest', 'pytest-cov'],
=======
    setup_requires=setup_requires,
    package_data={'syconn': ['handler/config.yml']},
    include_package_data=True,
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
>>>>>>> d7b8a8b9652050cca79384c362fc1099bced807c
    ext_modules=cython_out,
    entry_points={
        'console_scripts': [
        ],
    },
)
