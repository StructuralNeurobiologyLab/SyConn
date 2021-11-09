.. _installation:

************
Installation
************

Setup
=====

Before you can set up SyConn, ensure that the
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_
package manager is installed on your system.
Then you can install SyConn and all of its dependencies into a new conda
`environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_
named "syco" by running::

    git clone https://github.com/StructuralNeurobiologyLab/SyConn
    cd SyConn
    conda env create -f environment.yml -n syco python=3.7
    conda activate syco
    pip install --no-deps -v -e .


The last command will install SyConn in
`editable <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_
mode, which is useful for development on SyConn itself. If you want to install
it as a regular read-only package instead, replace the last command with::

    pip install --no-deps -v .


To update the environment, e.g. if the environment file changed, use::

    conda env update --name syco --file environment.yml --prune


Troubleshooting
===============

In case you encounter the following import error with vigra::

    ImportError: libemon.so.1.3.1: cannot open shared object file: No such file or directory


try to install lemon from scratch or via conda::

    conda install lemon=1.3.1=he9d42e9_3 -c conda-forge


If glibc >= 2.27 is available on you compute environment you can relax the open3d and pytorch versio. restriction.
