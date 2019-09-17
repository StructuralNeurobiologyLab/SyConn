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
named "pysy" by running::

    git clone https://github.com/StructuralNeurobiologyLab/SyConn
    cd SyConn
    conda env create -f environment.yml -n pysy
    conda activate pysy
    pip install -v -e --no-deps .


The last command will install SyConn in
`editable <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_
mode, which is useful for development on SyConn itself. If you want to install
it as a regular read-only package instead, replace the last command with::

    pip install -v --no-deps .
