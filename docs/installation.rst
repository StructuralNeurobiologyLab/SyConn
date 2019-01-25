.. _installation:

************
Installation
************

Setup
=====

We recommend installing the latest Anaconda release. Then set up an environment:

    conda create -n py36 python=3.6 anaconda

    source activate py36

Then install all prerequisites and finally git clone and install syconn:

    conda install vigra -c conda-forge

    conda install mesa -c menpo

    conda install osmesa -c menpo

    conda install freeglut

    conda install pyopengl

    git clone https://github.com/StructuralNeurobiologyLab/SyConn.git

    cd SyConn

    pip install -r requirements.txt

    pip install .

Or alternatively with the developer flag:

    pip install -e .


In order to use elektronn3 models, python>=3.6 is required (when installing elektronn3 checkout
 branch `phil`; this will be updated soon to work with the `master` branch).

In the case that there are problems with snappy/python-snappy remove previous installations and
install them via conda:

    conda uninstall snappy

    conda install snappy

    conda install python-snappy


