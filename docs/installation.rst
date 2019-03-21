.. _installation:

************
Installation
************

Setup
=====

We recommend installing the latest Anaconda release. Then set up the python environment:

    conda create -n pysy python=3.6 anaconda

    source activate pysy

Then install all prerequisites and finally git clone and install syconn:

    conda install cmake

    conda install vigra -c conda-forge

    conda install mesa -c menpo

    conda install osmesa -c menpo

    conda install freeglut

    conda install pyopengl

    conda install snappy

    conda install python-snappy

    git clone https://github.com/StructuralNeurobiologyLab/SyConn.git

    cd SyConn

    pip install -r requirements.txt

    pip install .

Or alternatively with the developer flag:

    pip install -e .


In the case that there are problems with snappy/python-snappy remove previous installations and
install them via conda:

    conda uninstall snappy

    conda install snappy

    conda install python-snappy
