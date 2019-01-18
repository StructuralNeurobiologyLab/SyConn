.. _installation:

************
Installation
************

Setup
=====

We recommend installing the latest Anaconda release. Then set up an environment:

    conda create -n py35 python=3.5 anaconda
    source activate py35

Then install all prerequisites and finally git clone and install syconn:

    conda install vigra -c ukoethe
    conda install mesa -c menpo
    conda install osmesa -c menpo
    conda install pyopengl
    git clone https://github.com/StructuralNeurobiologyLab/SyConn.git
    cd SyConn
    pip install -r requirements.txt
    pip install .

Or alternatively with the developer flag:

    pip install -e .


In the case that there are problems with snappy/python-snappy remove previous installations and
install them via conda:

    conda uninstall snappy
    conda install python-snappy

