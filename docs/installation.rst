.. _installation:

************
Installation
************

Setup
=====

We recommend installing the latest Anaconda release. Then set up an environment:

    conda create -n py35 python=3.5 anaconda
    source activate py35

Then git clone SyConn, install all requirements and finally syconn:

    git clone https://github.com/StructuralNeurobiologyLab/SyConn.git`
    cd SyConn
    pip install -r requirements.txt
    conda install vigra -c ukoethe
    conda install mesa -c menpo
    conda install osmesa -c menpo
    pip install .

Or alternatively with the developer flag:

    pip install -e .
