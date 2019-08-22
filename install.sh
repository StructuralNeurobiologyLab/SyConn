#!/bin/bash
# cd to the SyConn directory an run this file via 'sh install.sh'

echo y | conda install vigra -c conda-forge
echo y | conda install -c conda-forge opencv
echo y | conda install mesa -c anaconda
echo y | conda install osmesa -c menpo
echo y | conda install freeglut
echo y | conda install pyopengl
echo y | conda install snappy
echo y | conda install python-snappy
echo y | conda install numba==0.45.0 llvmlite==0.29
echo y | conda install tensorboard tensorflow
# the following torch setting seems to be more stable for new GPU/driver
echo y | conda install pytorch==1.1.0 torchvision cudatoolkit=10.0 -c pytorch


pip install -r requirements.txt
pip install -e .

