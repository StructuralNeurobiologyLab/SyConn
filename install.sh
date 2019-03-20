#!/bin/bash
# cd to the SyConn directory an run this file via 'sh install.sh'

echo y | conda install cmake
echo y | conda install vigra -c conda-forge
echo y | conda install -c conda-forge opencv
echo y | conda install mesa -c menpo
echo y | conda install osmesa -c menpo
echo y | conda install freeglut
echo y | conda install pyopengl
echo y | conda install snappy
echo y | conda install python-snappy

pip install -r requirements.txt
pip install -e .