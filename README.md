# SyConn v2
Synaptic connectivity inference toolkit developed at Max-Planck-Institute of Neurobiology, Munich <br />
Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld <br />

Refactored (still an early stage construction) version of SyConn for automated synaptic connectivity inference based on dense EM segmentation data. 
For the first version see branch 'dorkenwald2017nm'. 

Version 2 currently features:
- introduction of supervoxel and agglomerated supervoxel classes
- added support for (sub-) cellular compartment (spines, axon/dendrite/soma) and cell type classification with [skeleton](https://www.nature.com/articles/nmeth.4206)- and [multiview-based](https://www.biorxiv.org/content/early/2018/07/06/364034) approaches
- cell organelle prediction, extraction and mesh generation
- [glia identification and splitting](https://www.biorxiv.org/content/early/2018/07/06/364034)
- generation of connectivity matrix


## System Requirements & Installation

* Python 3.5
* The whole pipeline was designed and tested on Linux systems (CentOS, Arch)
* SyConn is based on the packages `elektronn <http://elektronn.org>`_, `knossos-utils <https://github.com/knossos-project/knossos_utils>`_
* `KNOSSOS <http://knossostool.org/>`_ is used for visualization and annotation of 3D EM data sets.
* [VIGRA](https://ukoethe.github.io/vigra/), e.g. ``conda install -c ukoethe vigra``
* osmesa, e.g.: ``conda install -c menpo osmesa``

You can install SyConn using  ``git`` and  ``pip``:

    git clone https://github.com/SyConn
    cd SyConn
    pip install -r requirements.txt
    pip install .

## Documentation

For documentation see [here](docs/doc.md)