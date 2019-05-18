[![Documentation Status](https://readthedocs.org/projects/syconn/badge/?version=latest)](https://syconn.readthedocs.io/en/latest/?badge=latest)

# SyConn
Refactored version of SyConn for automated synaptic connectivity inference based on dense EM segmentation data. For the first version
 see below or checkout the branch [dorkenwald2017nm](https://github.com/StructuralNeurobiologyLab/SyConn/tree/dorkenwald2017nm).

Current features:
- introduction of supervoxel and agglomerated supervoxel classes
- added support for (sub-) cellular compartment (spines, axon/dendrite/soma) and cell type classification with skeleton- [\[1\]](https://www.nature.com/articles/nmeth.4206) and multiview-based [\[2\]](https://www.biorxiv.org/content/early/2018/07/06/364034) approaches
- cell organelle prediction, extraction and mesh generation
- glia identification and supervoxel graph splitting [\[2\]](https://www.biorxiv.org/content/early/2018/07/06/364034)
- generation of connectivity matrix

Documentation
--------------
To get started, please have a look at our [documentation](https://structuralneurobiologylab.github.io/SyConn/documentation/), but be aware that it is currently outdated and applies only to SyConn v1. We also present more general information about SyConn on our [Website](https://structuralneurobiologylab.github.io/SyConn/).

The documentation for the refactored version is still work-in-progress and can be found [here](docs/doc.md). Alternatively see the latest [readthedocs build](https://syconn.readthedocs.io/en/latest/).

SyConn v1 Publication
---------------------
The first version of SyConn (see branch [dorkenwald2017nm](https://github.com/StructuralNeurobiologyLab/SyConn/tree/dorkenwald2017nm)) was published in [Nature Methods](http://www.nature.com/nmeth/journal/vaop/ncurrent/full/nmeth.4206.html) on February 27th 2017. If you use parts of this code base in your academic projects, please cite the corresponding publication. <br />

  ```
 @ARTICLE{SyConn2017,
   title     = "Automated synaptic connectivity inference for volume electron
                microscopy",
   author    = "Dorkenwald, Sven and Schubert, Philipp J and Killinger, Marius F
                and Urban, Gregor and Mikula, Shawn and Svara, Fabian and
                Kornfeld, Joergen",
   abstract  = "SyConn is a computational framework that infers the synaptic
                wiring of neurons in volume electron microscopy data sets with
                machine learning. It has been applied to zebra finch, mouse and
                zebrafish neuronal tissue samples.",
   journal   = "Nat. Methods",
   publisher = "Nature Publishing Group, a division of Macmillan Publishers Limited. All Rights Reserved.",
   year      = 2017,
   month     = Feb,
   day       = 27,
   url       = http://dx.doi.org/10.1038/nmeth.4206
 }
  ```

# The Team
The Synaptic connectivity inference toolkit is developed at Max-Planck-Institute of Neurobiology, Munich.

Authors: Philipp Schubert, Sven Dorkenwald, Rangoli Saxena, Joergen Kornfeld


# Acknowledgements
We thank deepmind for providing egl extension code to handle multi-gpu rendering on
 the same machine, which is under the Apache License 2.0. The original code snippet used in our
 project can be found [here](https://github.com/deepmind/dm_control/blob/30069ac11b60ee71acbd9159547d0bc334d63281/dm_control/_render/pyopengl/egl_ext.py).