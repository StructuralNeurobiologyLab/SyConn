# SyConn v2

Refactored version of SyConn for automated synaptic connectivity inference based on dense EM segmentation data. For the first version
see 'SyConn' below. Current features:
- introduction of supervoxel and agglomerated supervoxel classes
- added support for (sub-) cellular compartment (spines, axon/dendrite/soma) and cell type classification with [skeleton](https://www.nature.com/articles/nmeth.4206)- and [multiview-based](https://www.biorxiv.org/content/early/2018/07/06/364034) approaches
- cell organelle prediction, extraction and mesh generation
- glia identification and supervoxel graph splitting
- generation of connectivity matrix

Documentation
--------------

To get started, please have a look at our [documentation](https://structuralneurobiologylab.github.io/SyConn/documentation/), but be aware that it is currently outdated and applies only to SyConn v1. We also present more general information about SyConn on our [Website](https://structuralneurobiologylab.github.io/SyConn/).

SyConn v1 Publication
-----------

The first version of SyConn (see branch 'dorkenwald2017nm') was published in [Nature Methods](http://www.nature.com/nmeth/journal/vaop/ncurrent/full/nmeth.4206.html) on February 27th 2017. If you use parts of this code base in your academic projects, please cite the corresponding publication. <br />

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
