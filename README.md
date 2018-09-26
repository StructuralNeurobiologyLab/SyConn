# SyConn v2

Refactored (still an early stage construction) version of SyConn for automated synaptic connectivity inference based on dense EM segmentation data. For the first version
see 'SyConn' below. Current features:
- introduction of supervoxel and agglomerated supervoxel classes
- added support for (sub-) cellular compartment (spines, axon/dendrite/soma) and cell type classification with [skeleton](https://www.nature.com/articles/nmeth.4206)- and [multiview-based](https://www.biorxiv.org/content/early/2018/07/06/364034) approaches
- cell organelle prediction, extraction and mesh generation
- glia identification and splitting
- generation of connectivity matrix

Documentation
--------------
_in progress_


# SyConn

This version of SyConn is deprecated and should not be used anymore.

Synaptic connectivity inference toolkit developed at the Max-Planck-Institute for Medical Research, Heidelberg and
Max-Planck-Institute of Neurobiology, Munich <br />
Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld <br />

Documentation
--------------
To get started, please have a look at our [documentation](https://structuralneurobiologylab.github.io/SyConn/documentation/), with information on how to run our [example](https://github.com/StructuralNeurobiologyLab/SyConn/blob/master/examples/full_run_example.py). We also present more general information about SyConn on our [Website](https://structuralneurobiologylab.github.io/SyConn/).

Publication
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
