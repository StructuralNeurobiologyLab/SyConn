[![Documentation Status](https://readthedocs.org/projects/syconn/badge/?version=latest)](https://syconn.readthedocs.io/en/latest/?badge=latest)

# SyConn
Refactored version of SyConn for automated synaptic connectivity inference based on dense EM segmentation data. For the first version
 see below or checkout the branch [dorkenwald2017nm](https://github.com/StructuralNeurobiologyLab/SyConn/tree/dorkenwald2017nm).

Current features:
- introduction of supervoxel and agglomerated supervoxel classes
- added support for (sub-) cellular compartment (spines, axon/dendrite/soma) and cell type classification with skeleton- [\[1\]](https://www.nature.com/articles/nmeth.4206) and multiview-based [\[2\]](https://www.nature.com/articles/s41467-019-10836-3) approaches
- cell organelle prediction, extraction and mesh generation
- glia identification and separation [\[2\]](https://www.nature.com/articles/s41467-019-10836-3)
- generation of connectivity matrix

If you use parts of this code base in your academic projects, please cite the corresponding publication.

Documentation
--------------
To get started, please have a look at our [documentation](https://structuralneurobiologylab.github.io/SyConn/documentation/), but be aware that it is currently outdated and applies only to SyConn v1. We also present more general information about SyConn on our [Website](https://structuralneurobiologylab.github.io/SyConn/).

The documentation for the refactored version is still work-in-progress and can be found [here](docs/doc.md). Alternatively see the latest [readthedocs build](https://syconn.readthedocs.io/en/latest/).


# The Team
The Synaptic connectivity inference toolkit is developed at the Max-Planck-Institute of Neurobiology in Martinsried by
 Philipp Schubert, Maria Kawula, Carl Constantin v. Wedemeyer, Atul Mohite, Gaurav Kumar and Joergen Kornfeld.


# Acknowledgements
We thank deepmind for providing egl extension code to handle multi-gpu rendering on the same machine, which is under the Apache License 2.0. The original code snippet used in our
 project can be found [here](https://github.com/deepmind/dm_control/blob/30069ac11b60ee71acbd9159547d0bc334d63281/dm_control/_render/pyopengl/egl_ext.py).


# References
\[1\] [Automated synaptic connectivity inference for volume electron microscopy](https://www.nature.com/articles/nmeth.4206)
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


\[2\] [Learning cellular morphology with neural networks](https://doi.org/10.1038/s41467-019-10836-3)
  ```
  @Article{Schubert2019,
author={Schubert, Philipp J.
and Dorkenwald, Sven
and Januszewski, Michal
and Jain, Viren
and Kornfeld, Joergen},
title={Learning cellular morphology with neural networks},
journal={Nature Communications},
year={2019},
volume={10},
number={1},
pages={2736},
abstract={Reconstruction and annotation of volume electron microscopy data sets of brain tissue is challenging but can reveal invaluable information about neuronal circuits. Significant progress has recently been made in automated neuron reconstruction as well as automated detection of synapses. However, methods for automating the morphological analysis of nanometer-resolution reconstructions are less established, despite the diversity of possible applications. Here, we introduce cellular morphology neural networks (CMNs), based on multi-view projections sampled from automatically reconstructed cellular fragments of arbitrary size and shape. Using unsupervised training, we infer morphology embeddings (Neuron2vec) of neuron reconstructions and train CMNs to identify glia cells in a supervised classification paradigm, which are then used to resolve neuron reconstruction errors. Finally, we demonstrate that CMNs can be used to identify subcellular compartments and the cell types of neuron reconstructions.},
issn={2041-1723},
doi={10.1038/s41467-019-10836-3},
url={https://doi.org/10.1038/s41467-019-10836-3}
}
  ```
