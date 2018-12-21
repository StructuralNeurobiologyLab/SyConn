# SuperSegmentation datasets

`SuperSegmentationDatasets` (SSD) and `SuperSegmentationObjects` (SSO; see corresponding section)
 are implemented in `super_segmentation_object.py` and `super_segmentation_object` (`syconn.reps`).
It is accompanied by helper functions in `super_segmentation_helper.py` for basic functionality such as
 loading and storing and `ssd_proc.py` and `ssd_proc.assembly` (`syconn.proc`) which contain processing methods.

The first initializing of an SSD usually happens after [glia removal](glia_removal.md).

## Prerequisites
* Knossos- and SegmentationDataset of the super voxel segmentation
* Initial RAG/SV-agglomeration

## Initialization

In order to create a SuperSegmentationDataset from scratch one has to provide
the agglomerated super voxel (SSV) defined as a dict (coming soon!; AGG_SOURCE; keys: SSV IDs and values: list of SVs) or stored as a
KNOSSOS mergelist (text file; variable holding the path string: AGG_SOURCE) and pass it
to the constructor (kwarg: 'sv_mapping'). The `version` kwarg is used to distinguish between different SSV datasets, e.g. if one
 is interested in separating the initial RAG into neuron and glia segmentation one could use `version='glia'` and `version='neuron'`.
 By default, the version is incremented by one starting at 0 for same `ssd_type`'s.

    ssd = ss.SuperSegmentationDataset(working_dir=WORKING_DIR,
                                      version=VERSION, ssd_type="ssv",
                                      sv_mapping=AGG_SOURCE)
    ssd.save_dataset_shallow()
    ssd.save_dataset_deep(qsub_pe="openmp", n_max_co_processes=100)
    # alternatively for small datasets: ssd.save_dataset_deep(nb_cpus=20, stride=5)

It is recommended to cache the SSV meshes, which means that they are copied together from the meshes of the underlying SVs. For this use:

    syconn.proc.ssd_proc.mesh_proc_ssv(WD, VERSION, ssd_type="ssv", nb_cpus=20)

A summary script for the initial SSD generation, called `create_ssd.py`, can be found at `SyConn/scripts/`.
It combines the above procedures, the [mapping of cellular organelles](object_mapping.md) and saves a SV-graph for every SSV.







