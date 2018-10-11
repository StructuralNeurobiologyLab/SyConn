# SuperSegmentation datasets

`SuperSegmentationDatasets` (SSD) and `SuperSegmentationObjects` (SSO; see corresponding section) are implemented in `super_segmentation_object.py` and `super_segmentation_object` (`syconn.reps`). 
It is accompanied by helper functions in `super_segmentation_helper.py` for basic functionality such as loading and storing and 
`ssd_proc.py` and `ssd_proc.assembly` (`syconn.proc`) which contain processing methods. 

Typically, initializing the SSD happens after glia removal. 
Please check the corresponding documentation to learn more about that.


## Initialization

In order to create a SuperSegmentationDataset from scratch one has to provide
the agglomerated super voxel (SSV) defined as a dict (coming soon!; agglomeration_source; keys: SSV IDs and values: list of SVs) or stored as a
KNOSSOS mergelist (text file; variable holding the path string: agglomeration_source) and parse it
to the constructor (kwarg: 'sv_mapping').



    ssd = ss.SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/",
                                      version="spgt", ssd_type="ssv",
                                      sv_mapping=agglomeration_source)
    ssd.save_dataset_shallow()
    ssd.save_dataset_deep(qsub_pe="openmp", n_max_co_processes=100)
    # alternatively for small datasets: ssd.save_dataset_deep(nb_cpus=20, stride=5)








