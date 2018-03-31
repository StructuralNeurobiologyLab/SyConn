# Contact Site Classification

Contact sites are the basis for synaptic classification. Therefore, contact sites need to be combined with the synapse `SegmentationObjects` and then classified as synaptic or not-synaptic using an Random Forest Classifier (RFC).
The code is in `syconn.extraction.cs_processing_steps`, `syconn.proc.sd_proc` and `syconn.proc.ssd_proc`.

## Overlap mapping 

Synapse `SegmentationObjects` are mapped to contact sites by volume overlap the same way `SegmentationObjects` are mapped to supervoxels. First, the aggreagted contact sites (see `contact_site_extraction`) need to be exported to a `knossosdataset`:

```
from syconn.proc import sd_proc
sd_proc.export_sd_to_knossosdataset(cs_sd, cs_kd, block_edge_length=512,
                                    qsub_pe=my_qsub_pe, n_max_co_processes=100)

```
Once exported, the synapse objects can be mapped with 

```
from syconn.extraction import cs_processing_steps as cps
cps.overlap_mapping_sj_to_cs_via_kd(cs_sd, sj_sd, cs_kd, qsub_pe=my_qsub_pe, n_max_co_processes=100, n_folders_fs=10000)
```

This creates a new `SegmentationDataset` of type `conn`. These are contact site objects that overlapped at least with one voxel with a synapse `SegmentationObject`.

Other objects such as vesicle clouds and mitochondria are mapped by proximity. Mapping these objects helps to improve the features used for classifying the contact sites.

```
cps.map_objects_to_conn(...)
```


## Classifying conn objects 

```
cps.create_conn_syn_gt(conn_sd, path_to_gt_kzip)
```

creates the ground truth for the RFC and also trains and stores the classifier. Then, the `conn` `SegmentationObjects` can be classified with

```
cps.classify_conn_objects(working_dir, qsub_pe=my_qsub_pe, n_max_co_processes=100)
```

## Writing the connectivity information to the SuperSegmentationDataset

For convenience and efficiency, the connectivity information created in the last step can be written to the `SuperSegmentationDataset`.

```
ssd_proc.map_synaptic_conn_objects(ssd, qsub_pe=my_qsub_pe, n_max_co_processes=100)
```
