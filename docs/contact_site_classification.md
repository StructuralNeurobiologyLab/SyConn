# Contact Site Classification
Contact sites are the basis for synaptic classification. Therefore, contact sites need to be combined with the synapse `SegmentationObjects` to conn `SegmentationObjects` and then further classified as synaptic or not-synaptic using an Random Forest Classifier (RFC).
The code is in `syconn.extraction.cs_processing_steps`, `syconn.proc.sd_proc` and `syconn.proc.ssd_proc`.

## Prerequisites
* SegmentationDataset of [extracted contact sites](contact_site_extraction.md)
* [Synapse type](synapse_type.md) predictions

## Overlap mapping
Synapse `SegmentationObjects` are mapped to contact sites by volume overlap the same way `SegmentationObjects` are mapped to supervoxels. First, the aggreagted contact sites (see `contact_site_extraction`) need to be exported to a `knossosdataset`:

    from syconn.proc import sd_proc
    sd_proc.export_sd_to_knossosdataset(cs_sd, cs_kd, block_edge_length=512,
                                        qsub_pe=my_qsub_pe, n_max_co_processes=100)

Once exported, the synapse objects can be mapped with 

    from syconn.extraction import cs_processing_steps as cps
    cps.overlap_mapping_sj_to_cs_via_kd(cs_sd, sj_sd, cs_kd, qsub_pe=my_qsub_pe, n_max_co_processes=100, n_folders_fs=10000)


This creates a new `SegmentationDataset` of type `conn`. These are contact site objects that overlapped at least with one voxel with a synapse `SegmentationObject`.

Other objects such as vesicle clouds and mitochondria are mapped by proximity. Mapping these objects helps to improve the features used for classifying the conn objects.

    cps.map_objects_to_conn(...)


## Classifying conn objects

In principle, one could imagine that the overlap between a synapse object and a contact site object is already a sufficient identification of a synapse between two neurites. In practice, we found that a further classification can improve the performance,
because it can incorporate other relevant features, such as vesicles clouds in proximity.

    cps.create_conn_syn_gt(conn_sd, path_to_gt_kzip)

creates the ground truth for the RFC and also trains and stores the classifier. Then, the `conn` `SegmentationObjects` can be classified with


    cps.classify_conn_objects(working_dir, qsub_pe=my_qsub_pe, n_max_co_processes=100)


## Collecting directionality information (axoness)
`conn` `SegmentationObjects` can acquire information about the "axoness" of both partners around the synapse. This allows
a judgement about the direction of the synapse. To collect this information from the `ssv` partners do

    cps.collect_axoness_from_ssv_partners(wd, qsub_pe=my_qsub_pe, n_max_co_processes=100)

The axoness prediction version used her can currently only be changed in the code directly (see `cps._collect_axoness_from_ssv_partners_thread`).


## Writing the connectivity information to the SuperSegmentationDataset
For convenience and efficiency, the connectivity information created in the last step can be written to the `SuperSegmentationDataset`.

    ssd_proc.map_synaptic_conn_objects(ssd, qsub_pe=my_qsub_pe, n_max_co_processes=100)

This enables direct look-ups on the level of ssv's, without having to go back to the sv objects, which would add delays.

## Exporting the connectivity matrix

The connectivity matrix can be exported in various formats, such as a networkx graph, or a csv file.

    cps.export_matrix(wd)
