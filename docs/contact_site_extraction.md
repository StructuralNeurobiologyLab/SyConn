# Contact Site Extraction
Contact sites are extracted from an existing segmentation. The main functionality is in
`syconn.extraction.cs_extraction_steps` and `syconn.extraction.cs_processing_steps`.

The exection script is located at `SyConn/scrips/syns/syn_gen.py`.

It consists of two steps: (1) Finding and extracting contact sites between supervoxels and (2)
 combining the contact sites between supersegmentation objects. These steps have some similarity
  with the object extraction and they share some functionality. The main difference is that for objects
  (eg. mitochondria) the extent is known at extraction time, meaning that they are not split up into
  smaller fragments that have to be assembled later on, as is the case for supervoxels and super-supervoxels,
  because an oversegmentation is necessary in these cases.

## Prerequisites
* Knossos- and SegmentationDataset of the super voxel segmentation
* SuperSegmentationDataset containg the cell reconstructions (SV-agglomeration/RAG)
* KnossosDataset of symmetric, asymmetric and SJ predictions (WIP)

## Finding Contact Sites

Contact sites are detected from a segmentation stored in knossos overlaycubes and saved to a chunk dataset (see `chunk_prediction` for details on how to create chunk datasets).
This combines all contact sites between two supervoxels into a single object. `combine_and_split_cs_agg` splits these apart later, based on connected components.
The first step reads from a KNOSSOS dataset with the segmentation and saves the extracted contact sites into an hdf5 chunk dataset.

    from syconn.extraction import cs_extraction_steps as ces
    ces.find_contact_sites(cset, knossos_path, n_max_co_processes=200,
                           qsub_pe=my_qsub_pe)

The second step used the hdf5 chunk dataset and generates a segmentation dataset with the results.

    from syconn.extraction import cs_extraction_steps as ces
    ces.extract_agg_contact_sites(cset, working_dir,
                                  n_folders_fs=10000, suffix="",
                                  n_max_co_processes=200, qsub_pe=my_qsub_pe)

Next, the resulting contact sites are overlapped with synaptic
 junction objects (SJ) using the previous CS ChunkDataset, which will create
 a SegmentationDataset of 'syn' objects:

    from syconn.extraction import cs_processing_steps as cps
    cps.syn_gen_via_cset(working_dir)

This creates a new `SegmentationDataset` of type `synn`. These are contact site objects that overlapped at least with one voxel with a synaptic junction `SegmentationObject`.

## Aggregating Contact Sites
To agglomerate 'syn' objects, the SV-agglomeration i.e. SSVs are required. Based on their region-adjacency graph SV-wise 'syn' objects
are agglomerated to 'syn_ssv' objects, stored in a SegmentationDataset.

    from syconn.extraction import cs_processing_steps as cps
    cps.combine_and_split_cs_agg(working_dir)

It combines contact sites between the same supersupervoxels and splits them based on a
maximal voxel distance of `cs_gap_nm` which can be specified in `syconn/config/global_params.py`.




