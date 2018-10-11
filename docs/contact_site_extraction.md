# Contact Site Extraction

Contact sites are extracted from an existing segmentation. The main functionality is in `syconn.extraction.cs_extraction_steps` and `syconn.extraction.cs_processing_steps`.

It consists of two steps: (1) Finding and extracting contact sites between supervoxels and (2) combining the contact sites between supersegmentation objects. 
These steps have some similarity with the object extraction and they share some functionality. The main difference is that for objects (eg. mitochondria) the extent is known at extraction time. This is not the case for contact sites because they are extracted based on the supervoxels which are imperfect.

## Finding Contact Sites

Contact sites are detected from a segmentation stored in knossos overlaycubes and saved to a chunk dataset (see `chunk_prediction` for details on how to create chunk datasets).
This combines all contact sites between two supervoxels into a single object. `combine_and_split_cs_agg` splits these apart later.

```
from syconn.extraction import cs_extraction_steps as ces
ces.find_contact_sites(cset, knossos_path, filename, n_max_co_processes=200,
                       qsub_pe=my_qsub_pe)
```

Extracting contact site objects is then same as extracting suprevoxel objects from the segmentation:

```
from syconn.extraction import cs_extraction_steps as ces
ces.extract_agg_contact_sites(cset, filename, hdf5name, working_dir,
                              n_folders_fs=10000, suffix="",
                              n_max_co_processes=200, qsub_pe=my_qsub_pe)
```

See `segmentation_datasets` for explanations for parameters related to `SegmentationDatasets`. 


## Aggregating Contact Sites

Once a `SuperSegmentationDataset` is created the contact site `SegmentationDataset` might be transformed to a new one that incorporates the agglomarations from the `SuperSegmentationDatset`.
The main step is

```
from syconn.extraction import cs_processing_steps as cps
cps.combine_and_split_cs_agg(working_dir, cs_gap_nm=300,
                             stride=100, qsub_pemyqsub_pe,
                             n_max_co_processes=200)
```

It combines contact sites between the same supersupervoxels and splits them based on a maximal voxel distance of `cs_gap_nm`.
