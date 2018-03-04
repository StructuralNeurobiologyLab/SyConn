# From Probability maps to segmentation objects

Probability maps and segmentations are stored in `ChunkDatasets` (see `chunky.py` in `knossos_utils`) 
and are transformed to `SegmentationDatasets` (see `segmentationdataset` in `syconn.reps`) in multiple steps. 
The code associated with this can be found in `syconn.extraction`. `object_extraction_wrapper.py` 
combines all necessary steps from `object_extraction_steps.py`.

Please note that there is further preprocessing needed before the `SegmentationDataset` created by 
this pipeline can be used. These steps are covered by the corresponding documentation.

## Wrappers

Currently, wrappers for two use cases exist: `from_probabilities_to_objects` and `from_ids_to_objects`. 
The latter is a subset of the former and is mainly used to transform supervoxel segmentations 
to `SegmentationDatasets`. The relevant inputs are a `ChunkDataset`, the `filename` of the 
specific prediction within the `ChunkDataset` and the `hdf5names` in the file that are used in this extraction.

## Step by Step

The wrappers sequentially call specific functions from `object_extraction_steps.py`. Parallelism is only 
possible within these steps. `from_ids_to_objects` starts at step 4.

1. **Connected components** within each chunk are created for each chunk by applying a Gaussian smoothing  (optional) and threshold first (`gauss_threshold_connected_components(...)`).
2. `make_unique_labels` reassignes globally **unique labels** to all segments
3. `make_stitch_list` collects information of which segments in different chunks are in fact the same and `make_merge_list` resolves this to a global **mergelist** that is then applied by `apply_merge_list`.
4. `extract_voxels` writes the voxels of each object to a **temporary voxel storage** (similar to the voxel store of a `SegmentationDataset`) and guarantees no write conflicts.
5. In `combine_voxels` each worker then reads the voxels belonging to each object from the temporary voxel storage and writes them to their final location, essentially **creating a `SegmentationDataset`**.

Steps 4 and 5 are necessary to prevent two workers to write to the same `VoxelDict` (hence, to avoid having locks) . This would happen because an object extends 
over multiple chunks or because the ids of two different objects are assigned to the same `VoxelDict`. it also allows to balancing the 
size of individual `VoxelDicts` in the final `SegmentationDataset`.
