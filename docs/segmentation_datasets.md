# Segmentation datasets

`SegmentationDatasets` and `SegmentationObjects` are implemented in `segmentation.py` (`syconn.reps`). 
It is accompanied by helper functions in `segmentation_helper.py` and `rep_helper.py` (`syconn.reps`) for 
basic functionality such as loading and storing and `sd_proc.py` (`syconn.proc`) intensive processing that 
is usually parallelized. 

Typically, the voxel storage of a  `SegmentationDatasets` is created first (eg. by the object extraction). 
Please check the corresponding documentation to learn more about that.

On a fundamental level, each `SegmentationObject` owns voxels, attributes, a skelton and a mesh which 
are stored in different dictionaries (`VoxelDict`, `AttributeDict`, `SkeletonDict`, `MeshDict`; see section 'Backend'). 
Each dictionary consists of the associated data from many objects and compresses it individually for
efficient storage. The number of dictionaries per data type can be defined with `n_folders_fs` (only powers of 10). 
Please note for the general way of creating `SegmentationDatasets` this has to be passed to the object extraction as well.

## Initialization

To load (or create) a `SegmentationDataset` at least the `obj_type` has to be defined. Defaults exist for 
other parameters such as `version` and `working_dir`. These are stored in `config.ini` (eg. `version`) in 
the `working_dir` and project wide in `config.global_params` (eg. `working_dir`). 

```
sd_cell_sv = SegmentationDataset("sv", working_dir="path/to/wd")
```

It is useful to run `sd_proc.dataset_analysis(...)` when loading a `SegmentationDataset` the first time 
(after writing its voxel storage) or after making changes to the attributes. `dataset_analysis` creates global `numpy`
arrays for fast access for each attribute and calculates some attributes itself (such as `size` and `bounding box`). This can 
be viewed as a distributed column store of the underlying database.

```
sd_proc.dataset_analysis(sd_cell_sv)
```

When running `dataset_analysis` one can include only a subset of the attributes to avoid problems with non-consistent 
entries (see below). As most functions, `dataset_analysis` can either run on a single shared memory system or on 
a distributed custer using `qsub`.

It also is recommended to preprocess the meshes of the SegmentationObjects.
See `mesh_proc_chunked` in `syconn/proc/sd_proc.py`.

## Usage

If `sd_proc.dataset_analysis(...)` was applied, the `SegmentationDataset` can access the values of an attribute of all objects 
as an array. For instance, the attribute `size` can be accesses via

```
sizes = sd_cell_sv.load_numpy_data("size")
```

Some attributes, such as `size` and `id`, are also available as object attributes (e.g. `sd_cell_sv.sizes`). Values in 
different attribute arrays are always sorted in the same way. Hence, one can use the id array (`sd_cell_sv.ids`) as a reference.

A `SegmentationDataset` allows easy access to its `SegmentationObjects` by

```
cell_sv_obj = sd_cell_sv.get_segmentation_object(obj_id)
```

There are four additional data structures for each `SegmentationObject`: voxels (`VoxelStorage`), attributes 
(`AttributeDict`), meshes (`MeshStorage`) and skeletons (`SkeletonStorage`). 
Typically, every `SegmentationObject` owns the first three while only supervoxels (`sv`) have a skeleton. While 
voxels, meshes and skeletons are predefined datatypes, attributes are an arbitrary key value store. It is advised though to be consistent in type and
naming of attributes across the `SegmentationDataset` to avoid problems with the aforementioned numpy arrays.

The different data structures can be accessed by e.g.

```
voxels = cell_sv_obj.voxels
mesh = cell_sv_obj.mesh
skeleton = cell_sv_obj.skeleton
attr_value = cell_sv_obj.lookup_in_attribute_dict("attr_key")
```

The attribute dict can also be accessed as a whole

```
cell_sv_obj.load_attr_dict()
attr_dict = cell_sv_obj.attr_dict
```

`SegmentationObjects` cache data that was accessed. This can be disabled by
```
cell_sv_obj.mesh_caching = False
cell_sv_obj.voxel_caching = False
cell_sv_obj.skeleton_caching = False
```

and the cache can be cleared by 

```
cell_sv_obj.clear_cache()
```






