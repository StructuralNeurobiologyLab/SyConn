# Meshes
Like skeletons, meshes are also computed on `SegmentationObject` level and can
be merged for `SuperSegmentationObjects`, which agglomerate `SegmentationObject`
instances.

Mesh computation is implemented in `syconn.proc.meshes` and based on marching
cubes as implemented in skimage.

The method `triangulation` can operate on sparse point clouds or on dense
volumetric bool arrays.

SSO meshes can be called like `sso.load_mesh(obj_type)` where`obj_type` can be
 in `['sv', 'mi', 'sj', 'vc']` or directly from properties `sso.mesh`, `sso.mi_mesh`, ...
 For SO you can access meshes via `so.mesh`, if they do not exist yet they will
 be computed and cached automatically.

For caching all object meshes and of SSVs in a SuperSegmentationDataset (SSD)
see `mesh_preprocessing.py` in `SyConn/scripts/backend`. This is run on a
single node with 20 cpus by default (sufficiently fast).




