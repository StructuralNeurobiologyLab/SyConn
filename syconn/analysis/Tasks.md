# Tasks
- [x] Skeleton rendering (neuroglancer.SkeletonSource)
- [x] Simple mesh generation in LocalVolume
- [x] Mitochondira mesh rendering
- [x] Synaptic junction mesh rendering
- [x] Synapse mesh rendering
- [x] Merge mesh with skeleton in one layer
- [x] Seg ids are only connected to the ssv_mesh. When unselecting a mesh/data, only the ssv_mesh disappears not all the cell organelles too. Make a tab in Neuroglancer for visibility of organelle meshes? All the local volumes of the organelles needs to be in the same segmentation layer for disappearing (conflicts with segment colors)
- [x] Data is passed in every single layer (redundant neuroglancer.LocalVolume) => slow. Find a way to optimize or merge layers. Create an independent mesh source?
- [x] Change colors of cell organelles and opacity of ssv mesh
- [x] Colors are overlayed in the 2d layouts because of the same `segment_colors` argument of the segmentation layer. Need to define color only for the mesh.
- [x] Get colors working with linking segmentation layers
- [x] Level of detail integration
- [x] Precomputed skeleton source 
- [x] Dynamically generate layers for the obj_types wanted
- [x] Response in flask json.dumps and binary
- [x] Integrate flask into cli main
- [x] Specify organelle types in the main parameters
- [ ] Look into import folders for absolute path
- [x] Look into the downsampling where it is happening in the frontend or Neuroglancer server
- [x] Compare highest factor and upsampling with lowest factor and downsampling (runtime)
- [x] Render differently colored organelle meshes with a mouse click (without the seg query)
- [ ] Coloring of meshes (diff obj types)
- [x] Fix rendering of subvolume chunks (gray boxes); check available mags
- [x] Property selector list (manually created); Text boxes to select the values of the property (1. select the attribute, 2. select the number of objects to display)
    1. celltype_cnn_e3 e.g MSN, EA
    2. number of mitochndria in ssv (check mapping_mi_idss.npy)
    3. ssv size (sizes.npy) e.g min, max
- [ ] svs.npy (all the supervoxel ids)
- [ ] Invert cell_ids to supervoxel_ids (mapping_dict and mapping_dict_reversed). Same for rag_flat datasets
- [ ] Method in syconn server receiving the coords, bounding box and cell type and give back the subset of ids (using numba or cython)
- [x] Segment query even for the properties will be huge; return a random subset of the segment query or return a range of segment query based on user input or show segments page-wise
- [x] Check deepcopy behaviour (entire viewer state deepcopied or not)
- [x] Check references of locks (_lock, __lock)

