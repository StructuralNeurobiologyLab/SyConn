# Tasks

- [x] Skeleton rendering
- [x] Simple mesh generation in LocalVolume
- [x] Mitochondira mesh rendering
- [x] Synaptic junction mesh rendering
- [x] Synapse mesh rendering
- [x] Merge mesh with skeleton in one layer
- [ ] Level of detail integration
- [ ] Dynamically generate layers for the obj_types wanted
- [ ] Seg ids are only connected to the ssv_mesh. When unselecting a mesh/data, only the ssv_mesh disappears not all the cell organelles too. Make a tab in Neuroglancer for visibility of organelle meshes? All the local volumes of the organelles needs to be in the same segmentation layer for disappearing (conflicts with segment colors)
- [x] Data is passed in every single layer => slow. Find a way to optimize or merge layers. Create an independent mesh source?
- [ ] Change colors of cell organelles and opacity of ssv mesh
- [ ] Colors are overlayed in the 2d layouts because of the same `segment_colors` argument of the segmentation layer. Need to define color only for the mesh.
