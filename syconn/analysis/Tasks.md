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
- [x] Coloring of meshes (diff obj types)
- [x] Fix rendering of subvolume chunks (gray boxes); check available mags
- [x] Property selector list (manually created); Text boxes to select the values of the property (1. select the attribute, 2. select the number of objects to display)
    1. celltype_cnn_e3 e.g MSN, EA
    2. number of mitochndria in ssv (check mapping_mi_idss.npy)
    3. ssv size (sizes.npy) e.g min, max
    4. supervoxel ids (svs.npy)
- [ ] Invert cell_ids to supervoxel_ids (mapping_dict and mapping_dict_reversed). Same for rag_flat datasets
- [x] Segment query even for the properties will be huge; return a random subset of the segment query (page) or return a range of segment query based on user input
- [x] Check deepcopy behaviour (entire viewer state deepcopied or not)
- [x] Check references of locks (_lock, __lock)
- [x] Hotkey to show the cell connected to the selected cell by the largest synapse
- [x] Include the synapse probs in gettign the largest synapse; ignore the soma connections (axon -> dendrite; axon -> soma)
- [x] Reduce the memory consumption in get_synaptic_partner (combine same dimensional condition checks)
- [x] Write the status message in the format: <pre-synaptic> -> <post-synaptic>
- [x] Center the viewport at the point of the synapse (check rep. coords); zoom-in (optional)
- [x] Include celltypes in status message 
- [x] remove ngrok; directly serve with http behind port 80 
- [x] multiple independent users without accidental synchronization
- [ ] Modify the index page (include a background maybe)
- [ ] Tutorial page (refer on the index page of server)
- [ ] Suppress dual print statements in python interactive mode
- [x] Viewer modifications:
    * [x] Fix type in synaptic partner status message (synpatic)
    * [x] Add 'Loading synaptic partner for selected ssv' when `action_handler` is invoked
    * [x] Implicitly load page 1 in the property filter without having to type '_pg1'
    * [x] Celltype query support in lower and upper
    * [x] Replace placeholder text in segment query
- [x] Time fetching and computation operations (synaptic filtering, skeleton/mesh retrieval, page splitting)
- [ ] Optimize synaptic partner property
- [x] Serve precomputed volume source (on flask-async/quart/tornado)
- [ ] Suppress typescript property map (or use segment properties precomputed)
- [ ] Garbage collector for destroying python objects (ViewerState, PropertyFilter)
- [ ] 

