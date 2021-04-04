## API code examples

### Usage of the SegmentationDataset property caches

``SegmentationObjects`` retrieved through the factory method ``SegmentationDataset.get_segmentation_object``
will not load the object's attribute dictionary by default in order to reduce IO overhead. E.g.
using the factory method can be convenient to access the voxels or meshes of many objects, but this does not
require any properties from the attribute dictionary.

    In [77]: from syconn.reps.super_segmentation import *                                                                 

    In [78]: global_params.wd = "/wholebrain/scratch/pschuber/SyConn/example_cube/"                                      
    
    In [79]: sd = SegmentationDataset('sv')                                                                               
    
    # choose any supervoxel, here the first in the ids array
    In [80]: sv = sd.get_segmentation_object(sd.ids[0])                                                                   
    
    In [81]: sv.attr_dict                                                                                                 
    Out[81]: {}

The properties stored in ``SegmentationObject.attr_dict`` are cached as numpy arrays and stored on disk
in the ``SegmentationDataset`` folder during ``dataset_analysis``. Many, sparse look-ups for specific properties can
be realized by loading the cache arrays via ``load_cached_data`` and using ``SegmentationDataset.ids`` to retrieve the
index of the object of interest in the property cache:

    In [82]: prop_look_up = dict(size=sd.load_numpy_data('size'), rep_coord=sd.load_numpy_data('rep_coord'))
    
    In [83]: prop_look_up['rep_coord'][np.where(sd.ids == sv.id)]                                                         
    Out[83]: array([[1475, 1297,  882]], dtype=int32)
    
    In [84]: prop_look_up['size'][np.where(sd.ids == sv.id)]                                                              
    Out[84]: array([1750])

Another solution with similar complexity but a more convenient interface is to pass the property keys during the init. of ``SegmentationDataset``.
Now ``get_segmentation_object`` will use a lookup from object ID to the index in the respective property numpy array to populate the object's attribute dictionary:

    In [85]: sd = SegmentationDataset('sv', cache_properties=('rep_coord', 'size'))                                       
    
    In [86]: sv = sd.get_segmentation_object(sd.ids[0])                                                                   
    
    In [87]: sv.attr_dict                                                                                                 
    Out[87]: {'rep_coord': array([1475, 1297,  882], dtype=int32), 'size': 1750}

This mechanism allows quick access to specific attributes instead of loading the entire dictionary from
file for every single object. Besides a considerable reduction of file reads this also avoids to read
properties which are not of interest for the current process, which can be quite many:

    In [88]: sv.load_attr_dict()                                                                                          

    In [89]: sv.attr_dict                                                                                                 
    Out[89]: 
    {'mapping_mi_ids': [],
     'mapping_mi_ratios': [],
     'mapping_sj_ids': [],
     'mapping_sj_ratios': [],
     'mapping_vc_ids': [],
     'mapping_vc_ratios': [],
     'rep_coord': array([1475, 1297,  882], dtype=int32),
     'bounding_box': array([[1475, 1297,  881],
            [1492, 1351,  898]], dtype=int32),
     'size': 1750,
     'mesh_bb': [array([14700.   , 12940.411, 17632.592], dtype=float32),
      array([14875.668, 13468.265, 17934.43 ], dtype=float32)],
     'mesh_area': 0.194492}

In addition, the properties which are assigned to every ``SegmentationObject`` instance
are also available to be used for further processing. For example, the cache-enabled ``SegmentationDataset`` can be passed
to ``SuperSegmentationDataset`` via the ``sd_lookup`` kwarg, which will be passed on to ``SuperSegmentationObject``s
instantiated through the factory method ``SuperSegmentationDataset.get_supersegmentation_object``. Calls to sub-cellular structures of
neurons, e.g. synapses via ``SuperSegmentationObject.syn_ssv`` or mitochondria via ``SuperSegmentationObject.mis`` will contain the
properties ``cache_properties``.
The method ``SuperSegmentationObject.typedsyns2mesh`` needs access to the synaptic sign of all the cell's synapses, which
requires zero file reads when enabling the cache (meshes still have to be loaded of course):

    In [50]: sd_syn_ssv = SegmentationDataset('syn_ssv', cache_properties=('syn_sign', ))
    
    In [51]: ssd = SuperSegmentationDataset(sd_lookup=dict(syn_ssv=sd_syn_ssv))
    
    # choose any, here the first in the ssv_id array, cell reconstruction
    In [52]: ssv = ssd.get_super_segmentation_object(ssd.ssv_ids[0])
    
    In [53]: ssv.typedsyns2mesh() 

In case all data points are of interest, the recommended way to use the cache is via numpy array indexing:

    In [89]: sd = SegmentationDataset('sv')
    
    # load the mesh bounding box
    In [90]: mesh_bb = sd.load_numpy_data('mesh_bb')  # N, 2, 3

    # calculate the diagonal
    In [91]: mesh_bb = np.linalg.norm(mesh_bb[:, 1] - mesh_bb[:, 0], axis=1)                                              
    
    # get supervoxels with a bounding box diagonal above 5um
    In [92]: filtered_ids = sd.ids[mesh_bb > 5e3]                                                                         
    
    # percentage of supervoxels above threshold
    In [93]: len(filtered_ids) / len(sd.ids)                                                                              
    Out[93]: 0.0632114971144053
    
    # Get the coordinates of the supervoxels above and the size in voxels of those below
    In [94]: mask = mesh_bb > 5e3                                                                                         

    In [95]: sv_coords_of_interest = sd.rep_coords[mask]                                                                  
    
    In [96]: sv_sizes_vx_filtered = sd.sizes[~mask] 


### Myelin prediction

The entire myelin prediction for a single cell reconstruction including a smoothing
is implemented and can be manually invoked as follows::

    from syconn import global_params
    from syconn.reps.super_segmentation import *
    from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property

    # init. example data set
    global_params.wd = '~/SyConn/example_cube1/'

    # initialize example cell reconstruction
    ssd = SuperSegmentationDataset()
    ssv = list(ssd.ssvs)[0]
    ssv.load_skeleton()

    # get myelin predictions
    myelinated = map_myelin2coords(ssv.skeleton["nodes"])
    ssv.skeleton["myelin"] = myelinated
    # this will generate a smoothed version at ``ssv.skeleton["myelin_avg10000"]``
    majorityvote_skeleton_property(ssv, "myelin")
    # store results as a KNOSSOS readable k.zip file
    ssv.save_skeleton_to_kzip(dest_path='~/{}_myelin.k.zip'.format(ssv.id),
        additional_keys=['myelin', 'myelin_avg10000'])