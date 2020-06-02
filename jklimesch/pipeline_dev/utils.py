import numpy as np
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property


def sso2ce(sso: SuperSegmentationObject, mi: bool = True, vc: bool = True,
           sy: bool = True, my: bool = False, my_avg: bool = True, mesh: bool = False) -> CloudEnsemble:
    """ Converts a SuperSegmentationObject into a CloudEnsemble (ce). Cell organelles are saved
        as additional cloud in the ce, named as in the function parameters (e.g. 'mi' for
        mitochondria). The no_pred (no prediction) flags of the ce are set for all additional
        clouds. Myelin is added in form of the types array of the HybridCloud, where myelinated
        vertices have type 1 and 0 otherwise.

    Args:
        sso: The SuperSegmentationObject which should get converted to a CloudEnsemble.
        mi: Flag for including mitochondria.
        vc: Flag for including vesicle clouds.
        sy: Flag for including synapses.
        my: Flag for including myelin.
        my_avg: Flag for applying majority vote on myelin property.
        mesh: Flag for storing all objects as HybridMesh objects with additional faces.

    Returns:
        CloudEnsemble object as described above.
    """
    # convert cell organelle meshes
    clouds = {}
    if mi:
        indices, vertices, normals = sso.mi_mesh
        if mesh:
            clouds['mi'] = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)))
        else:
            clouds['mi'] = PointCloud(vertices=vertices.reshape((-1, 3)))
    if vc:
        indices, vertices, normals = sso.vc_mesh
        if mesh:
            clouds['vc'] = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)))
        else:
            clouds['vc'] = PointCloud(vertices=vertices.reshape((-1, 3)))
    if sy:
        indices, vertices, normals = sso._load_obj_mesh('syn_ssv', rewrite=False)
        if mesh:
            clouds['sy'] = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)))
        else:
            clouds['sy'] = PointCloud(vertices=vertices.reshape((-1, 3)))
    # convert cell mesh
    indices, vertices, normals = sso.mesh
    sso.load_skeleton()
    if mesh:
        hm = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)),
                        nodes=sso.skeleton['nodes']*sso.scaling, edges=sso.skeleton['edges'])
    else:
        hm = HybridCloud(vertices=vertices.reshape((-1, 3)), nodes=sso.skeleton['nodes']*sso.scaling,
                         edges=sso.skeleton['edges'])
    # merge all clouds into a CloudEnsemble
    ce = CloudEnsemble(clouds, hm, no_pred=[obj for obj in clouds])
    if my:
        add_myelin(sso, hm, average=my_avg)
    return ce


def sso2hc(sso: SuperSegmentationObject, mi: bool = True, vc: bool = True,
           sy: bool = True, my: bool = False, my_avg: bool = True) -> HybridCloud:
    """ Converts a SuperSegmentationObject into a HybridCloud (hc). The object boundaries
        are stored in the obj_bounds attribute of the hc. The no_pred (no prediction) flags
        are set for all cell organelles. Myelin is added in form of the types array of the
        hc, where myelinated vertices have type 1 and 0 otherwise.

    Args:
        sso: The SuperSegmentationObject which should get converted to a CloudEnsemble.
        mi: Flag for including mitochondria.
        vc: Flag for including vesicle clouds.
        sy: Flag for including synapses.
        my: Flag for including myelin.
        my_avg: Flag for applying majority vote on myelin property.

    Returns:
        HybridCloud object as described above.
    """
    vertex_num = 0
    # convert cell organelle meshes
    clouds = []
    obj_names = []
    if mi:
        indices, vertices, normals = sso.mi_mesh
        clouds.append(vertices.reshape((-1, 3)))
        obj_names.append('mi')
        vertex_num += len(vertices.reshape((-1, 3)))
    if vc:
        indices, vertices, normals = sso.vc_mesh
        clouds.append(vertices.reshape((-1, 3)))
        obj_names.append('vc')
        vertex_num += len(vertices.reshape((-1, 3)))
    if sy:
        indices, vertices, normals = sso._load_obj_mesh('syn_ssv', rewrite=False)
        clouds.append(vertices.reshape((-1, 3)))
        obj_names.append('sy')
        vertex_num += len(vertices.reshape((-1, 3)))
    # convert cell mesh
    indices, vertices, normals = sso.mesh
    hc_vertices = vertices.reshape((-1, 3))
    vertex_num += len(hc_vertices)
    # merge all clouds into a HybridCloud
    total_verts = np.zeros((vertex_num, 3))
    bound = len(hc_vertices)
    obj_bounds = {'hc': [0, bound]}
    total_verts[0:bound] = hc_vertices
    for ix, cloud in enumerate(clouds):
        if len(cloud) == 0:
            # ignore cell organelles with zero vertices
            continue
        obj_bounds[obj_names[ix]] = [bound, bound+len(cloud)]
        total_verts[bound:bound+len(cloud)] = cloud
        bound += len(cloud)
    sso.load_skeleton()
    hc = HybridCloud(vertices=total_verts, nodes=sso.skeleton['nodes']*sso.scaling, edges=sso.skeleton['edges'],
                     obj_bounds=obj_bounds, no_pred=obj_names)
    if my:
        add_myelin(sso, hc, average=my_avg)
    return hc


def add_myelin(sso: SuperSegmentationObject, hc: HybridCloud, average: bool = True):
    """ Tranfers myelin prediction from a SuperSegmentationObject to an existing
        HybridCloud (hc). Myelin is added in form of the types array of the hc,
        where myelinated vertices have type 1 and 0 otherwise. Works in-place.

    Args:
        sso: SuperSegmentationObject which contains skeleton to which myelin should get mapped.
        hc: HybridCloud to which myelin should get added.
        average: Flag for applying majority vote to the myelin property
    """
    sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    if average:
        majorityvote_skeleton_property(sso, 'myelin')
        myelinated = sso.skeleton['myelin_avg10000']
    else:
        myelinated = sso.skeleton['myelin']
    nodes_idcs = np.arange(len(hc.nodes))
    myel_nodes = nodes_idcs[myelinated.astype(bool)]
    myel_vertices = []
    for node in myel_nodes:
        myel_vertices.extend(hc.verts2node[node])
    # myelinated vertices get type 1, not myelinated vertices get type 0
    types = np.zeros(len(hc.vertices))
    types[myel_vertices] = 1
    hc.set_types(types)
