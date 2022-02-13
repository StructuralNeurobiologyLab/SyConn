from audioop import reverse
from multiprocessing import cpu_count
import time
import json
from typing import Union
import argparse

import numpy as np

from syconn import global_params
from syconn.analysis.backend import SyConnBackend
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.logger import log_main as log_gate
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap


def get_encoded_skeleton(ssd: SuperSegmentationDataset, ssv_id: int, reverse_lookup=False) -> bytes:
    """Gets encoded skeleton for ssv_id.

    Args:
        ssd (SuperSegmentationDataset): 
        ssv_id (int): segment id
        reverse_lookup (bool, optional): If True, map supervoxel id to super-supervoxel id. Defaults to False.

    Returns:
        bytes: encoded skeleton
    """    
    start = time.time()
    ssv = ssd.get_super_segmentation_object(int(ssv_id))

    if reverse_lookup:
        log_gate.debug(f"Reverse lookup for sv_id {ssv_id}")
        result = ssd.sv2ssv_ids(ids=np.array([ssv_id]))

        if result == {}:
            log_gate.info(f"Reverse lookup unsuccessful for sv_id {ssv_id}")
            return bytes()

        new_ssv_id = result[ssv_id]
        ssv = ssd.get_super_segmentation_object(int(new_ssv_id))
    
    ssv.load_skeleton()
    skeleton = ssv.skeleton
    if skeleton is None:
        return bytes()

    skel_attr = ["nodes", "edges", "diameters"]
    pred_key_ax = "{}_avg{}".format(ssv.config['compartments']['view_properties_semsegax']['semseg_key'],
                                    ssv.config['compartments']['dist_axoness_averaging'])
    pred_key_sp = ssv.config['spines']['semseg2mesh_spines']['semseg_key']
    keys = [
            global_params.config['compartments']['view_properties_semsegax']['semseg_key'],
            pred_key_ax,
            pred_key_ax + '_comp_maj',
            pred_key_sp,
            'myelin_avg10000',  # TODO: use global_params.py value !
            'myelin']  # TODO: use global_params.py value !

    for k in keys:
        if k in skeleton:
            skel_attr.append(k)
            if type(skeleton[k]) is list:
                skeleton[k] = np.array(skeleton[k])

        else:
            log_gate.warning("Couldn't find requested key in "
                                 "skeleton '{}'. Existing keys: {}".format(k, skeleton.keys()))

    skeleton = {k: skeleton[k].flatten().tolist() for k in
                skel_attr}

    dtime = time.time() - start
    log_gate.debug(f"Got ssv {ssv_id} skeleton after {dtime:.2f}s")

    nodes = np.array(skeleton["nodes"], dtype=np.float32).reshape(-1, 3)
    nodes[:, 0] *= ssd.scaling[0]
    nodes[:, 1] *= ssd.scaling[1]
    nodes[:, 2] *= ssd.scaling[2]

    edges = np.array(skeleton["edges"], dtype=np.uint32).reshape(-1, 2)
    num_vert = nodes.shape[0]
    num_edges = edges.shape[0]

    data = [
        np.uint32(num_vert),
        np.uint32(num_edges),
        nodes,
        edges
    ]
    
    encoded_skeleton = b''.join([array.tobytes('C') for array in data])

    return encoded_skeleton


def get_encoded_mesh(ssd: SuperSegmentationDataset, ssv_id: int, obj_type: str, reverse_lookup = False) -> bytes:
    """Gets encoded mesh for ssv_id.

    Args:
        ssd (SuperSegmentationDataset): 
        ssv_id (int): segment id
        obj_type (str): segmentation object (sv, mi, vc, syn_ssv)
        reverse_lookup (bool, optional): If True, map supervoxel id to super-supervoxel id. Defaults to False.

    Returns:
        bytes: encoded mesh
    """    
    start = time.time()
    ssv = ssd.get_super_segmentation_object(int(ssv_id))

    if reverse_lookup:
        log_gate.debug(f"Reverse lookup for sv_id {ssv_id}")
        result = ssd.sv2ssv_ids(ids=np.array([ssv_id]))

        if result == {}:
            log_gate.info(f"Reverse lookup unsuccessful for sv_id {ssv_id}")
            return bytes()

        new_ssv_id = result[ssv_id]
        ssv = ssd.get_super_segmentation_object(int(new_ssv_id))

    ssv.load_attr_dict()

    if obj_type == "sj":
        try:
            obj_type = "syn_ssv"
            _ = ssv.attr_dict[obj_type]  # try to query mapped syn_ssv objects
            log_gate.debug("Loading '{}' objects instead of 'sj' for SSV "
                            "{}.".format(obj_type, ssv_id))
        except KeyError:
            obj_type = "sj"

    mesh = ssv.load_mesh(obj_type)
    dtime = time.time() - start
    log_gate.debug(f"Got ssv {ssv_id} {obj_type} mesh after {dtime:.2f}s")

    indices = np.array(mesh[0], dtype=np.uint32).reshape(-1, 3)
    vertices = np.array(mesh[1], dtype=np.float32).reshape(-1, 3)
    num_vert = vertices.shape[0]

    data = [
        np.uint32(num_vert),
        vertices,
        indices
    ]

    encoded_mesh = b''.join([array.tobytes('C') for array in data])

    return encoded_mesh


def get_encoded_skeleton_old(backend: SyConnBackend, ssv_id: int, scales: Union[list, np.ndarray]) -> bytes:
    """Gets encoded skeleton for ssv_id. 

    Args:
        backend:
        ssv_id:
        scales: voxel size

    Returns:
        encoded skeleton or -1 (skeleton not available)
    """    

    log_gate.info('Getting binary encoded skeleton for ssv_id {}'.format(ssv_id))
    
    skeleton = {}
    
    try:
        start = time.time()
        skeleton = backend.ssv_skeleton(ssv_id)
        dtime = time.time() - start
        log_gate.debug('Got ssv skeleton {} after {:.2f}'.format(ssv_id, dtime))
        nodes = np.array(skeleton["nodes"], dtype=np.float32).reshape(-1, 3)

    except:
        return -1
        
    # accomodate dimension scaling
    nodes[:, 0] *= scales[0] # z
    nodes[:, 1] *= scales[1] # y
    nodes[:, 2] *= scales[2] # x
    
    edges = np.array(skeleton["edges"], dtype=np.uint32).reshape(-1, 2)
    num_vert = nodes.shape[0]
    num_edges = edges.shape[0]

    data = [
        np.uint32(num_vert),
        np.uint32(num_edges),
        nodes,
        edges
    ]

    encoded_skeleton = b''.join([array.tobytes('C') for array in data])

    return encoded_skeleton
    

def get_encoded_mesh_old(backend: SyConnBackend, ssv_id: int, obj_type: str):
    """Gets encoded mesh of SegmentationObject for ssv_id. 

    Args:
        backend: 
        ssv_id: 
        obj_type: SegmentationObject ('sv', 'mi', 'vc', 'syn_ssv')

    Returns:
        encoded mesh or -1 (mesh not available)
    """    

    log_gate.info('Getting binary encoded {} mesh {}'.format(obj_type, ssv_id))

    mesh = {}

    if obj_type == 'sv':
        try:
            mesh = backend.ssv_mesh(ssv_id)

        except:
            # logger.error('{} mesh not available for ssv_id: {}'.format(obj_type, ssv_id))
            return -1
    else:
        try:
            start = time.time()
            object_vert = backend.ssv_obj_vert(ssv_id, obj_type)
            object_ind = backend.ssv_obj_ind(ssv_id, obj_type)
            mesh['vertices'] = object_vert['vert']
            mesh['indices'] = object_ind['ind']
            dtime = time.time() - start
            log_gate.debug('Got {} mesh {} after {:.2f}'.format(obj_type, ssv_id, dtime))

        except:
            # logger.error('{} mesh not available for ssv_id: {}'.format(obj_type, ssv_id))
            return -1

    vertices = np.array(mesh['vertices'], dtype=np.float32).reshape(-1, 3)
    indices = np.array(mesh['indices'], dtype=np.uint32).reshape(-1, 3)
    num_vert = len(vertices)

    data = [
        np.uint32(num_vert),
        vertices,
        indices
    ]

    encoded_mesh = b''.join([array.tobytes('C') for array in data])
    return encoded_mesh


def get_mesh_meta(ssv_id: int, lod: int):
    """Gets mesh meta data for ssv_id.

    Args:
        ssv_id: 
        lod: level of detail

    Returns:
        json string of mesh meta data
    """    
    fragments = []
    fragments.append("{}:{}:{}_mesh".format(ssv_id, lod, ssv_id))
    meta = json.dumps({"fragments": fragments})

    return meta


def get_mean_mesh_areas(ssv_ids: Union[np.ndarray, list]) -> np.ndarray:
    """Retrieves the mean synapse mesh areas of ssvs. Same ordering as\
         :attr:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset.ssv_ids` 

    Args:
        ssvs: ssv ids (N,)

    Returns:
        mean mesh areas (N,)
    """
    mean_mesh_areas = []
    ssd = SuperSegmentationDataset()

    for ssv_id in ssv_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)
        if len(ssv.syn_ssv) == 0:
            mean_mesh_area = 0.0
        else:
            mean_mesh_area = np.mean([syn.mesh_area for syn in ssv.syn_ssv])

        mean_mesh_areas.append(mean_mesh_area)

    return np.array(mean_mesh_areas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve mean synapse mesh areas of the cells')
    parser.add_argument('--wd', type=str, help='Path to SuperSegmentationDataset', default="/ssdscratch/songbird/j0251/rag_flat_Jan2019_v3")
    args = parser.parse_args()
    
    global_params.wd = args.wd
    sd_syn_ssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir, cache_properties=('mesh_area', ))

    global ssd
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir, sd_lookup=dict(syn_ssv=sd_syn_ssv))

    mean_mesh_areas = np.concatenate(start_multiprocess_imap(get_mean_mesh_areas, params=list(basics.chunkify_successive(ssd.ssv_ids, 500)), nb_cpus=cpu_count()), axis=0)

    np.save(f"mean_mesh_areas.npy", mean_mesh_areas)


