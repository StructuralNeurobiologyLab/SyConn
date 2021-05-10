from syconn import global_params
from syconn.analysis.storage import MeshStorage
from syconn.handler.logger import log_main as logger
from syconn import global_params
import json
import numpy as np
import threading as th
import os
import shutil
from knossos_utils import KnossosDataset
from neuroglancer.chunks import encode_npz

def get_encoded_skeleton(backend, ssv_id, scales):
    """
    Get encoded skeleton for ssv_id.
    :param backend: SyConnBackend object initialized with 'global_params.config.working_dir'
    :param ssv_id: int
    :param scales: np.array (KnossosDataset.scale)
    :return: bytes
    """
    logger.info('Getting binary encoded skeleton for ssv_id {}'.format(ssv_id))
    
    skeleton = {}
    
    try:
        skeleton = backend.ssv_skeleton(ssv_id)
    except:
        logger.error('Skeleton not available for ssv_id: {}'.format(ssv_id))

    if skeleton == {}:
        print('empty skeleton')
        logger.error('Skeleton could not be retrieved for ssv_id: {}'.format(ssv_id))
        
    nodes = np.array(skeleton["nodes"], dtype=np.float32).reshape(-1, 3)

    # accomodate dimension scaling
    nodes[:, 0] *= scales[0]
    nodes[:, 1] *= scales[1]
    nodes[:, 2] *= scales[2]
    
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

def get_encoded_mesh(backend, ssv_id, obj_type):
    """
    Get encoded mesh of a specific obj type for ssv_id.
    :param ssv_id: int
    :param obj_type: str
    :return: bytes
    """
    logger.info('Getting binary encoded {} mesh for ssv_id {}'.format(obj_type,
                                                                      global_params.config.working_dir.split('/')[-1]))
    mesh = {}
    if obj_type == 'sv':
        try:
            mesh = backend.ssv_mesh(ssv_id)
        except:
            logger.error('{} mesh not available for ssv_id: {}'.format(obj_type, ssv_id))
    else:
        try:
            object_vert = backend.ssv_obj_vert(ssv_id, obj_type)
            object_ind = backend.ssv_obj_ind(ssv_id, obj_type)
        except:
            logger.error('Precomputed mesh not available for ssv_id: {}'.format(obj_type))
        mesh['vertices'] = object_vert['vert']
        mesh['indices'] = object_ind['ind']
    if not mesh:
        logger.error('Mesh could not be retrieved for ssv_id: {}'.format(ssv_id))
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

def handle_layer_args(ap):
    g = ap.add_argument_group(title='SyConn layer options')
    g.add_argument('--organelles', nargs='+', default=[],
                        help='Organelle layers to be displayed')
