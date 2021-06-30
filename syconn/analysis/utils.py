from syconn.handler.logger import log_main as logger
import numpy as np

def get_encoded_skeleton(backend, ssv_id, scales):
    """Gets encoded skeleton for ssv_id.

    :param backend: 
    :type backend: SyConnBackend
    :param ssv_id: 
    :type ssv_id: int
    :param scales: KnossosDataset.scale
    :type scales: numpy.array
    :return encoded_skeleton:
    :rtype encoded_skeleton: bytes, -1 (skeleton not available)
    """

    logger.info('Getting binary encoded skeleton for ssv_id {}'.format(ssv_id))
    
    skeleton = {}
    
    try:
        skeleton = backend.ssv_skeleton(ssv_id)
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

def get_encoded_mesh(backend, ssv_id, obj_type):
    """Gets encoded mesh of a specific obj type for ssv_id.

    :param backend: 
    :type backend: SyConnBackend
    :param ssv_id: 
    :type ssv_id: int
    :param obj_type: 'sv', 'mi', 'vc', 'sj'
    :type obj_type: str
    :return encoded_mesh: 
    :rtype encoded_mesh: bytes, -1 (mesh not available)
    """

    logger.info('Getting binary encoded {} mesh for ssv_id {}'.format(obj_type, ssv_id))

    mesh = {}

    if obj_type == 'sv':
        try:
            mesh = backend.ssv_mesh(ssv_id)

        except:
            # logger.error('{} mesh not available for ssv_id: {}'.format(obj_type, ssv_id))
            return -1
    else:
        try:
            object_vert = backend.ssv_obj_vert(ssv_id, obj_type)
            object_ind = backend.ssv_obj_ind(ssv_id, obj_type)
            mesh['vertices'] = object_vert['vert']
            mesh['indices'] = object_ind['ind']

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

def handle_layer_args(ap):
    g = ap.add_argument_group(title='SyConn layer options')
    g.add_argument('--organelles', nargs='+', default=[],
                        help='Organelle layers to be displayed')
