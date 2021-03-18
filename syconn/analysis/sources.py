import neuroglancer
import numpy as np

from syconn.handler.logger import log_main as log_gate

class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    """
    Overloads the neuroglancer.skeleton.SkeletonSource. Implements get_skeleton()

    Args:
        dimensions: neuroglancer.CoordinateSpace 
        backend: syconn.analysis.backend.SyConnBackend 
    """

    def __init__(self, dimensions, backend):
        super(SkeletonSource, self).__init__(dimensions)
        self.backend = backend

    def get_skeleton(self, object_id):
        """
        Creates a skeleton object from vertices and edges

        :param object_id: int (ssv_id)
        :return neuroglancer.skeleton.Skeleton (parsed in the SkeletonHandler)
        """
        skeleton = self.backend.ssv_skeleton(object_id)
        nodes = np.array(skeleton["nodes"]).reshape(-1, 3)[:, [2, 1, 0]]  # change to (z,y,x) order
        edges = np.array(skeleton["edges"]).reshape(-1, 2)
        return neuroglancer.skeleton.Skeleton(
            vertex_positions=nodes,
            edges=edges
        )

class MeshSource(neuroglancer.mesh.MeshSource):
    def __init__(self, dimensions, backend, object_type):
        super(MeshSource, self).__init__(dimensions)
        self.backend = backend
        self.object_type = object_type

    def get_mesh(self, object_id):
        mesh = {}
        logger = log_gate
        
        if self.object_type == 'sv':
            try:
                mesh = self.backend.ssv_mesh(object_id)
            except:
                logger.error('Precomputed mesh not available for ssv_id: {}'.format(object_id))
        else:
            try:
                object_vert = self.backend.ssv_obj_vert(object_id, self.object_type)
                object_ind = self.backend.ssv_obj_ind(object_id, self.object_type)
            except:
                logger.error('Precomputed mesh not available for ssv_id: {}'.format(object_id))

            mesh['vertices'] = object_vert['vert']
            mesh['indices'] = object_ind['ind']

        if not mesh:
            logger.error('Mesh could not be built for given object_id: {}'.format(object_id))

        vertices = np.array(mesh['vertices'], dtype=np.float32).reshape(-1, 3)[:, [2, 1, 0]] * 1e-9
        indices = np.array(mesh['indices'], dtype=np.uint32).reshape(-1, 3)
        num_vert = len(vertices)

        data = [
            np.uint32(num_vert),
            vertices,
            indices
        ]
        encoded_mesh = b''.join([array.tobytes('C') for array in data])

        return encoded_mesh