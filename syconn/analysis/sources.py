import numpy as np
import neuroglancer.skeleton

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