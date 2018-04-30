import numpy as np
from syconn.proc.meshes import id2rgb_array, rgb2id_array


if __name__ == "__main__":
    vertex_ids = np.arange(100, dtype=np.uint32)[:, None]
    rgb_arr = id2rgb_array(vertex_ids)
    remapped_vertex_ids = rgb2id_array(rgb_arr)
    assert np.all(vertex_ids == remapped_vertex_ids), \
        "Unique vertex mapping failed."
