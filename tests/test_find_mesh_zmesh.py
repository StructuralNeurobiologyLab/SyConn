from zmesh import Mesher
import numpy as np
from syconn.handler import basics
from syconn import global_params
from syconn.proc.sd_proc import find_meshes, merge_meshes_dict
from syconn.proc.meshes import write_meshes2kzip


def load_seg_data(offset):
    global_params.wd = "/u/mariakaw/SyConn/example_cube1"
    kd_cell = basics.kd_factory(global_params.config.kd_seg_path)
    return kd_cell.from_overlaycubes_to_matrix((256, 256, 256), offset)


def test_mesher():
    chunk1 = load_seg_data((0, 0, 0))
    mesh1 = find_meshes(chunk1, (0, 0, 0))

    chunk2 = load_seg_data((0, 0, 256))
    mesh2 = find_meshes(chunk2, (0, 0, 256))

    merge_meshes_dict(mesh1, mesh2)

    ind = []
    vert = []
    norm = []
    col =[]
    ply_fname = []
    for key, val in mesh1.items():
        ind.append(val[0])
        vert.append(val[1])
        norm.append(val[2])
        col.append(None)
        ply_fname.append("{}.ply".format(key))

    write_meshes2kzip("tets_mesh_cube1.k.zip", ind, vert, norm, col, ply_fname)


if __name__ == "__main__":
    test_mesher()