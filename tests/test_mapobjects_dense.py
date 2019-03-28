from knossos_utils import chunky
from syconn import global_params
from syconn.handler.basics import kd_factory


def _map_cellobjects_dense(args):
    return 0
    # chunk = args[0]
    # seg_cell = kd.from_overlaycubes_to_matrix(offset=chunk.coordinates,
    #                                           size=chunk.box_size)
    # cube_cell = ...
    # cubes_co = []

def map_ids(ref_ids, other_ids_list):
    #"""TODO: CYTHONIZE"""
    cube_cell = ref_ids
    res_map_dc = {}
    cube_shape = cube_cell.shape
    for x in range(cube_shape[0]):
        for y in range(cube_shape[1]):
            for z in range(cube_shape[2]):
                cell_id = cube_cell[x, y, z]
                for cube_co in other_ids_list:
                    break
                    #print(other_ids_list[cube_co])
                    #print(cube_co)
    return


if __name__ == "__main__":
    dictionary_elements = []
    seg_dict = []
    dictionary_elements.append("mi")
    dictionary_elements.append("vc")
    dictionary_elements.append("sj")
    #dictionary_elements.append("sv")
    #dictionary_elements.append("cs")
    print(global_params.wd)
    global_params.wd = '/wholebrain/u/atultm/SyConn/example_cube1/'
    cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"
    chunk_size = [128] * 3
    kd = kd_factory(global_params.config.kd_seg_path)
    cd_cell = chunky.ChunkDataset()
    cd_cell.initialize(kd, kd.boundary, chunk_size, cd_dir,
                       box_coords=[0, 0, 0], fit_box_size=True)

    ch = cd_cell.chunk_dict[0]

    seg_cell = kd.from_overlaycubes_to_matrix(offset=ch.coordinates,
                                              size=ch.box_size)
    for element in dictionary_elements:
        cd_dir = global_params.config.working_dir + "chunkdatasets/" + element + "/"
        # Class that contains a dict of chunks (with coordinates) after initializing it
        cd_mi = chunky.ChunkDataset()
        cd_mi.initialize(kd, kd.boundary, chunk_size, cd_dir,
                      box_coords=[0, 0, 0], fit_box_size=True)
        ch = cd_mi.chunk_dict[0]
        input_file_folder = element + "_stitched_components"
        seg_dict.append(ch.load_chunk(name=input_file_folder))
        #seg_dict.update({element: ch.load_chunk(name=input_file_folder)})

    cd_dir = global_params.config.working_dir + "chunkdatasets/vc/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd_vc = chunky.ChunkDataset()
    cd_vc.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    ch = cd_vc.chunk_dict[0]
    seg_vc = ch.load_chunk(name="vc_stitched_components")
    #for :
    print(seg_cell)
    print(seg_dict)
    print(len(seg_cell))
    print(len(seg_vc))
    print(seg_dict[0])
    print(len(seg_dict[0]))

    #print((seg_dict[0].shape))
    map_ids(seg_cell, seg_dict)

