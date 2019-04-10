from knossos_utils import chunky
from syconn import global_params
from syconn.handler.basics import kd_factory
import numpy as np
from syconn.handler.basics import chunkify
from syconn.mp import batchjob_utils as qu, mp_utils as sm



def map_ids(wd, n_max_co_processes=None, chunk_size=(128, 128, 128), debug=False):

    kd = kd_factory(global_params.config.kd_seg_path)
    cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"

    cd_cell = chunky.ChunkDataset()
    cd_cell.initialize(kd, kd.boundary, chunk_size, cd_dir,
                                           box_coords=[0, 0, 0], fit_box_size=True)

    multi_param = []
    for key in cd_cell.chunk_dict:
        multi_param.append([cd_cell.chunk_dict[key].coordinates, chunk_size, wd])


    _map_ids_thread((0,0,0), chunk_size, wd)
    #chunkified_multi_param = chunkify(multi_param, 2000)

    # sm.start_multiprocess_imap(_map_ids_thread, multi_param,
    #                                          nb_cpus=n_max_co_processes, verbose=debug, debug=debug)


def _map_ids_thread(args):
    coord = args[0]
    chunk_size = args[1]
    wd = args[2]

    kd_cell = kd_factory(global_params.config.kd_seg_path)
    seg_cell = kd_cell.from_overlaycubes_to_matrix(offset=coord, size=chunk_size)

    kd = {}
    feat = global_params.config.kd_paths

    for obj in feat:
        if obj == "kd_seg":
            continue
        kd[obj] = kd_factory(feat[obj]).from_overlaycubes_to_matrix(
                                            offset=coord, size=chunk_size)

    cell_ids, cell_size = np.unique(seg_cell, return_counts=True)
    print("cell_ids= ", cell_ids, "cell_size= ", cell_size)

    for cell_id in cell_ids:
        cache_dc = CompressedStorage(wd + "/cache_" + cell_id + ".pkl")





def main():
    map_ids(wd="/wholebrain/u/mariakaw/SyConn/example_cube1/dict")

    # kd_paths = global_params.config.kd_paths
    # kd = []
    # cd_cell = {}
    # for path in kd_paths:
    #     kd.append(kd_factory(kd_paths[path]))
    #
    #
    # chunk_size = [129] * 3
    # cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"
    # # print("cd_dir= ", cd_dir)
    #
    # cd_cell = chunky.ChunkDataset()

    # cd_cell.initialize(kd[1], kd[1].boundary, chunk_size, cd_dir,
    #                                        box_coords=[0, 0, 0], fit_box_size=True)
    #
    # # for key in cd_cell.chunk_dict:
    # #     print("key= ", key)
    # #     ch = cd_cell.chunk_dict[key]
    # #     print("ch.coordinates= ", ch.coordinates)
    # #     print("ch.size= ", ch.size)
    #
    # ch = cd_cell.chunk_dict[0]
    # seg_cell = kd[0].from_overlaycubes_to_matrix(offset=ch.coordinates,
    #                                           size=chunk_size)
    #
    # print("\n seg_cell = ", seg_cell)
    #
    # # map_ids('sv', '/wholebrain/songbird/j0126/areaxfs_v6/')
    #
    #
    #
    # print(global_params.wd)
    # global_params.wd = '/wholebrain/u/atultm/SyConn/example_cube1/'
    # cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"
    # chunk_size = [128] * 3
    # kd_dict = dict()
    # kd = kd_factory(global_params.config.kd_seg_path)
    # cd_cell = chunky.ChunkDataset()
    # cd_cell.initialize(kd, kd.boundary, chunk_size, cd_dir,
    #                    box_coords=[0, 0, 0], fit_box_size=True)
    #
    # ch = cd_cell.chunk_dict[0]
    #
    # seg_cell = kd.from_overlaycubes_to_matrix(offset=ch.coordinates,
    #                                           size=ch.size)
    # # for element in dictionary_elements:
    # #     cd_dir = global_params.config.working_dir + "chunkdatasets/" + element + "/"
    # #     # Class that contains a dict of chunks (with coordinates) after initializing it
    # #     cd_mi = chunky.ChunkDataset()
    # #     cd_mi.initialize(kd, kd.boundary, chunk_size, cd_dir,
    # #                   box_coords=[0, 0, 0], fit_box_size=True)
    # #     ch = cd_mi.chunk_dict[0]
    # #     input_file_folder = element + "_stitched_components"
    # #     seg_dict.append(ch.load_chunk(name=input_file_folder))
    # #     #seg_dict.update({element: ch.load_chunk(name=input_file_folder)})
    #
    # cd_dir = global_params.config.working_dir + "chunkdatasets/vc/"
    # # Class that contains a dict of chunks (with coordinates) after initializing it
    # cd_vc = chunky.ChunkDataset()
    # cd_vc.initialize(kd, kd.boundary, chunk_size, cd_dir,
    #               box_coords=[0, 0, 0], fit_box_size=True)
    # seg_vc = cd_vc.from_chunky_to_matrix(chunk_size, offset, name='vc_stitched_components', setnames=["vc", ])['vc']
    #
    # cube_shape = seg_cell.shape
    # #for x in range(cube_shape[0]):
    #  #   for y in range(cube_shape[1]):
    #   #      for z in range(cube_shape[2]):
    #    #         cell_id = [x, y, z]
    #             #print(cell_id)
    # #print(cell_id)
    # #print(len(cell_id))
    # #print(cell_id.shape)
    # #for :
    # print(seg_cell)
    # print(seg_vc)
    # #print(len(seg_cell))
    # #print(len(seg_vc))
    # #print(seg_dict[0])
    # #print(len(seg_dict[0]))
    #
    # #print((seg_dict[0].shape))
    # res = map_ids(seg_cell, [seg_vc], dictionary_elements)
    #
if __name__ == "__main__":
    main()