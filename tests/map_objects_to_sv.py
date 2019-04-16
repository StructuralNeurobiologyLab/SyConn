from knossos_utils import chunky
from syconn import global_params
from syconn.handler.basics import kd_factory
import numpy as np
from collections import defaultdict
from syconn.handler.basics import chunkify
from syconn.mp import batchjob_utils as qu, mp_utils as sm
from sys import getsizeof
import sys
from syconn.backend.storage import VoxelStorage, AttributeDict, VoxelStorageDyn
import pickle


def map_ids(wd, n_jobs=1000, qsub_pe=None, qsub_queue=None, nb_cpus=None,
            n_max_co_processes=None, chunk_size=(128, 128, 128), debug=False):

    global_params.wd = wd
    kd = kd_factory(global_params.config.kd_seg_path)

    cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"

    cd_cell = chunky.ChunkDataset()
    cd_cell.initialize(kd, kd.boundary, chunk_size, cd_dir,
                       box_coords=[0, 0, 0], fit_box_size=True)

    multi_params = []
    chunkify_id = 0
    for coord_chunk in chunkify([cd_cell.chunk_dict[key].coordinates for key in cd_cell.chunk_dict], 100):
        multi_params.append([coord_chunk, chunk_size, wd, chunkify_id])
        chunkify_id += 1

    sm.start_multiprocess_imap(_map_ids_thread, multi_params,
                               nb_cpus=n_max_co_processes, verbose=debug, debug=debug)


def _map_ids_thread(args):
    coord_list = args[0]
    chunk_size = args[1]
    wd = args[2]
    chunkify_id = args[3]

    worker_sv_dc = {}
    kd_obj = {}
    small_dc = {}
    stri = wd + '/voxel_%s.pkl' %chunkify_id
    f = open(stri, "wb")


    for obj_type in global_params.existing_cell_organelles:
        small_dc[obj_type] = {}
        kd_obj[obj_type] = kd_factory(global_params.config.entries['Paths']['kd_%s' %obj_type])

    kd_cell = kd_factory(global_params.config.kd_seg_path)

    for coord in coord_list:
        seg_cell = kd_cell.from_overlaycubes_to_matrix(offset=coord, size=chunk_size).flatten()

        seg_obj = {}
        for obj in kd_obj:
            # seg_obj[obj] = kd_obj[obj].from_overlaycubes_to_matrix(offset=coord, size=chunk_size).flatten()
            seg_obj[obj] = create_toy_data(chunk_size, 3).flatten()

        for unique_cell_id in np.unique(seg_cell):
            if unique_cell_id in worker_sv_dc:
                continue
            worker_sv_dc[unique_cell_id] = small_dc

        for vox in range(len(seg_cell)):
            cell_id = seg_cell[vox]

            for obj in kd_obj:
                j = seg_obj[obj][vox]
                if j in worker_sv_dc[cell_id][obj]:
                    worker_sv_dc[cell_id][obj][j] += 1
                else:
                    worker_sv_dc[cell_id][obj][j] = 1

    pickle.dump(worker_sv_dc, f)
    f.close()


def create_toy_data(m_size, moduloo):
    # np.random.seed(0)
    matrix = np.zeros(shape=m_size, dtype=int)
    for i in range(m_size[0]):
        for j in range(m_size[1]):
            for k in range(m_size[2]):
                matrix[i, j, k] = np.random.randint(moduloo, size=1)
    return matrix



def main():

    map_ids(wd="/wholebrain/u/mariakaw/SyConn/example_cube1/")

    # _map_ids_thread([[[0, 0, 0]], (128, 128, 128)])

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