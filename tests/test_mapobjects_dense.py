from knossos_utils import chunky
from syconn import global_params
from syconn.handler.basics import kd_factory
import numpy as np
from syconn.reps.find_object_properties_C import find_object_propertiesC, map_subcell_extract_propsC, map_subcell_C, cython_loop
from syconn.reps.rep_helper import _find_object_properties
import time


def python_loop(chunk):
    sizes = {}
    for x in range(chunk.shape[0]):
        for y in range(chunk.shape[1]):
            for z in range(chunk.shape[2]):
                key = chunk[x, y, z]
                if key == 0:
                    continue
                if key in sizes:
                    sizes[key] += 1
                else:
                    sizes[key] = 1
    return sizes


def timeit(func, args):
    for ii in range(3):
        start = time.time()
        _ = func(args)
        end = time.time()
        print("{} took {:.2f}s".format(str(func), end-start))


def timeit2(func, arg1, arg2):
    for ii in range(3):
        start = time.time()
        _ = func(arg1, arg2)
        end = time.time()
        print("{} took {:.2f}s".format(str(func), end-start))


if __name__ == "__main__":
    edge_s = 50
    toy = np.random.randint(low=0, size=edge_s**3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint64)
    # timeit(python_loop, toy)
    timeit(cython_loop, toy)
    timeit(find_object_propertiesC, toy)
    timeit(_find_object_properties, toy)
    # timeit2(map_subcell_C, toy, toy[None, ])
    # timeit2(map_subcell_extract_propsC, toy, toy[None, ])
    # dictionary_elements = []
    # seg_dict = []
    # dictionary_elements.append("mi")
    # dictionary_elements.append("vc")
    # dictionary_elements.append("sj")
    # offset = (10, 10, 10)
    # print(global_params.wd)
    # global_params.wd = '/wholebrain/u/atultm/SyConn/example_cube1/'
    # cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"
    # chunk_size = [128] * 3
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
    # res = map_ids(seg_cell, [seg_vc])

