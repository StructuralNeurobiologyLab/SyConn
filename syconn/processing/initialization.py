from knossos_utils import chunky
import numpy as np


def initialize_cset(kd, home_path, chunksize):
    """Initializes the chunkdataset

    :param kd: knossodataset
    :param home_path: str path to head
    :param chunksize: arr dimensions of the chunks
    """

    cset = chunky.ChunkDataset()
    cset.initialize(kd, kd.boundary, chunksize, home_path + "/chunkdataset/",
                    box_coords=[0, 0, 0], fit_box_size=True)
    chunky.save_dataset(cset)

    return cset


