from syconn.reps import segmentation
from syconn.analysis.cli import configure_backend
from syconn.analysis.sources import SkeletonSource
from syconn.analysis.cli import SyConnClient
from syconn.analysis.utils import handle_layer_args
from syconn.handler.logger import log_main as log_gate
from syconn import global_params
import neuroglancer
import argparse
import os
import copy
from timeit import default_timer as timer
import numpy as np
from knossos_utils import KnossosDataset
import re
import timeit

reg = r'(.*)_pg[0-9]'
logger = log_gate

def get_segmentation_layer(layers):
    '''
    Gets the first segmentation layer in a state
    :param layers: neuroglancer.Viewer().state
    :return:
    '''
    for i, layer in enumerate(layers):
        if isinstance(layer.layer, neuroglancer.SegmentationLayer):
            return i, layer

class CellTypeSelector(SyConnClient):
    '''
    Client to query segment ids corresponding to specific cell type with paging
    '''
    def __init__(self, backend, seg_path, organelles):
        super().__init__(backend, seg_path, organelles)
        self.states = []
        self.state_index = 0
        self.cur_message = None

        if os.path.basename(os.path.dirname(global_params.wd)) == 'j0251':
            self.CTs = ('STN', 'DA', 'MSN', 'LMAN', 'HVC', 'TAN', 'GPe', 'GPi', 'FS', 'LTS', 'NGF')
        else:
            self.CTs = ('EA', 'MSN', 'GP', 'INT')

        self.pages = {ct: [] for ct in self.CTs}

        self.viewer.shared_state.add_changed_callback(self.on_state_changed)
        # experimental: self.viewer.defer_callback(self.on_state_changed)


    def on_state_changed(self):
        ix, segmentation_layer = get_segmentation_layer(self.viewer.state.layers)
        segment_query = segmentation_layer.segment_query

        if not re.match(reg, segment_query) or segment_query.split('_')[0] not in self.CTs:
            message = '[No cell type selected]'
        else:
            message = '[{}]'.format(segmentation_layer.segment_query)
            self.state_index += 1

            new_state = self.set_state_segment_ids(self.viewer.state)
            if new_state is None:
                return

            self.viewer.set_state(new_state)

        if message != self.cur_message:
            with self.viewer.config_state.txn() as s:
                if message is not None:
                    s.status_messages['status'] = message
                else:
                    s.status_messages.pop('status')
            self.cur_message = message

    def set_state_segment_ids(self, state):
        '''
        Creates a deepcopy of the current state and sets the desired segment ids as the query of the new state
        :param state:
        :return: neuroglancer.Viewer().state the new state with the desired segment ids
        '''
        new_state = copy.deepcopy(self.viewer.state)
        ix, segmentation_layer = get_segmentation_layer(new_state.layers)

        split_query = segmentation_layer.segment_query.split('_')

        # check if already gotten celltype, load it if not cached
        if not len(self.pages[split_query[0]]):
            logger.info(f"Loading celltype {split_query[0]} ids into memory")
            start = timer()
            ssv_ids = self.backend.ssvs_of_ct(split_query[0])
            self.pages[split_query[0]] = ssv_ids
            end = timer()
            logger.info(f"Loaded celltype ids after {(end - start):.3f} seconds")

        # check if dict value is list of lists
        if isinstance(self.pages[split_query[0]][0], np.ndarray):
            pageNr = int(split_query[1].split('pg')[1])

            if pageNr < 1:
                logger.error("Numbering of the pages starts at 1.")
                return None

            ids = self.pages[split_query[0]][pageNr-1]                              # numbering of pages starts from 1
        else:
            ids = self.pages[split_query[0]]

        query = ', '.join(str(ssv_id) for ssv_id in ids)
        segmentation_layer.segment_query = query
        return new_state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    handle_layer_args(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    global backend
    if args.wd == '':
        logger.error('No working directory selected... Aborting')
    global_params.wd = os.path.expanduser(args.wd)
    backend = configure_backend()
    # load seg and raw data
    seg_path = global_params.config.kd_seg_path
    ct = CellTypeSelector(backend, seg_path, args.organelles)
    print(ct.viewer)
