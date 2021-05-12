from syconn.reps import super_segmentation_dataset as ss
from syconn.analysis.cli import configure_backend
from syconn.analysis.cli import SyConnClient
from syconn.analysis.utils import handle_layer_args
from syconn.handler.logger import log_main as log_gate
from syconn.handler.prediction import str2int_converter
from syconn import global_params
from timeit import default_timer as timer
import neuroglancer
import argparse
import os
import re
import numpy as np
import copy

logger = log_gate
PAGE_SIZE = 500

### query format ###
# celltype => only celltype
# >(or<)num_mito => only num_mito
# >(or<)ssv_size => only ssv_size
# _celltype_>(or<)num_mito_ => celltype and num_mito
# _celltype_>(or<)ssv_size_ => celltype and ssv_size
# _num_mito_>(or<)ssv_size_ => num_mito and ssv_size
# _celltype_>(or<)num_mito_>(or<)ssv_size_ => celltype, num_mito and ssv_size
# _#_>10_#
####################
def get_segmentation_layer(layers):
    for i, layer in enumerate(layers):
        if isinstance(layer.layer, neuroglancer.SegmentationLayer):
            return i, layer


class PropertyFilter(SyConnClient):
    """
    Filters ssv_ids based on the property (cell type, number of mitochondria, ssv size,...). The property is provided as a segment query in the 'seg.' tab of Neuroglancer viewer that triggers a state change. This state change is captured to retrieve a new set of ssv_ids. Finally a new viewer state is created with the segment query set to the retrieved ssv_ids. Following is the query format:
    _[celltype]_[>|<][mito_count]_[>|<][ssv_size]
    Args:
        backend (SyConnBackend): used for retrieving the skeleton, mesh and cell types
        seg_dataset (KnossosDataset): segmentation data for neuroglancer.LocalVolume
    """

    def __init__(self, backend, seg_path, organelles):
        super().__init__(backend, seg_path, organelles)

        self.gt_type = "ctgt"
        if 'j0251' in global_params.config.working_dir:
            self.gt_type = "ctgt_j0251_v2"
        self.CTs = backend.cts_in_data(self.gt_type)
        logger.info(f"Found [{self.CTs}] cell types in the dataset")

        self.ssd = ss.SuperSegmentationDataset(global_params.config.working_dir, sso_locking=False, sso_caching=True)
        self.ssv_ids = self.ssd.ssv_ids
        self.cur_message = None

        # dict of CT indices in the ssv_ids array
        self.CTmask = {ct: [] for ct in self.CTs}
        self.properties = ('mi', 'size')
        logger.info(f"Available filter properties {self.properties}")

        # don't forget to update here, if properties number increases
        self.master_pattern = r'^([a-zA-Z]{2,4}|#)((_((>|<)[0-9]+|#)){2})?_(pg[1-9][0-9]{0,3})$'
        self.ct_pattern = re.compile(r'^([a-zA-Z]{2,4})')
        self.property_filter = re.compile(r'_((>|<)[0-9]+)')
        self.page_pattern = re.compile(r'pg[1-9][0-9]{0,3}$')

        self.viewer.shared_state.add_changed_callback(self.on_state_changed)

    def on_state_changed(self):
        """Captures a state change and updates state."""
        ix, segmentation_layer = get_segmentation_layer(self.viewer.state.layers)
        segment_query = segmentation_layer.segment_query
        if segment_query != None and re.fullmatch(self.master_pattern, segment_query) != None:
            logger.info(f'Entered in query: {segment_query}')
            try:
                ct_match = next(self.ct_pattern.finditer(segment_query))
                celltype = ct_match.group(1)
                logger.info(f'Celltype {celltype}')
                if self.CTmask[celltype] == []:
                    logger.info(f"Storing ids mask of celltype {celltype} in memory")
                    start = timer()
                    self.CTmask[celltype] = np.where(
                        self.ssd.load_numpy_data('celltype_cnn_e3') == str2int_converter(celltype, self.gt_type))[0]
                    end = timer()
                    logger.info(f"Loaded celltype ids mask after {(end - start):.3f} seconds")
            except:
                ct_match = None
                logger.warning('No match found for celltype property!')

            # list of (prop_type, operation, threshhold)
            filter_list = []
            pageNr = -1

            split_seg_query = segment_query.split('_')
            filter_iter = self.property_filter.finditer(segment_query)
            prop_iter = iter(self.properties)

            #loop through property iterator
            while True:
                try:
                    # get the next item
                    element = next(filter_iter).group(1)
                    prop = next(prop_iter)
                    if element == '#':                                      # jump over missing condition
                        continue
                    filter_list.append((prop, element[0], element[1:]))

                except StopIteration:
                    # if StopIteration is raised, break from loop
                    break

            try:
                page_match = next(iter(self.page_pattern.finditer(segment_query)))
                pageNr = int(page_match.group(0).split("pg")[1])
            except:
                logger.info('No match found for page property')


            logger.info(f'Filter list: {filter_list}')
            ssv_ids = self.get_state_segment_ids(celltype, filter_list)

            if len(ssv_ids) == 0:
                return

            if isinstance(ssv_ids[0], np.ndarray):  # split happened

                if pageNr < 1:
                    logger.error("Numbering of the pages starts at 1.")
                    return

                if pageNr > len(ssv_ids):
                    pageNr = len(ssv_ids)

                page = ssv_ids[pageNr - 1]

            else:  # no split
                page = ssv_ids

            response = ', '.join(str(ssv_id) for ssv_id in page)
            new_state = copy.deepcopy(self.viewer.state)
            new_state.layers[ix].segment_query = response
            self.viewer.set_state(new_state)

            # message = f"Retrieved ssv_ids of cell type {ct_match} with mito count{} and ssv size{}".format(
            #     self.property_map["ct"], , self.property_map["ssv_size"])
            # if message != self.cur_message:
            #     with self.viewer.config_state.txn() as s:
            #         s.status_messages['status'] = message
            #     self.cur_message = message

        else:
            message = "Provide a property in the seg. tab search field to filter ssv ids."
            if message != self.cur_message:
                with self.viewer.config_state.txn() as s:
                    s.status_messages['status'] = message
                self.cur_message = message

    def get_state_segment_ids(self, celltype, filter_list):
        """Retrieves a subset of ssv_ids based on the query."""

        indices = self.CTmask[celltype]
        mask = np.ones(shape=(len(indices),), dtype=np.bool)
        logger.info(f"Indices  {indices}")
        for prop, op, thresh in filter_list:
            prop_array = self.ssd.load_numpy_data(prop)[indices]

            # check for mito
            if prop == 'mi':
                # TODO vectorize this
                prop_array = np.array([len(elem) for elem in prop_array])

            if op == ">":
                mask = np.logical_and(mask, (prop_array > int(thresh)))
            elif op == "<":
                mask = np.logical_and(mask, (prop_array < int(thresh)))

        ssv_ids_of_interest = self.ssv_ids[indices]
        ssv_ids_of_interest = ssv_ids_of_interest[mask]

        print(ssv_ids_of_interest)

        if len(ssv_ids_of_interest) > PAGE_SIZE:
            pages = self._split_pages(ssv_ids_of_interest)
            return pages
        else:
            return ssv_ids_of_interest

    def _split_pages(self, ssv_ids):
        pages = np.array_split(ssv_ids, len(ssv_ids) // PAGE_SIZE)
        num_pages = len(pages)
        message = f"Result is split across {num_pages} pages. Append _pg<number> at the end of query to access the other pages!"
        if message != self.cur_message:
            with self.viewer.config_state.txn() as s:
                s.status_messages['status'] = message
            self.cur_message = message
        return pages


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
    pf = PropertyFilter(backend, seg_path, args.organelles)
    print(pf.viewer)
