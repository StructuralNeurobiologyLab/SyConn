import argparse
import os
import re
import numpy as np
import copy
import time
from timeit import default_timer as timer
from bs4 import BeautifulSoup as Soup
from knossos_utils import KnossosDataset

from syconn.analysis.cli import configure_backend
from syconn.analysis.cli import SyConnClient
from syconn.analysis.utils import handle_layer_args
from syconn.handler.logger import log_main as log_gate
from syconn.handler.prediction import str2int_converter, int2str_converter
from syconn import global_params

import neuroglancer
import neuroglancer.cli
from neuroglancer.viewer_config_state import SegmentIdMapEntry

logger = log_gate

def get_segmentation_layer(layers):
    for i, layer in enumerate(layers):
        if isinstance(layer.layer, neuroglancer.SegmentationLayer):
            return i, layer

class PropertyFilter(SyConnClient):
    """Invokes SyConnClient with ssv_ids filtering based on the properties 
    -- cell type, mitochondria count, ssv size, synapse.

    Cell type, mitochondria count and ssv size are provided as a segment query
    in the 'seg.' tab of Neuroglancer viewer that triggers a state change. 
    This is captured to retrieve a new set of ssv_ids. Finally a new viewer
    state is created with the segment query set to the new ssv_ids. 
    Following are the query formats:

    - [celltype]_pg[page_num] -> filters by the cell type and divides the
        resulting ssv_ids across pages (cached)
    - [celltype]_[>|<][mito_count]_[>|<][ssv_size]_pg[page_num] -> filters
        by the cell type, mitochondria count and ssv size, and divides the
        resulting ssv_ids across pages (if required)  
    
    To filter with the synapse property, select a segment (ssv_id) and press
    key 'p'. This will render the synaptic partner in terms of synapse area, 
    probability and axon -> dendrite and axon -> soma connections, and
    center the viewer at the synapse position (representation coordinates).

    :cvar properties: properties with operator-operand pair
    :type properties: tuple
    :cvar master_pattern: query format for specifying all the properties
    :type master_pattern: str
    :cvar ct_pattern: cell type format
    :type ct_pattern: re.Pattern
    :cvar property_filter: properties format with operator and value
    :type property_filter: re.Pattern
    :cvar page_pattern: page format
    :type page_pattern: re.Pattern
    :cvar PAGE_SIZE: max number of ssv_ids displayed in the side panel
    :type PAGE_SIZE: int
    :cvar syn_type: maps compartment to pre- or post-synaptic
    :type syn_type: dict

    :param backend: to retrieve the skeleton, mesh and cell types
    :type backend: SyConnBackend
    :param seg_dataset: segmentation data for neuroglancer.LocalVolume
    :type seg_dataset: KnossosDataset
    :param organelles: command line argument; supported organelles (mi, vc, sj)
    :type organelles: list
    """

    properties = ('mi', 'size')
    # don't forget to update here, if properties number increases
    # master_pattern = r'^([a-zA-Z]{2,4}|#)((_((>|<)[0-9]+|#)){2})?_(pg[1-9][0-9]{0,3})$'
    master_pattern = r'^([a-zA-Z]{2,4})?_?(mito[>|<|=][0-9][0-9]{0,3})?_?(size[>|<][1-9][0-9]{1,6})?_?(pg[1-9][0-9]{0,3})?$' # TODO(hashir): Independent property querying 
    ct_pattern = re.compile(r'^([a-zA-Z]{2,4})')
    property_filter = re.compile(r'_((>|<)[0-9]+)')
    page_pattern = re.compile(r'pg[1-9][0-9]{0,3}$')
    PAGE_SIZE = 500
    syn_type = {
        "dendrite": "post-synaptic",
        "axon": "pre-synaptic",
        "soma": "post-synaptic",
    }

    def __init__(self, params, organelles, token=None):
        super().__init__(params, organelles, token)

        self.params = params
        self.version = params.version
        self._token = token

        self.gt_type = "ctgt"
        if params.acquisition == "j0251":
            self.gt_type = "ctgt_j0251_v2"
        
        self.CTs = params["backend"].cts_in_data(self.gt_type)
        logger.info(f"Found {self.CTs} cell types in the dataset")
        
        self.ssv_ids = params["ssvs"]
        self.cur_message = None

        # dict of CT indices in the ssv_ids array
        self.CTmask = {ct: [] for ct in self.CTs}
        logger.info(f"Available filter properties {self.__class__.properties}")

        # add action handler for finding synaptic partner
        self.viewer.actions.add('show-largest-synaptic-connection', self._handle_select)

        # bind action to key
        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.data_view['keyp'] = 'show-largest-synaptic-connection'
        
        # defer callback necessary to avoid deadlock 
        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_state_changed)
        )

    @property
    def token(self):
        return self._token

    def _handle_select(self, action_state):
        """Action handler for synaptic filtering [keyp]
        
        :param action_state: current state of the viewer
        :type action_state: neuroglancer.viewer_config_state.ActionState
        :return:
        """

        segment_id = action_state.selected_values.get(self.seg_name) # super().seg_name

        if segment_id is None: 
            return
        
        ssv_id = segment_id.value

        # if segment_id.value is a collections.namedtuple (key=ssv_id, value=None, label=celltype), extract the ssv_id
        if isinstance(ssv_id, SegmentIdMapEntry):
            ssv_id = ssv_id[0]
        
        message = 'Loading synaptic partner for selected ssv {}'.format(ssv_id)

        if message != self.cur_message:
            with self.viewer.config_state.txn() as cfs:
                cfs.status_messages['status'] = message
            self.cur_message = message

        with self.viewer.txn() as s:
            segments = get_segmentation_layer(s.layers)[1].segments
            if ssv_id in segments:
                start = time.time()
                result = self.get_synaptic_partner(ssv_id)
                dtime = time.time() - start
                logger.debug('Got synaptic partner after {:.2f}'.format(dtime))

                # if no synaptic partner is found
                if result == -1:
                    message = 'No synaptic partner found for the selected ssv {}'.format(ssv_id)

                    if message != self.cur_message:
                        with self.viewer.config_state.txn() as cfs:
                            cfs.status_messages['status'] = message
                        self.cur_message = message

                    return

                # get synaptic partner id, compartment predictions and celltypes
                partner_ssv_id = result["partner_ssv"]
                ssv_comp = int2str_converter(result["ssv_comp"], "axgt").split('_')[1]
                partner_ssv_comp = int2str_converter(result["partner_ssv_comp"], "axgt").split('_')[1]
                ssv_ct = int2str_converter(result["ssv_ct"], self.gt_type)
                partner_ssv_ct = int2str_converter(result["partner_ssv_ct"], self.gt_type)
                rep_coords = result["rep_coords"]

                # add synaptic partner to segment list
                segments.add(partner_ssv_id)
                # s.position = np.flip(rep_coords) # use for neuroglancer.LocalVolume source
                s.position = rep_coords # set position to the representative coordinate

                # TODO(hashir): use bootstrap alert instead of status messages
                # soup = Soup(open("/home/hashir/neuroglancer/python/neuroglancer/static/index.html"))
                # container = soup.find(id="neuroglancer-container")
                # alert_div = soup.new_tag('div')
                # alert_div['class'] = "alert alert-success mb-0"
                # alert_div.string = "abcd"
                # container.insert_before(alert_div)
                # with open("/home/hashir/neuroglancer/python/neuroglancer/static/index.html", "w") as f:
                #     f.write(str(soup))
                
                # pre-synaptic -> post-synaptic message format
                if self.__class__.syn_type[ssv_comp] == "pre-synaptic":
                    message = f'{ssv_id}: {ssv_ct} \
                        {self.__class__.syn_type[ssv_comp]} \
                            ({ssv_comp}) \
                        -> {partner_ssv_id}: {partner_ssv_ct} \
                            {self.__class__.syn_type[partner_ssv_comp]} \
                            ({partner_ssv_comp})'
                else:
                    message = f'{partner_ssv_id}: {partner_ssv_ct} \
                        {self.__class__.syn_type[partner_ssv_comp]} \
                            ({partner_ssv_comp}) \
                        -> {ssv_id}: {ssv_ct} \
                            {self.__class__.syn_type[ssv_comp]} \
                            ({ssv_comp})'

                if message != self.cur_message:
                    with self.viewer.config_state.txn() as s:
                        s.status_messages['status'] = message
                    self.cur_message = message
            else:
                return

    def get_synaptic_partner(self, ssv_id):
        """Gets synaptic partner ssv_id, compartment predictions and cell type
        of synaptic partners and rep. coords of synapse.

        :param ssv_id: segment id
        :type ssv_id: int
        :return result: info about the synaptic partners
        :rtype result: dict, -1
        """

        partners_ix = np.where(np.any(self.params["neuron_partners"] == ssv_id, axis=1))[0]
        mask = np.ones_like(partners_ix, dtype=bool)
        mask = mask & (self.params["syn_probs"][partners_ix] > 0.9)
        
        if not np.any(mask):
            return -1
        
        mask_axon_den = np.where((self.params["partner_axoness"][partners_ix][:,0] == 1) | (self.params["partner_axoness"][partners_ix][:,1] == 1), True, False) & \
                np.where((self.params["partner_axoness"][partners_ix][:,0] == 0) | (self.params["partner_axoness"][partners_ix][:,1] == 0), True, False)

        mask_axon_soma = np.where((self.params["partner_axoness"][partners_ix][:,0] == 1) | (self.params["partner_axoness"][partners_ix][:,1] == 1), True, False) & \
                np.where((self.params["partner_axoness"][partners_ix][:,0] == 2) | (self.params["partner_axoness"][partners_ix][:,1] == 2), True, False)

        mask = mask & (mask_axon_den | mask_axon_soma)
        
        if not np.any(mask):
            return -1
        
        loc = self.params["syn_areas"][partners_ix][mask].argmax()
        partners = self.params["neuron_partners"][partners_ix][mask][loc]
        partner_loc = np.where(partners != ssv_id)[0].item() 
        
        result = {
            "partner_ssv": partners[partner_loc],
            "ssv_comp": self.params["partner_axoness"][partners_ix][mask][loc][1-partner_loc],
            "partner_ssv_comp": self.params["partner_axoness"][partners_ix][mask][loc][partner_loc],
            "ssv_ct": self.params["partner_celltypes"][partners_ix][mask][loc][1-partner_loc],
            "partner_ssv_ct": self.params["partner_celltypes"][partners_ix][mask][loc][partner_loc],
            "rep_coords": self.params["rep_coords"][partners_ix][mask][loc]
        }
        
        return result

    def on_state_changed_deprecated(self):
        """Captures a state change and updates state."""

        ix, segmentation_layer = get_segmentation_layer(self.viewer.state.layers)
        segment_query = segmentation_layer.segment_query
        
        # full pattern match required to avoid response generation
        if segment_query != None and re.fullmatch(self.__class__.master_pattern, segment_query) != None:
            logger.info(f'Entered in query: {segment_query}')
            try:
                ct_match = next(self.__class__.ct_pattern.finditer(segment_query))
                celltype = ct_match.group(1).upper() # case insensitive match
                logger.info(f'Celltype {celltype}')
                if self.CTmask[celltype] == []:
                    logger.info(f"Storing ids mask of celltype {celltype} in memory")
                    start = timer()
                    self.CTmask[celltype] = np.where(
                        self.params["celltypes"] == str2int_converter(celltype, self.gt_type))[0]
                    end = timer()
                    logger.info(f"Loaded celltype ids mask after {(end - start):.3f} seconds")

            except AttributeError as e:
                logger.error(e)

            except KeyError as e:
                logger.error(f"No matching cell type found for {e}")

            except FileNotFoundError as e:
                logger.error(f"{e} celltype_cnn_e3s.npy not found")            

            # list of (prop_type, operation, threshhold)
            filter_list = []
            pageNr = -1

            filter_iter = self.__class__.property_filter.finditer(segment_query)
            prop_iter = iter(self.__class__.properties)

            #loop through property iterator
            while True:
                try:
                    # get the next item
                    element = next(filter_iter).group(1)
                    prop = next(prop_iter)
                    if element == '#': # jump over missing condition
                        continue
                    filter_list.append((prop, element[0], element[1:]))

                except StopIteration:
                    # if StopIteration is raised, break from loop
                    break

            page_match = next(iter(self.__class__.page_pattern.finditer(segment_query)))
            if page_match == None:
                pageNr = 1
            else:
                pageNr = int(page_match.group(0).split("pg")[1])
            
            # logger.info('No match found for page property')

            logger.info(f'Filter list: {filter_list}')
            # get filtered ssv_ids
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

            # update segment query and state
            response = ', '.join(str(ssv_id) for ssv_id in page)
            new_state = copy.deepcopy(self.viewer.state)
            new_state.layers[ix].segment_query = response
            self.viewer.set_state(new_state)

    def on_state_changed(self):
        """Captures a state change and updates state. Individual
        property querying supported"""

        ix, segmentation_layer = get_segmentation_layer(self.viewer.state.layers)
        segment_query = segmentation_layer.segment_query
        
        # full pattern match required to avoid response generation
        if segment_query != None and re.fullmatch(self.__class__.master_pattern, segment_query, re.IGNORECASE) != None: # ignore case for string based properties
            logger.info(f'Entered in query: {segment_query}')
            matches = re.match(self.__class__.master_pattern, segment_query, re.IGNORECASE)
            logger.debug(matches.groups())

            try:
                celltype = matches.group(1).upper() # case insensitive match
                if celltype == "GPE":
                    celltype = "GPe"

                elif celltype == "GPI":
                    celltype = "GPi"

                elif celltype == "MODULATORY":
                    celltype = "modulatory"

                if len(self.CTmask[celltype]) == 0:
                    logger.info(f"Storing ids mask of cell type {celltype} in memory")
                    start = timer()
                    self.CTmask[celltype] = np.where(
                        self.params["celltype_cnn_e3s"] == str2int_converter(celltype, self.gt_type))[0]
                    end = timer()
                    logger.debug(f"Loaded cell type ids mask after {(end - start):.3f} seconds")

            # AttributeError: if cell type is not matched
            # KeyError: if cell type is matched but not present in the dataset
            # FileNotFoundError: if celltype_cnn_e3s.npy is not found
            except (AttributeError, KeyError, FileNotFoundError) as e: 
                celltype = None
                if type(e).__name__ == "KeyError":
                    logger.warning('{}: {} not found in the dataset'.format(type(e).__name__, e))
                else:
                    logger.warning('{}: {}'.format(type(e).__name__, e))

            finally:
                logger.info(f'Cell type {celltype}')
            celltype = None

            if celltype != None:
                message = 'Loading ssv ids of cell type {}'.format(celltype)

            else:
                message = 'Loading ssv ids'

            # list of (prop_type, operation, threshold)
            filter_list = []
            pageNr = -1
            
            filter_iter = iter([prop for prop in matches.groups()[1:-1] if prop != None]) # excluding cell type, page and NoneType properties
            
            # add operator-operand pairs to filter_list
            while True:
                try:
                    prop = next(filter_iter)
                    filter_list.append(tuple(re.split('(>|<|=)', prop))) # property split (prop, >|<|=, int)
                    if filter_list[0][0].lower() == "size":
                        message += ' with {}000'.format(prop) # add 1000 to size
                    else:
                        message += ' with {}'.format(prop)
                    
                # StopIteration: to end the iterator
                except (StopIteration, TypeError) as e: 
                    logger.info('{}: End of parsing'.format(type(e).__name__))
                    break

            logger.info(f'Filter list: {filter_list}')

            # handle pages
            page_match = matches.group(4)
            if page_match == None:
                pageNr = 1
            else:
                pageNr = int(page_match.split("pg")[1])

            if message != self.cur_message:
                with self.viewer.config_state.txn() as s:
                    s.status_messages['status'] = message
                self.cur_message = message
            
            # get filtered ssv_ids
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

            # update segment query and state
            response = ', '.join(str(ssv_id) for ssv_id in page)
            new_state = copy.deepcopy(self.viewer.state)
            new_state.layers[ix].segment_query = response
            self.viewer.set_state(new_state)

    def get_state_segment_ids(self, celltype, filter_list):
        """Retrieves a subset of ssv_ids based on the query.
        
        :param celltype: queried cell type
        :type celltype: str
        :param filter_list: [(<property>, <operator>, <value>),...]
        :type filter_list: list of tuple
        :return ssv_ids_of_interest: 
        :rtype ssv_ids_of_interest: numpy.ndarray 
        """

        if celltype != None:
            indices = self.CTmask[celltype] # use cached cell type indices
        else:
            indices = list(range(len(self.ssv_ids)))

        mask = np.ones(shape=(len(indices),), dtype=np.bool)       

        for prop, op, thresh in filter_list:
            if prop.lower() == "mito":
                prop_name = "mi"
            else:
                prop_name = prop.lower()

            prop_array = self.params[prop_name+"s"][indices]

            # check for mito
            if prop_name == 'mi':
                start = timer()
                func = np.vectorize(len)
                prop_array = func(prop_array)
                end = timer()
                logger.info('Time: {:.2f}'.format(end-start))

            if prop_name == 'size':
                thresh *= 1000

            if op == ">":
                mask = np.logical_and(mask, (prop_array > int(thresh)))

            elif op == "<":
                mask = np.logical_and(mask, (prop_array < int(thresh)))

            else: # equality check
                mask = np.logical_and(mask, (prop_array == int(thresh)))

        ssv_ids_of_interest = self.ssv_ids[indices]
        ssv_ids_of_interest = ssv_ids_of_interest[mask]

        # divide if greater than PAGE_SIZE
        if len(ssv_ids_of_interest) > self.__class__.PAGE_SIZE:
            pages = self._split_pages(ssv_ids_of_interest)
            return pages

        else:
            return ssv_ids_of_interest

    def _split_pages(self, ssv_ids):
        """Splits filtered ssv_ids across multiple pages (if required).

        :param ssv_ids: 
        :type ssv_ids: numpy.ndarray
        :return pages: 
        :rtype pages: list of numpy.ndarray
        """

        pages = np.array_split(ssv_ids, len(ssv_ids) // self.__class__.PAGE_SIZE)
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

    if args.wd == '':
        logger.error('No working directory selected... Aborting')

    global_params.wd = os.path.expanduser(args.wd)
    
    global backend
    backend = configure_backend()

    if "example_cube" in args.wd:
        params = dict(backend=backend, segmentation=KnossosDataset(global_params.config.working_dir+"/knossosdatasets/seg"), image=KnossosDataset(global_params.config.working_dir+"/knossosdatasets/seg"))
    else:
        params = dict(backend=backend, segmentation=KnossosDataset(global_params.config.kd_seg_path), image=KnossosDataset("/wholebrain/songbird/j0251/j0251_72_clahe2"))
    
    # load seg and raw data
    seg_path = global_params.config.kd_seg_path
    
    pf = PropertyFilter(params, args.organelles)
    print(pf.viewer)
