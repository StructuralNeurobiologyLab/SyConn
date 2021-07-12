from humanfriendly.terminal import message
from syconn.reps import super_segmentation_dataset as ss
from syconn.reps import segmentation
from syconn.analysis.cli import configure_backend
from syconn.analysis.cli import SyConnClient
from syconn.analysis.utils import handle_layer_args
from syconn.handler.logger import log_main as log_gate
from syconn.handler.prediction import str2int_converter, int2str_converter
from syconn import global_params
from timeit import default_timer as timer
# from neuroglancer.server import global_server
import neuroglancer
import argparse
import os
import re
import numpy as np
import copy
import time

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
    :cvar data_dir: directory containing .npy files 
    :type data_dir: str

    :param backend: to retrieve the skeleton, mesh and cell types
    :type backend: SyConnBackend
    :param seg_dataset: segmentation data for neuroglancer.LocalVolume
    :type seg_dataset: KnossosDataset
    :param organelles: command line argument; supported organelles (mi, vc, sj)
    :type organelles: list
    """

    properties = ('mi', 'size')
    # don't forget to update here, if properties number increases
    master_pattern = r'^([a-zA-Z]{2,4}|#)((_((>|<)[0-9]+|#)){2})?_(pg[1-9][0-9]{0,3})$'
    # master_pattern = r'_?(mito[>|<|=][0-9][0-9]{0,3})?_?(size[>|<][1-9][0-9]{1,6})?_?(pg[1-9][0-9]{0,3})?$' # TODO(hashir): Independent property querying 
    ct_pattern = re.compile(r'^([a-zA-Z]{2,4})')
    property_filter = re.compile(r'_((>|<)[0-9]+)')
    page_pattern = re.compile(r'pg[1-9][0-9]{0,3}$')
    PAGE_SIZE = 500
    syn_type = {
        "dendrite": "post-synaptic",
        "axon": "pre-synaptic",
        "soma": "post-synaptic",
    }
    data_dir = '/home/shared'

    def __init__(self, backend, seg_path, organelles, clargs: dict, token=None):
        super().__init__(backend, seg_path, organelles, clargs, token)

        self.gt_type = "ctgt"
        if 'j0251' in global_params.config.working_dir:
            self.gt_type = "ctgt_j0251_v2"

        self.CTs = backend.cts_in_data(self.gt_type)
        logger.info(f"Found [{self.CTs}] cell types in the dataset")
        self.master_pattern = r'(' + '|'.join(ct for ct in self.CTs) + ')?' + self.__class__.master_pattern

        self.ssd = ss.SuperSegmentationDataset(global_params.config.working_dir, sso_locking=False, sso_caching=True)
        # sd = segmentation.SegmentationDataset(obj_type='syn_ssv', working_dir=global_params.config.working_dir)
        
        self.ssv_ids = self.ssd.ssv_ids
        self.cur_message = None

        if not os.path.exists(self.__class__.data_dir) or len(os.listdir(self.__class__.data_dir)) == 0:
            logger.error('Data directory {} does not exist or is empty'.format(self.__class__.data_dir))

        self.neuron_partners = np.load(os.path.join(self.__class__.data_dir, 'neuron_partnerss.npy'), allow_pickle=True)
        self.axoness_partners = np.load(os.path.join(self.__class__.data_dir, 'partner_axonesss.npy'), allow_pickle=True)
        self.syn_probs = np.load(os.path.join(self.__class__.data_dir, 'syn_probs.npy'), allow_pickle=True)
        self.syn_areas = np.load(os.path.join(self.__class__.data_dir, 'mesh_areas.npy'), allow_pickle=True)
        self.partner_celltypes = np.load(os.path.join(self.__class__.data_dir, 'partner_celltypess.npy'), allow_pickle=True)
        self.rep_coords = np.load(os.path.join(self.__class__.data_dir, 'rep_coords.npy'), allow_pickle=True)

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

                if result == -1:
                    message = 'No synaptic partner found for the selected ssv {}'.format(ssv_id)

                    if message != self.cur_message:
                        with self.viewer.config_state.txn() as cfs:
                            cfs.status_messages['status'] = message
                        self.cur_message = message

                    return

                partner_ssv_id = result["partner_ssv"]
                ssv_comp = int2str_converter(result["ssv_comp"], "axgt").split('_')[1]
                partner_ssv_comp = int2str_converter(result["partner_ssv_comp"], "axgt").split('_')[1]
                ssv_ct = int2str_converter(result["ssv_ct"], self.gt_type)
                partner_ssv_ct = int2str_converter(result["partner_ssv_ct"], self.gt_type)
                rep_coords = result["rep_coords"]

                segments.add(partner_ssv_id)
                s.position = np.flip(rep_coords)
                
                # pre-synaptic -> post-synaptic
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
                    with self.viewer.config_state.txn() as cfs:
                        cfs.status_messages['status'] = message
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

        mask = np.any(self.neuron_partners == ssv_id, axis=1) # synaptic partners
        mask = mask & np.where(self.syn_probs > 0.9, True, False) 
        mask_axon_den = np.where((self.axoness_partners[:,0] == 1) | (self.axoness_partners[:,1] == 1), True, False) & np.where((self.axoness_partners[:,0] == 0) | (self.axoness_partners[:,1] == 0), True, False) # axon -> dendrite connections
        mask_axon_soma = np.where((self.axoness_partners[:,0] == 1) | (self.axoness_partners[:,1] == 1), True, False) & np.where((self.axoness_partners[:,0] == 2) | (self.axoness_partners[:,1] == 2), True, False) # axon -> soma connections
        mask = mask & (mask_axon_den | mask_axon_soma) 

        if np.any(mask): # atleast one ssv_id satisfies the conditions   
            # get index of maximum synapse area conditioned on the mask
            largest_syn_ix = np.argmax(self.syn_areas[mask])

            # get largest synaptic partner
            largest_syn_partner = self.neuron_partners[mask][largest_syn_ix]
            # get synaptic partner location
            partner_loc = np.where(largest_syn_partner != ssv_id)[0].item()

            result = {
                "partner_ssv": largest_syn_partner[partner_loc],
                "ssv_comp": self.axoness_partners[mask][largest_syn_ix][1-partner_loc],
                "partner_ssv_comp": self.axoness_partners[mask][largest_syn_ix][partner_loc],
                "ssv_ct": self.partner_celltypes[mask][largest_syn_ix][1-partner_loc],
                "partner_ssv_ct": self.partner_celltypes[mask][largest_syn_ix][partner_loc],
                "rep_coords": self.rep_coords[mask][largest_syn_ix]
            }

            return result

        return -1

    def on_state_changed(self):
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
                        np.load('celltype_cnn_e3s.npy', allow_pickle=True) == str2int_converter(celltype, self.gt_type))[0]
                    end = timer()
                    logger.info(f"Loaded celltype ids mask after {(end - start):.3f} seconds")

            except:
                ct_match = None
                logger.warning('No match found for celltype property!')

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

    # TODO(hashir): Independent property querying
    def on_state_changed_experimental(self):
        """Captures a state change and updates state."""

        ix, segmentation_layer = get_segmentation_layer(self.viewer.state.layers)
        segment_query = segmentation_layer.segment_query
        
        # full pattern match required to avoid response generation
        if segment_query != None and re.fullmatch(self.master_pattern, segment_query) != None:
        # if segment_query != None and re.fullmatch(self.__class__.master_pattern, segment_query).groups()
            logger.info(f'Entered in query: {segment_query}')
            matches = re.match(self.master_pattern, segment_query)
            logger.debug(matches.groups())
            try:
                # ct_match = next(self.__class__.ct_pattern.finditer(segment_query))
                ct_match = matches.group(1)
                celltype = ct_match.upper() # case insensitive match
                logger.info(f'Cell type {celltype}')
                
                if len(self.CTmask[celltype]) == 0:
                    logger.info(f"Storing ids mask of cell type {celltype} in memory")
                    start = timer()
                    self.CTmask[celltype] = np.where(
                        np.load(os.path.join(self.__class__.data_dir, 'celltype_cnn_e3s.npy'), allow_pickle=True) == str2int_converter(celltype, self.gt_type))[0]
                    end = timer()
                    logger.debug(f"Loaded cell type ids mask after {(end - start):.3f} seconds")

            except (KeyError, AttributeError) as e:
                celltype = None
                logger.warning('No match found for celltype property!')

            # list of (prop_type, operation, threshhold)
            filter_list = []
            pageNr = -1

            # filter_iter = self.__class__.property_filter.finditer(segment_query)
            # prop_iter = iter(self.__class__.properties)
            filter_iter = iter([prop for prop in matches.groups()[1:-1]]) # excluding cell type and page properties
            
            while True:
                try:
                    filter_list.append(tuple(re.split('(>|<|=)', next(filter_iter)))) # property split (prop, >|<|=, int)
                
                except (StopIteration, TypeError) as e:
                    break

            #loop through property iterator
            # while True:
            #     try:
            #         # get the next item
            #         element = next(filter_iter).group(1)
            #         prop = next(prop_iter)
            #         if element == '#': # jump over missing condition
            #             continue
            #         filter_list.append((prop, element[0], element[1:]))

            #     except StopIteration:
            #         # if StopIteration is raised, break from loop
            #         break

            page_match = matches.group(4)
            if page_match == None:
                pageNr = 1
            else:
                pageNr = int(page_match.split("pg")[1])

            # try:
            #     if page_match == None:
            #         pageNr = 1
            #     else:
            #         # page_match = next(iter(self.__class__.page_pattern.finditer(segment_query)))
            #         # pageNr = int(page_match.group(0).split("pg")[1])
            #         pageNr = int(page_match.split("pg")[1])
            
            # except:
            #     logger.info('No match found for page property')


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
            indices = self.CTmask[celltype]
        else:
            indices = list(range(len(self.ssv_ids)))

        mask = np.ones(shape=(len(indices),), dtype=np.bool)

        for prop, op, thresh in filter_list:
            ''' TODO(hashir): Independent property querying
            if prop == "mito":
                prop_name = "mi"
            else:
                prop_name = prop

            prop_array = np.load(os.path.join(self.__class__.data_dir, prop_name+"s.npy"), allow_pickle=True)[indices]
            '''
            prop_array = np.load(os.path.join(self.__class__.data_dir, prop+"s.npy"), allow_pickle=True)[indices]

            # check for mito
            if prop == 'mi':
                start = timer()
                func = np.vectorize(len)
                prop_array = func(prop_array)
                end = timer()
                logger.info('Time: {:.2f}'.format(end-start))

            if op == ">":
                mask = np.logical_and(mask, (prop_array > int(thresh)))

            elif op == "<":
                mask = np.logical_and(mask, (prop_array < int(thresh)))

            else:
                mask = np.logical_and(mask, (prop_array == int(thresh)))

        ssv_ids_of_interest = self.ssv_ids[indices]
        ssv_ids_of_interest = ssv_ids_of_interest[mask]

        print(ssv_ids_of_interest)

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

    global backend
    if args.wd == '':
        logger.error('No working directory selected... Aborting')

    global_params.wd = os.path.expanduser(args.wd)
    
    backend = configure_backend()
    
    # load seg and raw data
    seg_path = global_params.config.kd_seg_path
    
    pf = PropertyFilter(backend, seg_path, args.organelles, dict(host=args.host, port=args.port))
    print(pf.viewer)
