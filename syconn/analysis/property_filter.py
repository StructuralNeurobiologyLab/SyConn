import argparse
import os
import re
import copy
import time
from typing import List, Tuple, Optional, Union
from collections import defaultdict
import numpy as np
from timeit import default_timer as timer

from knossos_utils import KnossosDataset
from syconn.analysis.cli import configure_backend
from syconn.analysis.cli import SyConnClient
from syconn.handler.logger import log_main as logger
from syconn.handler.prediction import str2int_converter, int2str_converter
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn import global_params
import neuroglancer
import neuroglancer.cli
from neuroglancer.viewer_state import SegmentationLayer
from neuroglancer.config import NeuroConfig

# names of the segmentation layers that contain the segmentation data
SEG_LAYERS = ("j0251_72_seg_20210127_agglo2", "j0251_rag_flat_Jan2019_v3", "j0126_areaxfs_v10", "j0126_assembled_core_relabeled")


def get_segmentation_layer(layers: List[SegmentationLayer]) -> SegmentationLayer:
    """Returns the segmentation layer with the volume data

    Args:
        layers: list of neuroglancer layers

    TODO:
        * check if there is a better way to do this

    Returns:
        Segmentation layer with volume
    """

    for layer in layers:
        if isinstance(layer.layer, neuroglancer.SegmentationLayer) and (layer.name in SEG_LAYERS):
            return layer


def get_celltype(ct: int, gt_type: str) -> str:
    """Returns the celltype name from the celltype id. Adapted from
    `~syconn.handler.prediction.int2str_converter`

    Args:
        ct (int): cell type id

    Returns:
        str: celltype name
    """    
    if gt_type == "ctgt_j0251_v3":
        int2str_label = {0:'exc', 1: 'DA', 2: 'MSN', 3: 'LMAN', 4: 'HVC', 5: 'TAN', 6: 'GP', 7: 'GP', 8: 'int 3', 9: 'int 1', 10: 'int 2'}
    elif gt_type == "ctgt_v2":
        int2str_label = {0:"exc", 1: "modulatory", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "GP", 6: "INT"}
    else:
        raise ValueError("Unknown gt_type {}".format(gt_type))

    try:
        return int2str_label[ct]

    except KeyError:
        return "Unknown"


class PropertyFilter(SyConnClient):
    """Inherits from :class:`~syconn.analysis.cli.SyConnClient` and
    adds custom functionality to the neuroglancer viewer.

    The dynamic properties should be added here when extending the
    SyConnClient class. Currently, the following properties are
    supported:
    - Synaptic partner with the largest synapse area
    - Presynaptic partners
    - Postsynaptic partners
    - Depth first search (DFS) of the postsynaptic partners

    Attributes:
        params (NeuroConfig): configuration parameters for neuroglancer
            server
        acquisition (str): name of the dataset
        version (str): version of the dataset
        gt_type (str): name of the ground truth type
        CTs (list): list of cell types in the dataset
        ssv_ids (list): list of ssvs
        cur_message (str): current status message
        CTmask (dict): dictionary of cell type indices in the ssv_ids 
            array

    Note:
        This class also has a property filter. The properties are 
        specified in a tuple and follow the specification of the
        SuperSegmentationDataset (see :class:`~syonn.reps.\
            super_segmentation_dataset.SuperSegmentationDataset`)

        Filtering is done by specifying one or more of the properties
        in order in the text field of the viewer. Examples:
        msn_pg2 -> filters msn cells and displays the second page
            if there is more than one page
        msn_mito>100_size>100 -> filters msn cells with number of 
            mitochondria greater than 100 and size greater than 100000.
            The size is specified in micrometers.
    """    

    properties = ('mi', 'size')
    # don't forget to update here, if additional properties are added
    master_pattern = r'^([a-zA-Z]{2,4})?_?(mito[>|<|=][0-9][0-9]{0,3})?_?(size[>|<][1-9][0-9]{1,6})?_?(pg[1-9][0-9]{0,3})?$'
    ct_pattern = re.compile(r'^([a-zA-Z]{2,4})')
    property_filter = re.compile(r'_((>|<)[0-9]+)')
    page_pattern = re.compile(r'pg[1-9][0-9]{0,3}$')
    PAGE_SIZE = 500
    syn_type = {
        "dendrite": "post-synaptic",
        "axon": "pre-synaptic",
        "soma": "post-synaptic",
    }

    def __init__(self, params: NeuroConfig, token: Optional[str]=None, organelles: Optional[list]=None, use_tpl_mask=True, **kwargs):
        """Initializes the base client and property filter

        Args:
            params: server configuration
            token: 40-character hex string. Defaults to None.
            organelles: list of organelles to be displayed
            use_tpl_mask: whether to use the total path length mask or not. Defaults to True.
        """              

        super().__init__(params, token, organelles, **kwargs)

        self.params = params
        self.acquisition = params.acquisition
        self.version = params.version
        self.use_tpl_mask = use_tpl_mask

        if self.use_tpl_mask:
            logger.debug("Using total path length mask for dynamic properties")

        self.gt_type = "ctgt_v2"
        if params.acquisition == "j0251":
            self.gt_type = "ctgt_j0251_v3"
        
        # Uncomment when the old segment properties are used
        # self.CTs = params["backend"].cts_in_data(self.gt_type)
        # logger.info(f"Found {self.CTs} cell types in the dataset")
        
        self.ssv_ids = params["ssvs"]
        self.cur_message = None

        self.ssd = SuperSegmentationDataset(working_dir=params["working_dir"])

        # dict of cell type indices in the ssv_ids array
        # Uncomment when the old segment properties are used
        # self.CTmask = {ct: [] for ct in self.CTs}
        # logger.info(f"Available filter properties {self.__class__.properties}")

        # action handlers for dynamic properties
        self.viewer.actions.add('show-synaptic-partner-with-largest-area', self.get_synaptic_partner)
        self.viewer.actions.add('show-presynaptic-partners', self.get_presynaptic_partners)
        self.viewer.actions.add('show-postsynaptic-partners', self.get_postsynaptic_partners)
        self.viewer.actions.add('cycle-synaptic-partners', self.cycle_synaptic_partners)
        self.viewer.actions.add('depth-first-search', self.depth_first_search)
        self.viewer.actions.add('reset-viewer-state', self.reset_viewer_state)
        self.viewer.actions.add('share-viewer', self.generate_viewer_link)

        # bind actions to hotkeys
        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.data_view['keyp'] = 'show-synaptic-partner-with-largest-area'
            s.input_event_bindings.data_view['control+keyi'] = 'show-presynaptic-partners'
            s.input_event_bindings.data_view['control+keyo'] = 'show-postsynaptic-partners'
            s.input_event_bindings.data_view['control+keyd'] = 'depth-first-search'
            s.input_event_bindings.viewer['control+keyx'] = 'reset-viewer-state'
            s.input_event_bindings.viewer['control+keyl'] = 'share-viewer'
        
        self.counter = 0  # counter for cycling through synaptic partners
        self.active_cell = None  # active cell 
        self.active_ct = None  # cell type of the active cell
        self.partner_type = None  # type of synaptic partner (pre or post)
        self.partner_ids = None  # array of filtered partner ids
        self.rep_coords = None  # coordinates of the synapses
        self.cts = None  # cell types of the filtered partners
        self.mesh_areas = None  # areas of the synapses
        self.depth = 0  # depth of the depth-first search
        self.visited = set()  # set of visited nodes
        self.next_cell = None  # next cell to be visited
        
        # Precomputed segment properties are used now. Uncomment this when the old segment properties are used
        # defer callback necessary to avoid deadlock 
        # self.viewer.shared_state.add_changed_callback(
        #     lambda: self.viewer.defer_callback(self.on_state_changed)
        # )

    def update_status_message(self, message: str):
        """Updates the status message in the viewer

        Args:
            message (str): message to display
        """      
        with self.viewer.config_state.txn() as s:
            s.status_messages['status'] = message

    def generate_viewer_link(self, action_state):
        """Generates a link to the current viewer state

        Args:
            action_state: captures the hotkey action
        """        

        from neuroglancer import url_state
        from urllib.parse import unquote

        logger.info("Generating viewer link...")
        encoded_url = url_state.to_url(self.viewer.state, prefix=f"https://syconn.esc.mpcdf.mpg.de/share/{self.acquisition}/{self.version}/")
        url = unquote(encoded_url)
        # remove ! in url
        url_without_exclamation = re.sub("[!]", "ä", url)
        # replace hash with another character so that it can be reversed later
        url_without_hash = re.sub("[#]", "ß", url_without_exclamation)

        self.update_status_message(url_without_hash)

        logger.info(url_without_hash)

    def reset_viewer_state(self, action_state):
        """Resets the viewer state to the default state"""

        with self.viewer.txn() as s:
            layer = get_segmentation_layer(s.layers)
            layer.segments.clear()
            layer.segment_query = ""
            s.position = [b // 2 for b in self.params["boundary"]]  # set viewer position to the center of the dataset

        with self.viewer.config_state.txn() as s:
            s.status_messages['status'] = None  # clear status message

    def depth_first_search(self, action_state):
        """Depth-first search of the synaptic chain starting from the
        selected cell.

        Args:
            action_state (ActionState): captures the hotkey action
        """ 
        logger.debug(f"Current depth: {self.depth}")

        if self.depth >= 3:  # max depth
            self.depth = 0
            self.visited = set()
            self.next_cell = None
            self.update_status_message("Maximum depth reached. Select a new cell to start the search.")
            return

        elif self.depth == 0:  # start search
            segment_id = action_state.selected_values.get(self.seg_name) # super().seg_name

            if segment_id is None:
                message = "No cell selected! Double click on a segment in one of the cross-sectional views"
                self.update_status_message(message)
                return

            if isinstance(segment_id.value, int):
                ssv_id = segment_id.value
            else:
                ssv_id = int(segment_id.value.key)

        else:  # continue search
            ssv_id = self.next_cell

        # find strongest connection partner and update viewer
        self.update_viewer(ssv_id)

    def update_viewer(self, ssv_id):
        """Updates the viewer with the next cell in the search.

        Args:
            ssv_id (int): id of the selected segment

        TODO:
            * generalize this for other dynamic properties
        """        
        with self.viewer.txn() as s:
            layer = get_segmentation_layer(s.layers)

            if ssv_id in layer.segments and ssv_id not in self.visited:
                message = f"Searching postsynaptic partner of {ssv_id} with strongest connection"
                # self.update_status_message(message)
                logger.info(message)
                self.depth += 1  # increase depth
                self.visited.add(ssv_id)  # add current node to visited nodes

                tic = time.time()
                result = self._get_strongest_connection_partner(ssv_id)
                toc = time.time()
                logger.debug(f"Got strongly connected partner in {toc-tic:.2f} seconds")

                if result == -1:
                    message = f"No postsynaptic partner found for {ssv_id} that satisfies the condition. Search ended at depth {self.depth}. Select a new cell to start the search."
                    self.update_status_message(message)
                    logger.warning(message)
                    self.depth = 0  # reset depth
                    self.visited = set()  # reset visited nodes
                    self.next_cell = None
                    return

                partner_id = result[0]
                self.next_cell = partner_id
                rep_coord = result[1]
                partner_ct = get_celltype(result[2], self.gt_type)
                ssv_ct = get_celltype(result[3], self.gt_type)

                layer.segments.add(partner_id)
                s.position = rep_coord
                
                if self.depth == 1:
                    message = f"{ssv_id} ({ssv_ct}) → {partner_id} ({partner_ct})"
                else:
                    message = self.cur_message + f" → {partner_id} ({partner_ct})"

                self.cur_message = message
                self.update_status_message(message)
    
    def _get_strongest_connection_partner(self, ssv_id):
        """Finds the strongest connection partner of the selected
        segment.

        Args:
            ssv_id (int): id of the selected segment

        Returns:
            tuple: (partner_id, rep_coord, partner_ct, ssv_ct)
        """        
        mask = ((self.params["neuron_partners"][:,0] == ssv_id) & ((self.params["partner_axoness"][:,0] == 1) | (self.params["partner_axoness"][:,0] == 3) | (self.params["partner_axoness"][:,0] == 4)) & ((self.params["partner_axoness"][:,1] == 0) | (self.params["partner_axoness"][:,1] == 2))) |\
((self.params["neuron_partners"][:,1] == ssv_id) & ((self.params["partner_axoness"][:,1] == 1) | (self.params["partner_axoness"][:,1] == 3) | (self.params["partner_axoness"][:,1] == 4)) & ((self.params["partner_axoness"][:,0] == 0) | (self.params["partner_axoness"][:,0] == 2)))

        mask &= (self.params["syn_probs"] >= 0.5)

        if not np.any(mask):
            return -1

        outgoing = self.params["neuron_partners"][mask]
        areas = defaultdict(int)

        if len(np.unique(outgoing, axis=0)) < len(outgoing):
            logger.warning("Found multiple outgoing connections for the same partner")
            for syn_pair, a in zip(outgoing, self.params["mesh_areas"][mask]):
                key = syn_pair[syn_pair != ssv_id].item()
                areas[key] += a

            partner_id = max(areas, key=areas.get)
            cand_syn = np.where(outgoing == partner_id)[0]
            largest_syn_ix = self.params["mesh_areas"][mask][cand_syn].argmax()
            partner_loc_mask = outgoing[cand_syn][largest_syn_ix] != ssv_id
            rep_coord = self.params["rep_coords"][mask][cand_syn][largest_syn_ix]
            partner_ct = self.params["partner_celltypes"][mask][cand_syn][largest_syn_ix][partner_loc_mask].item()
            ssv_ct = self.params["partner_celltypes"][mask][cand_syn][largest_syn_ix][~partner_loc_mask].item()

        else:
            ix = self.params["mesh_areas"][mask].argmax()
            partner_id = outgoing[ix][outgoing[ix] != ssv_id].item()
            rep_coord = self.params["rep_coords"][mask][ix]
            partner_ct = self.params["partner_celltypes"][mask][ix][outgoing[ix] != ssv_id].item()
            ssv_ct = self.params["partner_celltypes"][mask][ix][outgoing[ix] == ssv_id].item()

        return partner_id, rep_coord, partner_ct, ssv_ct

    def cycle_synaptic_partners(self, action_state):
        """Cycles through the synaptic partners of the selected cell.

        Args:
            action_state: captures the hotkey action
        """    

        if self.counter < len(self.rep_coords):  # if there are still synaptic partners to be shown
            ct_pair = self.cts[self.counter]  # get the cell type pair
            pct = ct_pair[ct_pair != self.active_ct]  # get the partner cell type
        
            if len(pct) == 0:  # same cell type as the active cell
                pct = self.active_ct
            else:
                pct = pct.item()  # different from the active cell

            ct = get_celltype(self.active_ct, self.gt_type)
            partner_ct = get_celltype(pct, self.gt_type)

            message = f"{self.partner_type} partner {self.partner_ids[self.counter]} ({partner_ct}) of {self.active_cell} ({ct}). Synaptic area: {self.mesh_areas[self.counter]:.4f} µm²"
            self.update_status_message(message)

            # set the viewer to the position of the synapse
            with self.viewer.txn() as s:
                layer = get_segmentation_layer(s.layers)
                layer.segments.clear()
                layer.segments.add(self.active_cell)
                layer.segments.add(int(layer.segment_query.split(",")[self.counter]))
                s.position = self.rep_coords[self.counter]

        else:  # reset the counter and clear the segments
            message = f"You have viewed all the synaptic partners of {self.active_cell}. Pressing 'Ctrl+u' again will start from the beginning."
            self.update_status_message(message)
            self.counter = 0

            with self.viewer.txn() as s:
                layer = get_segmentation_layer(s.layers)
                layer.segments.clear()
                layer.segments.add(self.active_cell)

            return
        
        self.counter += 1  # increment the counter for next 'ctrl+u' action

    def _filter_presynaptic_partners(self, ssv_id: int) -> Union[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Filters the presynaptic partners of the selected cell.

        Args:
            ssv_id (int): id of the selected cell

        Returns:
            -1 or a tuple of partner cell ids, synapse coordinates and
            cell types of the presynaptic partners
        """ 
        if self.use_tpl_mask:
            pre_mask = self.params["tpl_mask"]  # filter out synaptic partners where either of the partner cells has total path length <= 150
        else:
            pre_mask = np.ones_like(self.params["syn_probs"], dtype=bool)  # no pre-filtering

        # get the presynaptic partners (active cell's soma and dendrite synapses)
        mask = (
            (pre_mask) & (self.params["neuron_partners"][:,0] == ssv_id) & \
                ((self.params["partner_axoness"][:,0] == 0) | (self.params["partner_axoness"][:,0] == 2))
        ) | \
        (
            (pre_mask) & (self.params["neuron_partners"][:,1] == ssv_id) & \
                ((self.params["partner_axoness"][:,1] == 0) | (self.params["partner_axoness"][:,1] == 2))
        )

        if not np.any(mask):
            return -1

        mask = mask & (self.params["syn_probs"] >= 0.5)

        if not np.any(mask):
            return -1

        incoming = self.params["neuron_partners"][mask]
    
        partner_ids, ix = np.unique(incoming[incoming != ssv_id], return_index=True)
        rep_coords = self.params["rep_coords"][mask][ix]
        cts = self.params["partner_celltypes"][mask][ix]
        mesh_areas = self.params["mesh_areas"][mask][ix]

        # sort the partners by the synapse mesh area (descending) 
        sorted_ix = np.argsort(mesh_areas)[::-1]

        return (partner_ids[sorted_ix], rep_coords[sorted_ix], cts[sorted_ix], mesh_areas[sorted_ix])

    def get_presynaptic_partners(self, action_state):
        """Adds the presynaptic partners of the selected cell to the
        segment query of the segmentation layer.

        Args:
            action_state (): captures the hotkey action
        """  

        segment_id = action_state.selected_values.get(self.seg_name) # super().seg_name

        if segment_id is None:
            message = "No cell selected! Double click on a segment in one of the cross-sectional views"
            self.update_status_message(message)
            return

        if isinstance(segment_id.value, int):
            ssv_id = segment_id.value
        else:
            ssv_id = int(segment_id.value.key)

        with self.viewer.txn() as s:
            layer = get_segmentation_layer(s.layers)

            if ssv_id in layer.segments:
                logger.info(f"Getting presynaptic partners of {ssv_id}")
                self.counter = 0

                tic = time.time()
                result = self._filter_presynaptic_partners(ssv_id)
                toc = time.time()
                logger.debug(f"Got presynaptic partners in {toc-tic:.2f} seconds")

                # get active cell's cell type
                ssv = self.ssd.get_super_segmentation_object(ssv_id)
                self.active_ct = ssv.celltype()
                logger.debug("Active cell's cell type: {}".format(self.active_ct))
                ct = get_celltype(self.active_ct, self.gt_type)
                del ssv

                if result == -1:
                    message = f"No presynaptic partners found for {ssv_id} ({ct})"
                    self.update_status_message(message)
                    logger.warning(message)
                    return

                # clear all the segments
                layer.segments.clear()
                self.partner_ids = result[0]
                self.rep_coords = result[1]
                self.cts = result[2]
                self.mesh_areas = result[3]

                # add the partners to the segment query list
                layer.segment_query = ",".join(str(x) for x in self.partner_ids)
                # add the active cell to the segments list
                layer.segments.add(ssv_id)

                self.active_cell = ssv_id
                self.partner_type = "Presynaptic"

                message = f"Presynaptic partners of {ssv_id} ({ct}) are present on the right. Press 'ctrl+u' to cycle through them"
                self.update_status_message(message)

                with self.viewer.config_state.txn() as cs:
                    cs.input_event_bindings.data_view['control+keyu'] = 'cycle-synaptic-partners'

    def _filter_postsynaptic_partners(self, ssv_id):
        """Filters the postsynaptic partners of the selected cell.

        Args:
            ssv_id (int): id of the selected cell

        Returns:
            -1 or a tuple of partner cell ids, synapse coordinates and
            cell types of the postsynaptic partners
        """
        if self.use_tpl_mask:
            pre_mask = self.params["tpl_mask"]  # filter out synaptic partners where either of the partner cells has total path length <= 150
        else:
            pre_mask = np.ones_like(self.params["syn_probs"], dtype=bool)  # no pre-filtering

        # get the postsynaptic partners (active cell's axon synapses)
        mask = (
            (pre_mask) & (self.params["neuron_partners"][:,0] == ssv_id) & \
                ((self.params["partner_axoness"][:,0] == 1) | (self.params["partner_axoness"][:,0] == 3) | (self.params["partner_axoness"][:,0] == 4))
        ) | \
        (
            (pre_mask) & (self.params["neuron_partners"][:,1] == ssv_id) & \
                ((self.params["partner_axoness"][:,1] == 1) | (self.params["partner_axoness"][:,1] == 3) | (self.params["partner_axoness"][:,1] == 4))
        )

        if not np.any(mask):
            return -1

        mask = mask & (self.params["syn_probs"] >= 0.5)

        if not np.any(mask):
            return -1

        incoming = self.params["neuron_partners"][mask]
    
        partner_ids, ix = np.unique(incoming[incoming != ssv_id], return_index=True)
        rep_coords = self.params["rep_coords"][mask][ix]
        cts = self.params["partner_celltypes"][mask][ix]
        mesh_areas = self.params["mesh_areas"][mask][ix]
        
        # sort the partners by the synapse mesh area (descending) 
        sorted_ix = np.argsort(mesh_areas)[::-1]

        return (partner_ids[sorted_ix], rep_coords[sorted_ix], cts[sorted_ix], mesh_areas[sorted_ix])

    def get_postsynaptic_partners(self, action_state):
        """Adds the postsynaptic partners of the selected cell to the
        segment query of the segmentation layer.

        Args:
            action_state (): captures the hotkey action
        """ 

        segment_id = action_state.selected_values.get(self.seg_name)

        if segment_id is None:
            message = "No cell selected! Double click on a segment in one of the cross-sectional views"
            self.update_status_message(message)
            return

        if isinstance(segment_id.value, int):
            ssv_id = segment_id.value
        else:
            ssv_id = int(segment_id.value.key)

        with self.viewer.txn() as s:
            layer = get_segmentation_layer(s.layers)
            if ssv_id in layer.segments:
                logger.info(f"Getting postsynaptic partners of {ssv_id}")
                self.counter = 0

                tic = time.time()
                result = self._filter_postsynaptic_partners(ssv_id)
                toc = time.time()
                logger.debug(f"Got postsynaptic partners in {toc-tic:.2f} seconds")

                # get active cell's cell type
                ssv = self.ssd.get_super_segmentation_object(ssv_id)
                self.active_ct = ssv.celltype()
                ct = get_celltype(self.active_ct, self.gt_type)
                del ssv

                if result == -1:
                    message = f"No postsynaptic partners found for {ssv_id} ({ct})"
                    self.update_status_message(message)
                    logger.warning(message)
                    return

                # clear all the segments
                layer.segments.clear()
                self.partner_ids = result[0]
                self.rep_coords = result[1]
                self.cts = result[2]
                self.mesh_areas = result[3]

                # add the partners to the segment query list
                layer.segment_query = ",".join(str(x) for x in self.partner_ids)
                # add the active cell to the segments list
                layer.segments.add(ssv_id)

                self.active_cell = ssv_id
                self.partner_type = "Postsynaptic"

                message = f"Postsynaptic partners of {ssv_id} ({ct}) are present on the right. Press 'ctrl+u' to cycle through them"
                self.update_status_message(message)

                with self.viewer.config_state.txn() as cs:
                    cs.input_event_bindings.data_view['control+keyu'] = 'cycle-synaptic-partners'

    def get_synaptic_partner(self, action_state):
        """Retrieves the synaptic partner of the currently selected 
        cell.

        Args:
            action_state: captures the hotkey action
        """        

        segment_id = action_state.selected_values.get(self.seg_name) # super().seg_name

        if segment_id is None:
            message = "No cell selected! Double click on a segment in one of the cross-sectional views"
            self.update_status_message(message)
            return
        
        if isinstance(segment_id.value, int):
            ssv_id = segment_id.value
        else:
            ssv_id = int(segment_id.value.key)
        
        with self.viewer.txn() as s:
            layer = get_segmentation_layer(s.layers)
            
            if ssv_id in layer.segments:

                message = f"Loading synaptic partner for the selected cell {ssv_id}"
                self.update_status_message(message)

                start = time.time()
                result = self._filter_synaptic_partner(ssv_id)
                dtime = time.time() - start
                logger.debug('Got synaptic partner after {:.2f}'.format(dtime))

                # if no synaptic partner is found
                if result == -1:
                    message = f'No synaptic partner found for the selected cell {ssv_id}'
                    self.update_status_message(message)
                    return

                # get synaptic partner id, compartment predictions and cell types
                partner_ssv_id = result["partner_ssv"]
                ssv_comp = get_celltype(result["ssv_comp"], "axgt").split('_')[1]
                partner_ssv_comp = get_celltype(result["partner_ssv_comp"], "axgt").split('_')[1]
                ssv_ct = get_celltype(result["ssv_ct"], self.gt_type)
                partner_ssv_ct = get_celltype(result["partner_ssv_ct"], self.gt_type)
                rep_coords = result["rep_coords"]

                # add synaptic partner to segment list
                layer.segments.add(partner_ssv_id)
                # use this for neuroglancer.LocalVolume source
                # s.position = np.flip(rep_coords) 
                s.position = rep_coords # set position to the synapse location
                
                # pre-synaptic -> post-synaptic message format
                if self.__class__.syn_type[ssv_comp] == "pre-synaptic":
                    message = f'{ssv_id}: {ssv_ct} \
                        {self.__class__.syn_type[ssv_comp]} \
                            ({ssv_comp}) \
                        → {partner_ssv_id}: {partner_ssv_ct} \
                            {self.__class__.syn_type[partner_ssv_comp]} \
                            ({partner_ssv_comp})'
                else:
                    message = f'{partner_ssv_id}: {partner_ssv_ct} \
                        {self.__class__.syn_type[partner_ssv_comp]} \
                            ({partner_ssv_comp}) \
                        → {ssv_id}: {ssv_ct} \
                            {self.__class__.syn_type[ssv_comp]} \
                            ({ssv_comp})'

                self.update_status_message(message)
                
            else:
                message = f"{ssv_id} is not selected. Double-click on the segment to select it"
                self.update_status_message(message)
                return

    def _filter_synaptic_partner(self, ssv_id: int) -> dict:
        """Gets synaptic partner ssv_id, compartment predictions, cell
        type, and rep. coords of synapse.

        Args:
            ssv_id: ssv if of the selected cell

        Returns:
            partners ssv ids, cell types, and compartment predictions
        """        

        partners_ix = np.where(np.any(self.params["neuron_partners"] == ssv_id, axis=1))[0]
        mask = np.ones_like(partners_ix, dtype=bool)
        mask = mask & (self.params["syn_probs"][partners_ix] > 0.9)
        
        if not np.any(mask):
            return -1
        
        # get the axo-dendritic synapse indices
        mask_axon_den = ((self.params["partner_axoness"][partners_ix][:,0] == 1) | (self.params["partner_axoness"][partners_ix][:,1] == 1)) & \
                        ((self.params["partner_axoness"][partners_ix][:,0] == 0) | (self.params["partner_axoness"][partners_ix][:,1] == 0))

        # get the axo-somatic synapse indices
        mask_axon_soma = ((self.params["partner_axoness"][partners_ix][:,0] == 1) | (self.params["partner_axoness"][partners_ix][:,1] == 1)) & \
                    ((self.params["partner_axoness"][partners_ix][:,0] == 2) | (self.params["partner_axoness"][partners_ix][:,1] == 2))

        mask = mask & (mask_axon_den | mask_axon_soma)
        
        if not np.any(mask):
            return -1
        
        loc = self.params["mesh_areas"][partners_ix][mask].argmax()
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

    def on_state_changed(self):
        """Captures a state change and updates state. Individual
        property querying supported."""

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
                        self.params["celltypes"] == str2int_converter(celltype, self.gt_type))[0]
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

    def get_state_segment_ids(self, celltype, filter_list) -> np.ndarray:
        """Retrieves a subset of ssv_ids based on the query.

        Args:
            celltype: [description]
            filter_list: [description]

        Returns:
            ssv ids of interest
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
