from syconn.analysis.property_selector import get_segmentation_layer
from syconn.analysis.utils import handle_layer_args
from syconn.handler import logger
from syconn.reps import segmentation
from syconn.handler.prediction import int2str_converter
from syconn.handler.logger import log_main as log_gate
from syconn import global_params
from syconn.analysis.cli import SyConnClient, configure_backend
from knossos_utils import KnossosDataset

import numpy as np
import os
import argparse
import neuroglancer
import copy

syn_type = {
    1: "pre-synaptic",
    3: "pre-synaptic",
    4: "pre-synaptic",
    0: "post-synaptic",
    2: "unknown"
}

logger = log_gate

def get_segmentation_layer(layers):
    for i, layer in enumerate(layers):
        if isinstance(layer.layer, neuroglancer.SegmentationLayer):
            return i, layer

class SynapseFilter(SyConnClient):
    """
    Retrieves and renders largest synaptic partner of the selected segment (ssv)

    Args:
        backend (SyConnBackend): used for retrieving the skeleton, mesh and cell types
        seg_dataset (KnossosDataset): segmentation data for neuroglancer.LocalVolume
        organelles (List): command line argument for cell organelles (mi, vc, sj)
    """

    def __init__(self, backend, seg_path, organelles):
        super().__init__(backend, seg_path, organelles)

        sd = segmentation.SegmentationDataset(obj_type='syn_ssv', working_dir=global_params.config.working_dir)
        self.neuron_partners = sd.load_numpy_data('neuron_partners')
        self.syn_area = sd.load_numpy_data('mesh_area')
        self.axoness_partners = sd.load_numpy_data('partner_axoness')

        self.viewer.actions.add('show-largest-synaptic-connection', self._handle_select)

        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.data_view['keyp'] = 'show-largest-synaptic-connection'

        # self.viewer.shared_state.add_changed_callback(
        #     lambda: self.viewer.defer_callback(self.on_state_changed)
        # )

        # self.cur_segments = set()
        self.cur_message = None

    
    def _handle_select(self, action_state):
        """
        Action handler for selected ssv [keyp]
        
        :param action_state: neuroglancer.viewer_config_state.ActionState (implicit invoking)
        """

        print('Action state invoked')
        segment_id = action_state.selected_values.get('segmentation_sv')
        if segment_id is None: 
            return
        ssv_id = segment_id.value

        with self.viewer.txn() as s:
            segments = get_segmentation_layer(s.layers)[1].segments
            if ssv_id in segments:
                # print('Clicked segment')
                result = self.get_largest_synaptic_partner(ssv_id)

                if result == -1:
                    message = 'No synpatic partner found for the selected ssv {}'.format(ssv_id)

                    if message != self.cur_message:
                        with self.viewer.config_state.txn() as cfs:
                            cfs.status_messages['status'] = message
                    
                    return

                partner_ssv_id, ssv_comp_label, partner_ssv_comp_label = result
                segments.add(partner_ssv_id)
                ssv_compartment = int2str_converter(ssv_comp_label, "axgt").split('_')[1]
                partner_ssv_compartment = int2str_converter(partner_ssv_comp_label, "axgt").split('_')[1]

                message = f'{ssv_id} -> \
                    {syn_type[ssv_comp_label]} ({ssv_compartment}) \
                    / {partner_ssv_id} -> \
                        {syn_type[partner_ssv_comp_label]} ({partner_ssv_compartment})'

                if message != self.cur_message:
                    with self.viewer.config_state.txn() as cfs:
                        cfs.status_messages['status'] = message
                    self.cur_message = message
            else:
                # print('Mouse hover')
                return

    def get_largest_synaptic_partner(self, ssv_id):
        """
        Gets synaptic partner ssv_id and compartment predictions of synaptic partners
        
        :param ssv_id: int (ssv_id of the selected segment)
        :return: tuple (partner ssv_id, compartment prediction of selected segment, compartment prediction of partner segment),
                -1 (No synaptic partner)
        """
        left_ix = np.where(self.neuron_partners[:,0] == ssv_id)[0]
        right_ix = np.where(self.neuron_partners[:,1] == ssv_id)[0]
        
        lflag = rflag = False

        if len(left_ix) != 0: 
            lflag = True
            left_partner = self.neuron_partners[left_ix]
            left_syn_area = self.syn_area[left_ix]
            left_axoness = self.axoness_partners[left_ix]
            left_syn_area_ix = np.argmax(left_syn_area)

        if len(right_ix) != 0:
            rflag = True
            right_partner = self.neuron_partners[right_ix] 
            right_syn_area = self.syn_area[right_ix]
            right_axoness = self.axoness_partners[right_ix]
            right_syn_area_ix = np.argmax(right_syn_area)

        if lflag and rflag:
            if left_syn_area[left_syn_area_ix] > right_syn_area[right_syn_area_ix]:
                partner_ssv_id = left_partner[left_syn_area_ix][1]
                ssv_comp_label = left_axoness[left_syn_area_ix][0]
                partner_ssv_comp_label = left_axoness[left_syn_area_ix][1]
                
            else:
                partner_ssv_id = right_partner[right_syn_area_ix][0]
                ssv_comp_label = right_axoness[right_syn_area_ix][1]
                partner_ssv_comp_label = right_axoness[right_syn_area_ix][0]

        elif lflag:
            partner_ssv_id = left_partner[left_syn_area_ix][1]
            ssv_comp_label = left_axoness[left_syn_area_ix][0]
            partner_ssv_comp_label = left_axoness[left_syn_area_ix][1]

        elif rflag:
            partner_ssv_id = right_partner[right_syn_area_ix][0]
            ssv_comp_label = right_axoness[right_syn_area_ix][1]
            partner_ssv_comp_label = right_axoness[right_syn_area_ix][0]

        else:
            return -1

        return (partner_ssv_id, ssv_comp_label, partner_ssv_comp_label)


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
    sf = SynapseFilter(backend, seg_path, args.organelles)
    print(sf.viewer)



