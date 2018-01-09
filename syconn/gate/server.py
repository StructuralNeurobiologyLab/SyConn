# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
# SyConn Gate is a thin flask server that allows clients
# over a RESTful HTTP API to interact with a SyConn dataset
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import sys
import copy

# temporary for easier development
sys.path.append('/u/jkor/repos/SyConnFS/')
sys.path.append('/u/jkor/repos/SyConnMP/')
sys.path.append('/u/jkor/repos/knossos_utils')

from syconnfs.representations import super_segmentation as ss
from syconnfs.representations import segmentation as sd
from syconnfs.representations import connectivity as conn

from flask import Flask
from flask import request

import json

app = Flask(__name__)


@app.route('/ssv_mesh/<ssv_id>', methods=['GET'])
def route_ssv_mesh(ssv_id):
    return json.dumps(state.backend.ssv_mesh(ssv_id))


@app.route('/ssv_list', methods=['GET'])
def route_ssv_list():
    return json.dumps(state.backend.ssv_list())


@app.route('/svs_of_ssv/<ssv_id>', methods=['GET'])
def route_svs_of_ssv(ssv_id):
    return json.dumps(state.backend.svs_of_ssv(ssv_id))


@app.route('/ssv_of_sv/<sv_id>', methods=['GET'])
def route_ssv_of_sv(sv_id):
    return json.dumps(state.backend.ssv_of_sv(sv_id))


@app.route('/all_syn_meta', methods=['GET'])
def route_all_syn_meta():
    return json.dumps(state.backend.all_syn_meta_data())


@app.route("/", methods=['GET'])
def route_hello():
    return json.dumps({'Welcome to': 'SyConnGate'})


class SyConnFS_backend(object):
    def __init__(self, syconnfs_path=''):
        """
        Initializes a SyConnFS backend for operation.
        This includes in-memory initialization of the
        most important caches. Currently, SyConn Gate
        does not support backend data changes and the server needs
        to restart for changes to be valid. If the backend data
        is changed while the server is running, old content
        might be served.
        All backend functions must return dicts.

        :param syconnfs_path: str 
        """
        self.ssd = ss.SuperSegmentationDataset(syconnfs_path)

        # directed networkx graph of connectivity
        self.conn_graph = conn.connectivity_to_nx_graph()

        # flat array representation of all synapses
        self.conn_dict = conn.load_cached_data_dict()

        idx_filter = self.conn_dict['synaptivity_proba'] > 0.5
        #  & (df_dict['syn_size'] < 5.)

        for k, v in self.conn_dict.iteritems():
            self.conn_dict[k] = v[idx_filter]

        idx_filter = (self.conn_dict['neuron_partner_ax_0'] \
                      + self.conn_dict['neuron_partner_ax_1']) == 1

        for k, v in self.conn_dict.iteritems():
            self.conn_dict[k] = v[idx_filter]

        return

    def ssv_mesh(self, ssv_id):
        """
        Get mesh for ssv_id.
        :param ssv_id: int
        :return: dict
        """
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        return {'vertices': ssv.mesh[0].tolist(),
                'indices': ssv.mesh[1].tolist()}

    def ssv_list(self):
        """
        Returns all ssvs in dataset.
        :return: dict
        """
        return {'ssvs': self.ssd.ssv_ids}

    def ssv_of_sv(self, sv_id):
        """
        Returns the ssv for a given sv_id.
        :param sv_id: 
        :return: 
        """
        return {'ssv': self.ssd.id_changer[int(sv_id)]}

    def svs_of_ssv(self, ssv_id):
        """
        Returns all sv ids for a ssv_id.
        :return: dict
        """
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        return {'svs': ssv.sv_ids.tolist()}

    def all_syn_meta_data(self):
        """
        Returns all synapse meta data. This works only well for fast
        connections and less than 1e6 synapses or so.
        :return: 
        """

        all_syn_meta_dict = copy.copy(self.conn_dict)

        # the self.conn_dict is not json serializeble, due to the numpy arrays
        for k, v in all_syn_meta_dict.iteritems():
            all_syn_meta_dict[k] = v.tolist()

        return all_syn_meta_dict

    def syn_objs_of_ssv_pre_post(self, ssv_id):
        """
        Returns all synapse objs of a given ssv_id.
        :return: 

        """
        syns = dict()
        # not the most efficient approach, a cached map might be necessary for
        # many synapses
        idx = (ssv_id == self.syn_ssv_partner_0) | (
        ssv_id == self.syn_ssv_partner_1)

        syns['ids'] = self.syn_ids[idx]
        syns['sizes'] = self.syn_sizes[idx]
        syns['p0'] = self.syn_ssv_partner_0[idx]
        syns['p1'] = self.syn_ssv_partner_1[idx]
        return syns

    def syn_objs_of_ssv_post(self, ssv_id):
        """
        Return the syn objs where this ssv_id is post synaptic,
        i.e. this ssv_id receives the synapse.

        :param ssv_id: 
        :return: 
        """
        syns = dict()
        # not the most efficient approach, a cached map might be necessary for
        # really many synapses
        idx = (ssv_id == self.syn_ssv_partner_0) | (
        ssv_id == self.syn_ssv_partner_1)

        # find the synapses where this ssv_id is post synaptic

        syns['ids'] = self.syn_ids[idx]
        syns['sizes'] = self.syn_sizes[idx]
        syns['p0'] = self.syn_ssv_partner_0[idx]
        syns['p1'] = self.syn_ssv_partner_1[idx]
        return syns


class ServerState(object):
    def __init__(self):
        self.backend = SyConnFS_backend('/wholebrain/scratch/areaxfs/')

        return


if __name__ == "__main__":
    state = ServerState()

    # context = ('cert.crt', 'key.key') enable later
    app.run(host='0.0.0.0',  # do not run this on a non-firewalled machine!
            port=8080,
            # ssl_context=context,
            threaded=True,
            debug=True)