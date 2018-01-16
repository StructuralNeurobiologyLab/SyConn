# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
# SyConn Gate is a thin flask server that allows clients
# over a RESTful HTTP API to interact with a SyConn dataset
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import sys
import copy
import logging

# temporary for easier development
sys.path.append('/u/jkor/repos/SyConn/')
sys.path.append('/u/jkor/repos/knossos_utils')
#sys.path.append('..')

#from syconnfs.representations import super_segmentation as ss
#from ..reps import super_segmentation as ss
import syconn.reps.super_segmentation as ss
import syconn.reps.connectivity_helper as conn

#from ..reps import connectivity_helper as conn
#from syconnfs.representations import segmentation as sd
#from syconnfs.representations import connectivity as conn

from flask import Flask
from flask import request

import json

app = Flask(__name__)


global sg_state


@app.route('/ssv_mesh/<ssv_id>', methods=['GET'])
def route_ssv_mesh(ssv_id):
    return json.dumps(sg_state.backend.ssv_mesh(ssv_id))

@app.route('/ssv_obj_mesh/<ssv_id>/<obj_type>', methods=['GET'])
def ssv_obj_mesh(ssv_id, obj_type):
    return json.dumps(sg_state.backend.ssv_obj_mesh(ssv_id, obj_type))

@app.route('/ssv_list', methods=['GET'])
def route_ssv_list():
    return json.dumps(sg_state.backend.ssv_list())


@app.route('/svs_of_ssv/<ssv_id>', methods=['GET'])
def route_svs_of_ssv(ssv_id):
    return json.dumps(sg_state.backend.svs_of_ssv(ssv_id))


@app.route('/ssv_of_sv/<sv_id>', methods=['GET'])
def route_ssv_of_sv(sv_id):
    return json.dumps(sg_state.backend.ssv_of_sv(sv_id))


@app.route('/all_syn_meta', methods=['GET'])
def route_all_syn_meta():
    return json.dumps(sg_state.backend.all_syn_meta_data())


@app.route("/", methods=['GET'])
def route_hello():
    return json.dumps({'Welcome to': 'SyConnGate'})


class SyConnFS_backend(object):
    def __init__(self, syconnfs_path='', logger=None):
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

        self.logger = logger
        self.logger.info('Initializing SyConn backend')

        self.ssd = ss.SuperSegmentationDataset(syconnfs_path)

        self.logger.info('SuperSegmentation dataset initialized.')

        # directed networkx graph of connectivity
        self.conn_graph = conn.connectivity_to_nx_graph()
        self.logger.info('Connectivity graph initialized.')
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
        self.logger.info('In memory cache of synapses initialized.')


        return

    def ssv_mesh(self, ssv_id):
        """
        Get mesh for ssv_id.
        :param ssv_id: int
        :return: dict
        """
        self.logger.info('Loading ssv mesh {0}'.format(ssv_id))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        mesh = ssv.load_mesh("sv")
        mesh = {'vertices': mesh[1].tolist(),
                'indices': mesh[0].tolist()}
        self.logger.info('Got ssv mesh {0}'.format(ssv_id))

        return mesh

    def ssv_obj_mesh(self, ssv_id, obj_type):
        """
        Get mesh of a specific obj type for ssv_id.
        :param ssv_id: int
        :param obj_type: str
        :return: dict
        """
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        mesh = ssv.load_mesh(obj_type)
        if mesh is None:
            return None
        return {'vertices': mesh[1].tolist(),
                'indices': mesh[0].tolist()}

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
    def __init__(self, log_file='/wholebrain/scratch/areaxfs3/gate/server_log'):

        self.logger = initialize_logging(log_file)

        self.logger.info('SyConn gate server starting up.')
        self.backend = SyConnFS_backend('/wholebrain/scratch/areaxfs3/',
                                        logger=self.logger)
        self.logger.info('SyConn gate server running.')
        return


def initialize_logging(log_file):
    logger = logging.getLogger('gate_logger')

    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(message)s')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

#print('This is the name')
#print(__name__)
#if __name__ == "__main__":
"""
Alternative way of running the server is currently:
export FLASK_APP=server.py
flask run --host=0.0.0.0 --port=8080 --debugger

"""
sg_state = ServerState()

    # context = ('cert.crt', 'key.key') enable later
    #app.run(host='0.0.0.0',  # do not run this on a non-firewalled machine!
    #        port=8080,
    #        # ssl_context=context,
    #        threaded=True,
    #        debug=True)