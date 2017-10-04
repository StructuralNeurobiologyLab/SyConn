# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
import sys

# temporary for easier development
sys.path.append('/u/jkor/repos/knossos_utils')

from ..representations import super_segmentation as ss

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


@app.route("/", methods=['GET'])
def route_hello():
    return json.dumps({'Welcome to': 'SyConnGate'})

class SyConnFS_backend(object):
    def __init__(self, syconnfs_path = ''):
        """
        Initializes a SyConnFS backend for operation.
        This includes in-memory initialization of the
        most important caches.
        All backend functions must return dicts.
        
        :param syconnfs_path: str 
        """
        self.ssd = ss.SuperSegmentationDataset(syconnfs_path)
        # warmup of ssv_ids?
        self.ssd.ssv_ids

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


class ServerState(object):
    def __init__(self):
        self.backend = SyConnFS_backend('/wholebrain/scratch/areaxfs/')

        return

if __name__ == "__main__":

    state = ServerState()

    #context = ('cert.crt', 'key.key') enable later
    app.run(host='0.0.0.0', # do not run this on a non-firewalled machine!
            port=8080,
            #ssl_context=context,
            threaded=True,
            debug=True)