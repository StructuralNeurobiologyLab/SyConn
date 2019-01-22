# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import copy
import time
import numpy as np
from flask import Flask
import json

from syconn.reps import super_segmentation as ss
from syconn.reps import connectivity_helper as conn
from syconn.config import global_params
from syconn.gate import log_gate

app = Flask(__name__)


global sg_state


@app.route('/ssv_skeleton/<ssv_id>', methods=['GET'])
def route_ssv_skeleton(ssv_id):
    d = sg_state.backend.ssv_skeleton(ssv_id)
    start = time.time()
    ret = json.dumps(d, cls=MyEncoder)
    log_gate.debug("JSON dump:", time.time() - start)
    return ret


@app.route('/ssv_mesh/<ssv_id>', methods=['GET'])
def route_ssv_mesh(ssv_id):
    d = sg_state.backend.ssv_mesh(ssv_id)
    start = time.time()
    ret = json.dumps(d, cls=MyEncoder)
    log_gate.debug("JSON dump:", time.time() - start)
    return ret


@app.route('/ssv_ind/<ssv_id>', methods=['GET'])
def route_ssv_ind(ssv_id):
    d = sg_state.backend.ssv_ind(ssv_id)
    return d


@app.route('/ssv_vert/<ssv_id>', methods=['GET'])
def route_ssv_vert(ssv_id):
    d = sg_state.backend.ssv_vert(ssv_id)
    return d


@app.route('/ssv_norm/<ssv_id>', methods=['GET'])
def route_ssv_norm(ssv_id):
    d = sg_state.backend.ssv_norm(ssv_id)
    return d


@app.route('/ssv_obj_mesh/<ssv_id>/<obj_type>', methods=['GET'])
def ssv_obj_mesh(ssv_id, obj_type):
    d = sg_state.backend.ssv_obj_mesh(ssv_id, obj_type)
    start = time.time()
    ret = json.dumps(d, cls=MyEncoder)
    log_gate.debug("JSON dump:", time.time() - start)
    return ret


@app.route('/ssv_obj_vert/<ssv_id>/<obj_type>', methods=['GET'])
def ssv_obj_vert(ssv_id, obj_type):
    d = sg_state.backend.ssv_obj_vert(ssv_id, obj_type)
    return d


@app.route('/ssv_obj_ind/<ssv_id>/<obj_type>', methods=['GET'])
def ssv_obj_ind(ssv_id, obj_type):
    d = sg_state.backend.ssv_obj_ind(ssv_id, obj_type)
    return d


@app.route('/ssv_obj_norm/<ssv_id>/<obj_type>', methods=['GET'])
def ssv_obj_norm(ssv_id, obj_type):
    d = sg_state.backend.ssv_obj_norm(ssv_id, obj_type)
    return d


@app.route('/ssv_list', methods=['GET'])
def route_ssv_list():
    return json.dumps(sg_state.backend.ssv_list(), cls=MyEncoder)


@app.route('/svs_of_ssv/<ssv_id>', methods=['GET'])
def route_svs_of_ssv(ssv_id):
    return json.dumps(sg_state.backend.svs_of_ssv(ssv_id), cls=MyEncoder)


@app.route('/ssv_of_sv/<sv_id>', methods=['GET'])
def route_ssv_of_sv(sv_id):
    return json.dumps(sg_state.backend.ssv_of_sv(sv_id), cls=MyEncoder)


@app.route('/ct_of_ssv/<ssv_id>', methods=['GET'])
def route_ct_of_sv(ssv_id):
    return json.dumps(sg_state.backend.ct_of_ssv(ssv_id), cls=MyEncoder)


@app.route('/all_syn_meta', methods=['GET'])
def route_all_syn_meta():
    return json.dumps(sg_state.backend.all_syn_meta_data(), cls=MyEncoder)


@app.route("/", methods=['GET'])
def route_hello():
    return json.dumps({'Welcome to': 'SyConnGate'}, cls=MyEncoder)


class SyConnBackend(object):
    def __init__(self, syconn_path='', logger=None):
        """
        Initializes a SyConn backend for operation.
        This includes in-memory initialization of the
        most important caches. Currently, SyConn Gate
        does not support backend data changes and the server needs
        to restart for changes to be valid. If the backend data
        is changed while the server is running, old content
        might be served.
        All backend functions must return dicts.

        :param syconn_path: str
        """
        self.logger = logger
        self.logger.info('Initializing SyConn backend')

        self.ssd = ss.SuperSegmentationDataset(syconn_path)

        self.logger.info('SuperSegmentation dataset initialized.')

        # directed networkx graph of connectivity
        self.conn_graph = conn.connectivity_to_nx_graph()
        self.logger.info('Connectivity graph initialized.')
        # flat array representation of all synapses
        self.conn_dict = conn.load_cached_data_dict()

        idx_filter = self.conn_dict['synaptivity_proba'] > 0.5
        #  & (df_dict['syn_size'] < 5.)

        for k, v in self.conn_dict.items():
            self.conn_dict[k] = v[idx_filter]

        idx_filter = (self.conn_dict['neuron_partner_ax_0']
                      + self.conn_dict['neuron_partner_ax_1']) == 1

        for k, v in self.conn_dict.items():
            self.conn_dict[k] = v[idx_filter]
        self.logger.info('In memory cache of synapses initialized.')

        return

    def ssv_mesh(self, ssv_id):
        """
        Get mesh for ssv_id.
        :param ssv_id: int
        :return: dict
        """
        start = time.time()
        self.logger.info('Loading ssv mesh {}'.format(ssv_id))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        mesh = ssv._load_obj_mesh_compr("sv")
        mesh = {'vertices': mesh[1],
                'indices': mesh[0],
                'normals': mesh[2] if len(mesh) == 2 else []}
        dtime = time.time() - start
        self.logger.info('Got ssv mesh {} after {:.2f}'.format(ssv_id, dtime))
        return mesh

    def ssv_ind(self, ssv_id):
        """
        :param ssv_id: int
        :return: dict
        """
        start = time.time()
        self.logger.info('Loading {} ssv mesh indices'.format(ssv_id))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        mesh = ssv._load_obj_mesh_compr("sv")
        dtime = time.time() - start
        self.logger.info('Got ssv {} mesh indices after'
                         ' {:.2f}'.format(ssv_id, dtime))
        try:
            return b"".join(mesh[0])
        except TypeError:  # contains str, not byte
            return "".join(mesh[0])

    def ssv_vert(self, ssv_id):
        """
        Get mesh vertices for ssv_id.
        :param ssv_id: int
        :return: dict
        """
        start = time.time()
        self.logger.info('Loading ssv {} mesh vertices'.format(ssv_id))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        mesh = ssv._load_obj_mesh_compr("sv")
        dtime = time.time() - start
        self.logger.info('Got ssv {} mesh vertices after'
                         ' {:.2f}'.format(ssv_id, dtime))
        try:
            return b"".join(mesh[1])
        except TypeError:  # contains str, not byte
            return "".join(mesh[1])

    def ssv_skeleton(self, ssv_id):
        """
        Get mesh vertices for ssv_id.
        :param ssv_id: int
        :return: dict
        """
        self.logger.info('Loading ssv skeleton {}'.format(ssv_id))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_skeleton()
        skeleton = ssv.skeleton
        if skeleton is None:
            return {}
        skel_attr = ["nodes", "edges", "diameters"]
        avg_dst = global_params.DIST_AXONESS_AVERAGING
        keys = ["axoness_avg{}".format(avg_dst),
                "axoness_avg{}_comp_maj".format(avg_dst)]
        for k in keys:
            if k in skeleton:
                skel_attr.append(k)
                if type(skeleton[k]) is list:
                    skeleton[k] = np.array(skeleton[k])
        return {k: skeleton[k].flatten().tolist() for k in
                skel_attr}

    def ssv_norm(self, ssv_id):
        """
        Get mesh normals for ssv_id.
        :param ssv_id: int
        :return: dict
        """
        start = time.time()
        self.logger.info('Loading ssv {} mesh normals'.format(ssv_id))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        mesh = ssv._load_obj_mesh_compr("sv")
        dtime = time.time() - start
        self.logger.info('Got ssv {} mesh normals after'
                         ' {:.2f}'.format(ssv_id, dtime))
        if len(mesh) == 2:
            return b""
        try:
            return b"".join(mesh[2])
        except TypeError:  # contains str, not byte
            return "".join(mesh[2])

    def ssv_obj_mesh(self, ssv_id, obj_type):
        """
        Get mesh of a specific obj type for ssv_id.
        :param ssv_id: int
        :param obj_type: str
        :return: dict
        """
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        if obj_type == "sj":
            try:
                obj_type = "syn_ssv"
                _ = ssv.attr_dict[obj_type]  # try to query mapped syn_ssv objects
                log_gate.debug("Loading '{}' objects instead of 'sj' for SSV "
                               "{}.".format(obj_type, ssv_id))
            except KeyError:
                pass
        # if not existent, create mesh
        _ = ssv.load_mesh(obj_type)
        mesh = ssv._load_obj_mesh_compr(obj_type)
        if mesh is None:
            return None
        ret = {'vertices': mesh[1],
               'indices': mesh[0],
               'normals': mesh[2] if len(mesh) == 2 else []}
        return ret

    def ssv_obj_ind(self, ssv_id, obj_type):
        """
        Get mesh indices of a specific obj type for ssv_id.
        :param ssv_id: int
        :param obj_type: str
        :return: dict
        """
        start = time.time()
        self.logger.info('Loading ssv {} {} mesh indices'
                         ''.format(ssv_id, obj_type))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        if obj_type == "sj":
            try:
                obj_type = "syn_ssv"
                _ = ssv.attr_dict[obj_type]  # try to query mapped syn_ssv objects
                log_gate.debug("Loading '{}' objects instead of 'sj' for SSV "
                               "{}.".format(obj_type, ssv_id))
            except KeyError:
                pass
        # if not existent, create mesh
        _ = ssv.load_mesh(obj_type)
        mesh = ssv._load_obj_mesh_compr(obj_type)
        dtime = time.time() - start
        self.logger.info('Got ssv {} {} mesh indices after'
                         ' {:.2f}'.format(ssv_id, obj_type, dtime))
        try:
            return b"".join(mesh[0])
        except TypeError:  # contains str, not byte
            return "".join(mesh[0])

    def ssv_obj_vert(self, ssv_id, obj_type):
        """
        Get mesh vertices  of a specific obj type for ssv_id.
        :param ssv_id: int
        :param obj_type: str
        :return: dict
        """
        start = time.time()
        self.logger.info('Loading ssv {} {} mesh vertices'
                         ''.format(ssv_id, obj_type))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        if obj_type == "sj":
            try:
                obj_type = "syn_ssv"
                _ = ssv.attr_dict[obj_type]  # try to query mapped syn_ssv objects
                log_gate.debug("Loading '{}' objects instead of 'sj' for SSV "
                               "{}.".format(obj_type, ssv_id))
            except KeyError:
                pass
        # if not existent, create mesh
        _ = ssv.load_mesh(obj_type)
        mesh = ssv._load_obj_mesh_compr(obj_type)
        dtime = time.time() - start
        self.logger.info('Got ssv {} {} mesh vertices after'
                         ' {:.2f}'.format(ssv_id, obj_type, dtime))
        try:
            return b"".join(mesh[1])
        except TypeError:  # contains str, not byte
            return "".join(mesh[1])

    def ssv_obj_norm(self, ssv_id, obj_type):
        """
        Get mesh normals of a specific obj type for ssv_id.
        :param ssv_id: int
        :param obj_type: str
        :return: dict
        """
        start = time.time()
        self.logger.info('Loading ssv {} {} mesh normals'
                         ''.format(ssv_id, obj_type))
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        if obj_type == "sj":
            try:
                obj_type = "syn_ssv"
                _ = ssv.attr_dict[obj_type]  # try to query mapped syn_ssv objects
                log_gate.debug("Loading '{}' objects instead of 'sj' for SSV "
                               "{}.".format(obj_type, ssv_id))
            except KeyError:
                pass
        # if not existent, create mesh
        _ = ssv.load_mesh(obj_type)
        mesh = ssv._load_obj_mesh_compr(obj_type)
        dtime = time.time() - start
        self.logger.info('Got ssv {} {} mesh normals after'
                         ' {:.2f}'.format(ssv_id, obj_type, dtime))
        if len(mesh) == 2:
            return ""
        try:
            return b"".join(mesh[2])
        except TypeError:  # contains str, not byte
            return "".join(mesh[2])

    def ssv_list(self):
        """
        Returns all ssvs in dataset.
        :return: dict
        """
        return {'ssvs': list(self.ssd.ssv_ids)}

    def ssv_of_sv(self, sv_id):
        """
        Returns the ssv for a given sv_id.
        :param sv_id: 
        :return: 
        """
        return {'ssv': self.ssd.id_changer[int(sv_id)]}

    def ct_of_ssv(self, ssv_id):
        """
        Returns the CT for a given SSV ID.
        :param sv_id:
        :return:
        """
        ssv = self.ssd.get_super_segmentation_object(int(ssv_id))
        ssv.load_attr_dict()
        label = ""
        if "celltype_cnn_probas" in ssv.attr_dict:
            l = ssv.attr_dict["celltype_cnn"]
            ct_label_dc = {0: "EA", 1: "MSN", 2: "GP", 3: "INT"}
            label = ct_label_dc[l]
        else:
            log_gate.debug("Celltype prediction not present in attribute "
                           "dict of SSV {}at {}.".format(ssv_id, ssv.attr_dict_path))
        return {'ct': label}

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
        for k, v in all_syn_meta_dict.items():
            all_syn_meta_dict[k] = v.tolist()

        return all_syn_meta_dict

    def syn_objs_of_ssv_pre_post(self, ssv_id):
        """
        TODO: Requires adaptions of 'SyConnBackend' class
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
        TODO: Requires adaptions of 'SyConnBackend' class
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

        self.logger = log_gate

        self.logger.info('SyConn gate server starting up.')
        self.backend = SyConnBackend(global_params.wd, logger=self.logger)
        self.logger.info('SyConn gate server running.')
        return


class MyEncoder(json.JSONEncoder):
    """
    From https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

"""
Alternative way of running the server is currently:
export FLASK_APP=server.py
flask run --host=0.0.0.0 --port=10001 --debugger

OR

FLASK_APP=server.py FLASK_DEBUG=1 flask run --host=0.0.0.0 --port 10001

"""
sg_state = ServerState()

# context = ('cert.crt', 'key.key') enable later
app.run(host='0.0.0.0',  # do not run this on a non-firewalled machine!
       port=10001, # ssl_context=context,
       threaded=True, debug=True, use_reloader=True)
