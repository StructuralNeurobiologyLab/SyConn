# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from PythonQt import QtGui, Qt, QtCore
from PythonQt.QtGui import QTableWidget, QTableWidgetItem
import traceback
try:
    import KnossosModule
except ImportError:
    import knossos as KnossosModule
import sys
import requests
import re
import json
sys.dont_write_bytecode = True
import time
from multiprocessing.pool import ThreadPool
from Queue import Queue
from threading import Thread
import numpy as np
try:
    try:
        from lz4.block import compress, decompress
    except ImportError:
        from lz4 import compress, decompress
except ImportError:
    print("lz4 could not be imported. Locking will be disabled by default."
          "Please install lz4 to enable locking (pip install lz4).")


class SyConnGateInteraction(object):
    """
    Query the SyConn backend server.
    """
    def __init__(self, server, synthresh=0.5, axodend_only=True):
        self.server = server
        self.session = requests.Session()
        self.ssv_from_sv_cache = dict()
        self.ct_from_cache = dict()
        self.ctcertain_from_cache = dict()
        self.svs_from_ssv = dict()
        self.synthresh = synthresh
        self.axodend_only = axodend_only

        self.get_download_queue = Queue()
        self.init_get_download_queue_worker()
        self.get_download_done = Queue()
        self.get_download_results_store = dict()

    def get_ssv_mesh(self, ssv_id):
        """
        Returns a mesh for a given ssv_id.
        Parameters
        ----------
        ssv_id

        Returns
        -------

        """
        r1 = self.session.get(self.server + '/ssv_ind/{0}'.format(ssv_id))
        r2 = self.session.get(self.server + '/ssv_vert/{0}'.format(ssv_id))
        r3 = self.session.get(self.server + '/ssv_norm/{0}'.format(ssv_id))
        ind = lz4stringtoarr(r1.content, dtype=np.uint32)
        vert = lz4stringtoarr(r2.content, dtype=np.float32)
        norm = lz4stringtoarr(r3.content, dtype=np.float32)
        if len(norm) == 0:
            norm = []
        return ind, vert, norm

    def get_ssv_skel(self, ssv_id):
        """
        Returns a skeleton for a given ssv_id.

        Parameters
        ----------
        ssv_id : int

        Returns
        -------
        dict
            Keys: "nodes", "edges", "diameters"
        """
        r = self.session.get(self.server + '/ssv_skeleton/{0}'.format(ssv_id))
        skel = json.loads(r.content)
        skel["nodes"] = np.array(skel["nodes"], dtype=np.uint32).reshape(-1, 3)
        skel_nodes = np.array(skel["nodes"])
        skel["nodes"][:, 0] = skel_nodes[:, 1]
        skel["nodes"][:, 1] = skel_nodes[:, 0]
        skel["edges"] = np.array(skel["edges"], dtype=np.uint32).reshape(-1, 2)
        for k in skel:
            if k in ['nodes', 'edges']:
                continue
            skel[k] = np.array(skel[k], dtype=np.float32)
        return skel if len(skel) > 0 else None

    def init_get_download_queue_worker(self):
        """
        Initialize mesh queue daemon workers.

        :return:
        """
        for i in range(20):
            worker = Thread(target=self.get_download_queue_worker)
            worker.setDaemon(True)
            worker.start()

        return

    def wait_for_all_downloads(self):
        while not self.get_download_done.empty():
            time.sleep(0.05)

    def get_download_queue_worker(self):
        while True:
            # this is blocking and therefore fine
            get_request = self.get_download_queue.get()
            r = self.session.get(self.server + get_request)
            self.get_download_results_store[get_request] = r
            self.get_download_queue.task_done() # not sure whether this is needed
            _ = self.get_download_done.get() # signal download done by removal

        return

    def add_ssv_obj_mesh_to_down_queue(self, ssv_id, obj_type):
        # if this queue is empty, all downloads will be done,
        # a poor man's sync mechanism
        for i in range(3):
            self.get_download_done.put('working')

        self.get_download_queue.put('/ssv_obj_ind/{0}/{1}'.format(ssv_id, obj_type))
        self.get_download_queue.put('/ssv_obj_vert/{0}/{1}'.format(ssv_id, obj_type))
        self.get_download_queue.put('/ssv_obj_norm/{0}/{1}'.format(ssv_id, obj_type))

    def get_ssv_obj_mesh_from_results_store(self, ssv_id, obj_type):
        ind_hash = '/ssv_obj_ind/{0}/{1}'.format(ssv_id, obj_type)
        vert_hash = '/ssv_obj_vert/{0}/{1}'.format(ssv_id, obj_type)
        norm_hash = '/ssv_obj_norm/{0}/{1}'.format(ssv_id, obj_type)

        #start = time.time()
        ind = lz4stringtoarr(self.get_download_results_store[ind_hash].content, dtype=np.uint32)
        vert = lz4stringtoarr(self.get_download_results_store[vert_hash].content, dtype=np.float32)
        norm = lz4stringtoarr(self.get_download_results_store[norm_hash].content, dtype=np.float32)
        #print('lz4 decompress took {}'.format(time.time()-start))
        # clean up - could also be extended into some more permanent results cache
        self.get_download_results_store.pop(ind_hash, None)
        self.get_download_results_store.pop(vert_hash, None)
        self.get_download_results_store.pop(norm_hash, None)

        return ind, vert, -norm  # invert normals

    def get_ssv_obj_mesh(self, ssv_id, obj_type):
        """
        Returns a mesh for a given ssv_id and a specified obj_type.
        obj_type can be sj, vc, mi ATM.
        Parameters
        ----------
        ssv_id
        obj_type

        Returns
        -------

        """
        #thread_pool = ThreadPool(processes=3)

        #result = thread_pool.map(self.get_mesh_fragment,
        #                         [(sv_id, frag_key) for frag_key in
        #                          fragment_keys])

        #thread_pool.close()
        #thread_pool.join()
        r1 = self.session.get(self.server + '/ssv_obj_ind/{0}/{1}'.format(ssv_id,
                                                                          obj_type))
        r2 = self.session.get(self.server + '/ssv_obj_vert/{0}/{1}'.format(ssv_id,
                                                                          obj_type))
        r3 = self.session.get(self.server + '/ssv_obj_norm/{0}/{1}'.format(ssv_id,
                                                                          obj_type))
        ind = lz4stringtoarr(r1.content, dtype=np.uint32)
        vert = lz4stringtoarr(r2.content, dtype=np.float32)
        norm = lz4stringtoarr(r3.content, dtype=np.float32)
        return ind, vert, -norm  # invert normals

    def get_list_of_all_ssv_ids(self):
        """
        Returns a list of all ssvs in the dataset
        Returns
        -------

        """
        r = self.session.get(self.server + '/ssv_list')
        return json.loads(r.content)

    def get_svs_of_ssv(self, ssv_id):
        """
        Returns a list of all svs of a given ssv.
        Parameters
        ----------
        ssv_id

        Returns
        -------

        """
        if ssv_id not in self.svs_from_ssv:
            r = self.session.get(self.server + '/svs_of_ssv/{0}'.format(ssv_id))
            self.svs_from_ssv[ssv_id] = json.loads(r.content)
        return self.svs_from_ssv[ssv_id]

    def get_ssv_of_sv(self, sv_id):
        """
        Gets the ssv for a given sv.
        Parameters
        ----------
        sv_id

        Returns
        -------

        """
        if sv_id not in self.ssv_from_sv_cache:
            start = time.time()
            r = self.session.get(self.server + '/ssv_of_sv/{0}'.format(sv_id))
            self.ssv_from_sv_cache[sv_id] = json.loads(r.content)
            print('Get ssv of sv {} without cache took {}'.format(sv_id, time.time() - start))
        return self.ssv_from_sv_cache[sv_id]

    def get_celltype_of_ssv(self, ssv_id):
        """
        Get SSV cell type if available.

        Parameters
        ----------
        ssv_id : int

        Returns
        -------
        str

        """
        # if not ssv_id in self.ct_from_cache:
        r = self.session.get(self.server + '/ct_of_ssv/{0}'.format(ssv_id))
        dc = json.loads(r.content)
        self.ct_from_cache[ssv_id] = dc["ct"]
        if 'certainty' in dc:
            certainty = '{:.3f}'.format(dc["certainty"])
        else:
            certainty = 'nan'
        self.ctcertain_from_cache[ssv_id] = certainty
        print("Celltype: {}".format(self.ct_from_cache[ssv_id]))
        return self.ct_from_cache[ssv_id], certainty

    def get_all_syn_metda_data(self):
        """

        Returns
        -------

        """
        params = {'synthresh': self.synthresh, 'axodend_only': self.axodend_only}
        r = self.session.get('{}/all_syn_meta/{}'.format(self.server, json.dumps(params)))
        return json.loads(r.content)

    def push_so_attr(self, so_id, so_type, attr_key, attr_value):
        """
        Will invoke `so.save_attributes([attr_key], [attr_value)` of
        `so = SegmentationDataset(obj_type=so_type).get_segmentation_object(so_id)`
        on the server.

        Parameters
        ----------
        so_id :
        so_type :
        attr_key :
        attr_value :

        Returns
        -------
        str | bytes
            Server response
        """
        r = self.session.get(self.server + '/push_so_attr/{}/{}/{}/{}'.format(
            so_id, so_type, attr_key, attr_value))
        return r.content

    def pull_so_attr(self, so_id, so_type, attr_key):
        """
        Will invoke `so.save_attributes([attr_key], [attr_value)` of
        `so = SegmentationDataset(obj_type=so_type).get_segmentation_object(so_id)`
        on the server.

        Parameters
        ----------
        so_id :
        so_type :
        attr_key :

        Returns
        -------
        str | bytes
            Server response
        """
        r = self.session.get(self.server + '/pull_so_attr/{}/{}/{}'.format(
            so_id, so_type, attr_key))
        return r.content


class InputDialog(QtGui.QDialog):
    """
    https://stackoverflow.com/questions/7046882/launch-a-pyqt-window-from-a-main-pyqt-window-and-get-the-user-input

    inputter = InputDialog(mainWindowUI, title="comments", label="comments", text="")
    inputter.exec_()
    comment = inputter.text.text()
    print comment
    """

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.aborted = False
        # --Layout Stuff---------------------------#
        mainLayout = QtGui.QVBoxLayout()

        layout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel()
        self.label.setText("port")
        layout.addWidget(self.label)
        self.text = QtGui.QLineEdit("10001")
        layout.addWidget(self.text)

        self.ip = QtGui.QLabel()
        self.ip.setText("host")
        layout.addWidget(self.ip)
        self.text_ip = QtGui.QLineEdit("localhost")
        layout.addWidget(self.text_ip)
        mainLayout.addLayout(layout)

        layout = QtGui.QHBoxLayout()
        self.synapse_tresh = QtGui.QLabel()
        self.synapse_tresh.setText("Syn. prob. thresh.")
        layout.addWidget(self.synapse_tresh)
        self.text_synthresh = QtGui.QLineEdit("0.6")
        layout.addWidget(self.text_synthresh)

        self.axodend_button = QtGui.QPushButton("Axo-dendr. syn. only")
        self.axodend_button.setCheckable(True)
        self.axodend_button.toggle()
        layout.addWidget(self.axodend_button)

        mainLayout.addLayout(layout)

        # --The Button------------------------------#
        layout = QtGui.QHBoxLayout()
        button = QtGui.QPushButton("connect")  # string or icon
        self.connect(button, QtCore.SIGNAL("clicked()"), self.close)
        layout.addWidget(button)

        button = QtGui.QPushButton("abort")  # string or icon
        self.connect(button, QtCore.SIGNAL("clicked()"), self.abort_button_clicked)
        layout.addWidget(button)

        mainLayout.addLayout(layout)
        self.setLayout(mainLayout)

        self.resize(450, 300)
        self.setWindowTitle("SyConnGate Settings")

    def abort_button_clicked(self):
        print('Closing SyConnGate.')
        self.aborted = True
        self.close()


class main_class(QtGui.QDialog):
    """
    KNOSSOS plugin class for the SyConn KNOSSOS viewer.
    """
    def __init__(self, parent=KnossosModule.knossos_global_mainwindow):
        #Qt.QApplication.processEvents()
        super(main_class, self).__init__(parent, Qt.Qt.WA_DeleteOnClose)
        try:
            exec(KnossosModule.scripting.getInstanceInContainerStr(__name__)
                 + " = self")
        except KeyError:
            # Allow running from __main__ context
            pass

        # get port
        while True:
            inputter = InputDialog(parent)
            inputter.exec_()
            if inputter.aborted:
                return
            port = int(inputter.text.text.decode())
            host = inputter.text_ip.text.decode()
            self._synthresh = float(inputter.text_synthresh.text.decode())
            self._axodend_only = inputter.axodend_button.isChecked()
            self.syconn_gate = None
            self.host = host
            self.port = port
            self.ssv_selected1 = 0
            self.syn_selected1 = None
            self.obj_tree_ids = set()
            self.obj_id_offs = 2000000000
            self.all_syns = None
            try:
                self.init_syconn()
                self.build_gui()
                #self.timer = QtCore.QTimer()
                #self.timer.timeout.connect(self.exploration_mode_callback_check)
                #self.timer.start(1000)

                #self.timer2 = QtCore.QTimer()
                #self.timer2.timeout.connect(self.release_gil_hack)
                #self.timer2.start(50)

                break
            except requests.exceptions.ConnectionError as e:
                print("Failed to establish connection to SyConn Server.", str(e))
                pass

    def release_gil_hack(self):
        time.sleep(0.01)
        return

    def init_syconn(self):
        # move to config file
        syconn_gate_server = 'http://{}:{}'.format(self.host, self.port)
        self.syconn_gate = SyConnGateInteraction(syconn_gate_server,
                                                 self._synthresh,
                                                 self._axodend_only)

    def populate_ssv_list(self):
        all_ssv_ids = self.syconn_gate.get_list_of_all_ssv_ids()['ssvs']

        #print('list of all here')
        #print(len(all_ssv_ids))
        for ssv_id in all_ssv_ids:
            #print(ssv_id)
            item = QtGui.QStandardItem(str(int(ssv_id)))
            self.ssv_item_model.appendRow(item)

        self.ssv_selector.setModel(self.ssv_item_model)
        return

    def populate_syn_list(self):
        self.all_syns = self.syconn_gate.get_all_syn_metda_data()

        #print('list of all here')
        #print(len(all_ssv_ids))
        for syn in zip(self.all_syns['ssv_partner_0'], self.all_syns['ssv_partner_1']):
            item = QtGui.QStandardItem(str(syn))
            self.syn_item_model.appendRow(item)
        self.syn_selector.setModel(self.syn_item_model)
        return

    def on_ssv_selector_changed(self, index):
        self.ssv_selected1 = int(self.ssv_selector.model().itemData(index)[0])
        #current, previous
        #print('selected: ' + str(self.ssv_selector.model().itemData(index)[0]))
        #ssv = self.get_ssv(self.ssv_selected1)
        #self.ssv_to_knossos(ssv)
        return

    def on_syn_selector_changed(self, index, signal_block=True):
        """
        `all_syns` contains the following keys:
    cd_dict['syn_size'] =\
        csd.load_cached_data('mesh_area') / 2  # as used in syn_analysis.py -> export_matrix
    cd_dict['synaptivity_proba'] = \
        csd.load_cached_data('syn_prob')
    cd_dict['coord_x'] = \
        csd.load_cached_data('rep_coord')[:, 0].astype(np.int)
    cd_dict['coord_y'] = \
        csd.load_cached_data('rep_coord')[:, 1].astype(np.int)
    cd_dict['coord_z'] = \
        csd.load_cached_data('rep_coord')[:, 2].astype(np.int)
    cd_dict['ssv_partner_0'] = \
        csd.load_cached_data('neuron_partners')[:, 0].astype(np.int)
    cd_dict['ssv_partner_1'] = \
        csd.load_cached_data('neuron_partners')[:, 1].astype(np.int)
    cd_dict['neuron_partner_ax_0'] = \
        csd.load_cached_data('partner_axoness')[:, 0].astype(np.int)
    cd_dict['neuron_partner_ax_1'] = \
        csd.load_cached_data('partner_axoness')[:, 1].astype(np.int)
    cd_dict['neuron_partner_ct_0'] = \
        csd.load_cached_data('partner_celltypes')[:, 0].astype(np.int)
    cd_dict['neuron_partner_ct_1'] = \
        csd.load_cached_data('partner_celltypes')[:, 1].astype(np.int)
    cd_dict['neuron_partner_sp_0'] = \
        csd.load_cached_data('partner_spiness')[:, 0].astype(np.int)
    cd_dict['neuron_partner_sp_1'] = \
        csd.load_cached_data('partner_spiness')[:, 1].astype(np.int)

        Parameters
        ----------
        index :
        signal_block :

        Returns
        -------

        """
        # disable knossos signal emission first - O(n^2) otherwise
        if signal_block:
            signalsBlocked = KnossosModule.knossos_global_skeletonizer.blockSignals(
                True)

        inp_str = self.syn_selector.model().itemData(index)[0]
        ssv1 = int(re.findall(r'\((\d+),', inp_str)[0])
        ssv2 = int(re.findall(r', (\d+)\)', inp_str)[0])
        ix = index.row()
        tree_id = hash((ssv1, ssv2))
        syn_id = self.all_syns['ids'][ix]
        self._currently_active_syn = syn_id
        # TODO: pull_so_attr and writing its results to `synapsetype_label_text` should run as a thread
        syn_gt_syntype = self.syconn_gate.pull_so_attr(so_id=syn_id, so_type='syn_ssv',
                                                       attr_key='gt_syntype')
        if len(syn_gt_syntype) == 0:
            self.synapsetype_label_text.clear()
        else:
            self.synapsetype_label_text.setText(syn_gt_syntype)
        c = [self.all_syns['coord_x'][ix], self.all_syns['coord_y'][ix],
             self.all_syns['coord_z'][ix]]

        k_tree = KnossosModule.skeleton.find_tree_by_id(tree_id)
        if k_tree is None:
            k_tree = KnossosModule.skeleton.add_tree(tree_id)
        # add synapse location
        kn = KnossosModule.skeleton.add_node([c[0] + 1, c[1] + 1, c[2] + 1],
                                            k_tree, {})
        KnossosModule.skeleton.jump_to_node(kn)

        # syn properties
        syn_size = self.all_syns["syn_size"][ix]
        syn_size = np.abs(syn_size)
        # coordinate
        self.synapse_field1.setItem(0, 1, QTableWidgetItem(str(c)))
        # synapse type
        self.synapse_field1.setItem(1, 1, QTableWidgetItem(str(self.all_syns['syn_sign'][ix])))
        # synaptic probability
        self.synapse_field1.setItem(2, 1, QTableWidgetItem(str(self.all_syns["synaptivity_proba"][ix])))
        # synaptic size (area in um^2)
        self.synapse_field1.setItem(3, 1, QTableWidgetItem(str(syn_size)))
        # object ID
        self.synapse_field1.setItem(4, 1, QTableWidgetItem(str(syn_id)))

        # pre- and post synaptic properties
        # IDs
        self.synapse_field2.setItem(1, 1, QTableWidgetItem(str(self.all_syns["ssv_partner_0"][ix])))
        self.synapse_field2.setItem(1, 2, QTableWidgetItem(str(self.all_syns["ssv_partner_1"][ix])))

        # cell type
        self.synapse_field2.setItem(2, 1, QTableWidgetItem(int2str_label_converter(self.all_syns["neuron_partner_ct_0"][ix], "ctgt_v2")))
        self.synapse_field2.setItem(2, 2, QTableWidgetItem(int2str_label_converter(self.all_syns["neuron_partner_ct_1"][ix], "ctgt_v2")))

        # cell compartments
        self.synapse_field2.setItem(3, 1, QTableWidgetItem(int2str_label_converter(self.all_syns["neuron_partner_ax_0"][ix], "axgt")))
        self.synapse_field2.setItem(3, 2, QTableWidgetItem(int2str_label_converter(self.all_syns["neuron_partner_ax_1"][ix], "axgt")))

        # cell compartments
        self.synapse_field2.setItem(4, 1, QTableWidgetItem(int2str_label_converter(self.all_syns["neuron_partner_sp_0"][ix], "spgt")))
        self.synapse_field2.setItem(4, 2, QTableWidgetItem(int2str_label_converter(self.all_syns["neuron_partner_sp_1"][ix], "spgt")))

        # enable signals again
        if signal_block:
            KnossosModule.knossos_global_skeletonizer.blockSignals(
                signalsBlocked)
            KnossosModule.knossos_global_skeletonizer.resetData()
        return

    def build_gui(self):
        self.setWindowFlags(Qt.Qt.Window)
        layout = QtGui.QGridLayout()
        layout.setSpacing(10)

        # Window layout
        #layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        self.show_button_neurite = QtGui.QPushButton('Show neurite')
        self.show_button_selected_neurite = QtGui.QPushButton('Add selected neurite(s)')
        self.show_button_synapse = QtGui.QPushButton('Show synapse')
        self.clear_knossos_view_button = QtGui.QPushButton('Clear view')

        self.ssv_selector = QtGui.QListView()
        self.ssv_selector.setUniformItemSizes(True)  # better performance
        self.ssv_item_model = QtGui.QStandardItemModel(self.ssv_selector)

        self.syn_selector = QtGui.QListView()
        self.syn_selector.setUniformItemSizes(True)  # better performance
        self.syn_item_model = QtGui.QStandardItemModel(self.syn_selector)

        self.direct_ssv_id_input = QtGui.QLineEdit()
        self.direct_ssv_id_input.setValidator(QtGui.QIntValidator())
        self.direct_ssv_id_input.setMaxLength(8)

        self.direct_syn_id_input = QtGui.QLineEdit()
        self.direct_syn_id_input.setValidator(QtGui.QIntValidator())
        self.direct_syn_id_input.setMaxLength(8)

        # celltype
        self.celltype_field = QtGui.QLabel("CellType:      ", self)

        # synapse
        self.synapse_field1 = QTableWidget()
        self.synapse_field1.setRowCount(5)
        self.synapse_field1.setColumnCount(2)
        self.synapse_field1.setItem(0, 0, QTableWidgetItem("coordinate"))
        self.synapse_field1.setItem(0, 1, QTableWidgetItem(""))
        self.synapse_field1.setItem(1, 0, QTableWidgetItem("synaptic type"))
        self.synapse_field1.setItem(1, 1, QTableWidgetItem(""))
        self.synapse_field1.setItem(2, 0, QTableWidgetItem("syn. proba."))
        self.synapse_field1.setItem(2, 1, QTableWidgetItem(""))
        self.synapse_field1.setItem(3, 0, QTableWidgetItem("size [um^2]"))
        self.synapse_field1.setItem(3, 1, QTableWidgetItem(""))
        self.synapse_field1.setItem(4, 0, QTableWidgetItem("Object ID"))
        self.synapse_field1.setItem(4, 1, QTableWidgetItem(""))
        # self.synapse_field1.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)  # qt5
        header = self.synapse_field1.horizontalHeader()
        header.setSectionResizeMode(0, QtGui.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtGui.QHeaderView.ResizeToContents)
        self.synapse_field1.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)

        self.synapse_field2 = QTableWidget()
        self.synapse_field2.setRowCount(5)
        self.synapse_field2.setColumnCount(3)
        # TODO: sort by pre and post in 'on_syn_selector_changed' and replace neuron1 and neuron2 by pre and post
        self.synapse_field2.setItem(0, 1, QTableWidgetItem("neuron 1"))
        self.synapse_field2.setItem(0, 2, QTableWidgetItem("neuron 2"))
        self.synapse_field2.setItem(1, 0, QTableWidgetItem("SSV ID"))
        self.synapse_field2.setItem(2, 0, QTableWidgetItem("cell type"))
        self.synapse_field2.setItem(3, 0, QTableWidgetItem("cell comp."))
        self.synapse_field2.setItem(4, 0, QTableWidgetItem("spiness"))
        # self.synapse_field2.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)  # qt5
        self.synapse_field2.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        header = self.synapse_field2.horizontalHeader()
        header.setSectionResizeMode(0, QtGui.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtGui.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtGui.QHeaderView.ResizeToContents)

        self.send_synapsetype_label_button = QtGui.QPushButton('Send')

        self.synapsetype_label = QtGui.QLabel()
        self.synapsetype_label.setText("Synapse type label [-1: inhib.; 0: non-syn.; 1: "
                                       "excit.]:")
        self.synapsetype_label_text = QtGui.QLineEdit()
        self.send_button_response_label = QtGui.QLabel()
        self.send_button_response_label.setText(None)

        #self.exploration_mode_chk_box = QtGui.QCheckBox('Exploration mode')
        #self.exploration_mode_chk_box.setChecked(True)
        #self.ssv_selection_model =
        # QtGui.QItemSelectionModel(self.ssv_select_model)

        #self.selectionModel =
        # self.ssv_selector.selectionModel(self.ssv_selector)
        #self.ssv_selector.setSelectionModel(self.ssv_selection_model)
        #print('selection model: ' + str(self.ssv_selector.selectionModel()))

        self.ssv_selector.clicked.connect(self.on_ssv_selector_changed)
        self.syn_selector.clicked.connect(self.on_syn_selector_changed)

        self.populate_ssv_list()

        self.populate_syn_list()
        print('Connected to SyConnGate.')

        layout.addWidget(self.direct_ssv_id_input, 1, 0, 1, 1)
        layout.addWidget(self.direct_syn_id_input, 1, 1, 1, 1)
        layout.addWidget(self.ssv_selector, 2, 0, 1, 1)
        layout.addWidget(self.syn_selector, 2, 1, 1, 1)
        layout.addWidget(self.show_button_neurite, 3, 0, 1, 1)
        layout.addWidget(self.show_button_synapse, 3, 1, 1, 1)
        layout.addWidget(self.clear_knossos_view_button, 4, 0, 1, 1)
        layout.addWidget(self.show_button_selected_neurite, 5, 0, 1, 1)
        layout.addWidget(self.celltype_field, 1, 2, 1, 2)

        layout.addWidget(self.synapse_field1, 2, 2, 1, 1)
        layout.addWidget(self.synapse_field2, 3, 2, 1, 1)
        layout.addWidget(self.synapsetype_label, 4, 1, 1, 1)
        layout.addWidget(self.synapsetype_label_text, 4, 2, 1, 2)
        layout.addWidget(self.send_button_response_label, 5, 1, 1, 1)
        layout.addWidget(self.send_synapsetype_label_button, 5, 2, 1, 1)

        #self.ssv_select_model.itemChanged.connect(self.on_ssv_selector_changed)
        #self.selectionModel.selectionChanged.connect(self.on_ssv_selector_changed)

        self.show_button_neurite.clicked.connect(self.show_button_neurite_clicked)
        self.show_button_selected_neurite.clicked.connect(self.show_button_selected_neurite_clicked)
        self.show_button_synapse.clicked.connect(self.show_button_synapse_clicked)
        self.clear_knossos_view_button.clicked.connect(self.clear_knossos_view_button_clicked)
        self.send_synapsetype_label_button.clicked.connect(self.send_synapsetype_label_button_clicked)
        #self.exploration_mode_chk_box.stateChanged.connect(self.exploration_mode_changed)

        # self.setGeometry(300, 300, 450, 300)
        self.setWindowTitle('SyConn Viewer v2')
        self.show()
        #self.merge_button = QtGui.QPushButton('Merge')
        #self.split_button = QtGui.QPushButton('Split')


#        QList < quint64 > subobjectIdsOfObject(const
 #       quint64
  #      objId);

   #     QList < quint64 > objects();
    #    QList < quint64 > selectedObjects();


        #self.bad_button = QtGui.QPushButton('Bad SV')
        #self.graph_split_button = QtGui.QPushButton('Graph split')
        #self.add_selected_sv_button = QtGui.QPushButton('Add selected SV')
        #self.mode_combo = QtGui.QComboBox()
        #self.stop_button = QtGui.QPushButton('Stop')
        #self.undo_button = QtGui.QPushButton('Undo')
        #self.redo_button = QtGui.QPushButton('Redo')

        #self.skip_task_line_edit = QtGui.QLineEdit('Skip reason')
        #self.server_line_edit = QtGui.QLineEdit()

        #self.password_line_edit = QtGui.QLineEdit()
        #  set echo mode to QLineEdit::PasswordEchoOnEdit

        #self.gui_auto_agglo_line_edit = QtGui.QLineEdit()
        #self.gui_auto_agglo_line_edit.setText('0')

    #def exploration_mode_changed(self):
    #    if self.exploration_mode_chk_box.isChecked():
    #        pass
            # enable selection polling timer
    #    else:
    #        pass
            # disable selection polling timer


    def exploration_mode_callback_check(self):
        #if self.exploration_mode_chk_box.isChecked():
            #print('expl')
        sel_seg_objs = KnossosModule.segmentation.selected_objects()
        if len(sel_seg_objs) == 0:
            return
        sel_sv_ids = []
        for sel_seg_obj in sel_seg_objs:
            sel_sv_ids.append(KnossosModule.segmentation.subobject_ids_of_object(sel_seg_obj)[0])


        trees = KnossosModule.skeleton.trees()
        ids_in_k = set([tree.tree_id() for tree in trees if
                        tree.tree_id() < self.obj_id_offs])
        # get selected ssv ids
        ssv_ids_selected = [self.syconn_gate.get_ssv_of_sv(sv_id)['ssv'] for sv_id in sel_sv_ids] #if not sv_id in ids_in_k



        # id_changer returns -1 for a supervoxel that is unconnected, add support for single svs
        ssv_ids_selected = [ssv_id for ssv_id in ssv_ids_selected if ssv_id != -1]

        #print('ssv_ids_selected {0}'.format(ssv_ids_selected))



        #print('self.obj_tree_ids {0}'.format(self.obj_tree_ids))
        #print('ids_in_k 1 {0}'.format(ids_in_k))

        #print('ids_in_k 2 {0}'.format(ids_in_k))

        # compare with the selected segmentation objects
        ids_selected = set(ssv_ids_selected)

        # add missing ones to knossos, delete if not needed anymore
        ids_to_add = ids_selected - ids_in_k
        ids_to_del = ids_in_k - ids_selected

        # remove segmentation objects that are not needed anymore
        all_objects = KnossosModule.segmentation.objects()

        objs_to_del = set(all_objects) - set(ids_selected)

        [KnossosModule.segmentation.remove_object(obj) for obj in objs_to_del]

        #print('ids to del {0} ids to add {1}'.format(ids_to_del, ids_to_add))

        #print('ids_selected {0}'.format(ids_selected))
        self.ids_selected = ids_selected

        [self.remove_ssv_from_knossos(ssv_id) for ssv_id in ids_to_del]
        [self.ssv_to_knossos(ssv_id) for ssv_id in ids_to_add]
        [self.ssv_skel_to_knossos_tree(ssv_id) for ssv_id in ids_to_add]
        [self.update_celltype(ssv_id) for ssv_id in ids_to_add]

        #if len(ids_in_k) != 1 or len(ids_to_del) > 0:
        #    [KnossosModule.skeleton.delete_tree(sv_id) for sv_id in
        #     self.obj_tree_ids]
        #    self.obj_tree_ids = set()

        return

    def remove_ssv_from_knossos(self, ssv_id):
        return
        KnossosModule.skeleton.delete_tree(ssv_id)
        # check whether there are object meshes that need to be deleted as well
        trees = KnossosModule.skeleton.trees()
        obj_mesh_ids = set([tree.tree_id() for tree in trees if
                        tree.tree_id() > self.obj_id_offs])
        for i in range(1, 4):
            obj_id_to_test = ssv_id + self.obj_id_offs + i
            if obj_id_to_test in obj_mesh_ids:
                KnossosModule.skeleton.delete_tree(obj_id_to_test)

    def show_button_selected_neurite_clicked(self):
        self.exploration_mode_callback_check()

    def show_button_neurite_clicked(self):
        try:
             ssvs = [x.strip() for x in self.direct_ssv_id_input.text.split(',')]
             ssvs = map(int, ssvs)
        except:
            ssvs = []
        for ssv in ssvs:
            self.ssv_to_knossos(ssv)
            self.ssv_skel_to_knossos_tree(ssv)
            self.update_celltype(ssv)
            self.ssv_selected1 = ssv
        return

    def show_button_synapse_clicked(self):
        try:
            self.syn_selected1 = int(self.direct_syn_id_input.text)
        except:
            pass
        # TODO
        if self.syn_selected1:
            # TODO: could be optimized: currently we need to get the index,
            #  and in on_syn_selector_changed the synapse ID is retrieved again
            syn_ix = self.syn_item_model.index(self.all_syns['ids'].index(self.syn_selected1), 0)
            self.on_syn_selector_changed(syn_ix)
        return

    def clear_knossos_view_button_clicked(self):
        # delete all existing objects in mergelist
        all_objects = KnossosModule.segmentation.objects()
        [KnossosModule.segmentation.remove_object(obj) for obj in all_objects]

        # iterate over all trees in knossos and delete
        trees = KnossosModule.skeleton.trees()
        ids_in_k = set([tree.tree_id() for tree in trees])
        [KnossosModule.skeleton.delete_tree(sv_id) for sv_id in ids_in_k]
        return

    def send_synapsetype_label_button_clicked(self):
        syntype_label = self.synapsetype_label_text.text.decode()
        if not syntype_label in ["-1", "0", "1"]:
            self.send_button_response_label.setText("INVALID LABEL '{}'".format(syntype_label))
        else:
            # TODO: parse syn_ssv ID from currently clicked synapse
            curr_syn_id = self._currently_active_syn
            r = self.syconn_gate.push_so_attr(so_id=str(curr_syn_id), so_type='syn_ssv',
                                              attr_key='gt_syntype_viewer',
                                              attr_value=syntype_label)
            if len(r) == 0:
                r = "push successful."
            self.send_button_response_label.setText(r)
        return

    def update_celltype(self, ssv_id):
        ct, certainty = self.syconn_gate.get_celltype_of_ssv(ssv_id)
        self.celltype_field.setText("CellType: {} ({})".format(ct, certainty))

    def ssv_to_knossos(self, ssv_id):
        start_tot = time.time()
        #self.clear_knossos_view_button_clicked()

        # to mergelist
        start = time.time()
        sv_ids = self.syconn_gate.get_svs_of_ssv(ssv_id)['svs']
        print('Get svs of ssv took {}'.format(time.time()-start))
        sv_ids = map(int, sv_ids)

        KnossosModule.segmentation.create_object(ssv_id, sv_ids[0], (1,1,1))
        #KnossosModule.segmentation.select_object(ssv.id)
        # query object should be red
        #KnossosModule.segmentation.changeColor(ssv_id, QtGui.QColor(255, 0, 0, 255))

        # one could cache this, not necessary to rebuild at every step
        for sv_id in sv_ids:
            try:
                KnossosModule.segmentation.add_subobject(ssv_id, sv_id)
            except:
                pass

        KnossosModule.segmentation.select_object(ssv_id)

        KnossosModule.segmentation.set_render_only_selected_objs(True)

        # create a 'fake' knossos tree for each obj mesh category;
        # this is very hacky since it can generate nasty ID collisions.
        mi_id = self.obj_id_offs + ssv_id + 1
        sym_id = self.obj_id_offs + ssv_id + 2
        asym_id = self.obj_id_offs + ssv_id + 3
        vc_id = self.obj_id_offs + ssv_id + 4
        neuron_id = self.obj_id_offs + ssv_id + 5

        params = [(self, ssv_id, neuron_id, 'sv', (128, 128, 128, 128)),
                  (self, ssv_id, mi_id, 'mi', (0, 153, 255, 255)),
                  (self, ssv_id, vc_id, 'vc', (int(0.175 * 255), int(0.585 * 255), int(0.301 * 255), 255)),
                  # (self, ssv_id, sj_id, 'sj', (240, 50, 50, 255))]
                  (self, ssv_id, sym_id, 'syn_ssv_sym', (50, 50, 240, 255)),
                  (self, ssv_id, asym_id, 'syn_ssv_asym', (240, 50, 50, 255))]
        start = time.time()

        # add all meshes to download queue
        for par in params:
            mesh_loader_threaded(*par)
        # wait for downloads
        self.syconn_gate.wait_for_all_downloads()
        print('Mesh download took {}'.format(time.time() - start))

        start = time.time()
        # add all to knossos
        for par in params:
            mesh_to_K(*par)
        print('Mesh to K took {}'.format(time.time() - start))
        return

    def ssv_skel_to_knossos_tree(self, ssv_id, signal_block=True):
        # disable knossos signal emission first - O(n^2) otherwise
        start = time.time()
        if signal_block:
            signalsBlocked = KnossosModule.knossos_global_skeletonizer.blockSignals(
                True)
        k_tree = KnossosModule.skeleton.find_tree_by_id(ssv_id)
        if k_tree is None:
            k_tree = KnossosModule.skeleton.add_tree(ssv_id)
        skel = self.syconn_gate.get_ssv_skel(ssv_id)
        #skel = None
        if skel is None:
            print("Loaded skeleton is None.")
            return
        # add nodes
        nx_knossos_id_map = dict()
        for ii, n_coord in enumerate(skel["nodes"]):
            # newsk_node.from_scratch(newsk_anno, nx_coord[1]+1, nx_coord[0]+1, nx_coord[2]+1, ID=nx_node)
            n_proeprties = {}
            for k in skel:
                if k in ["nodes", "edges", "diameters"]:
                    continue
                n_proeprties[k] = float(skel[k][ii])
            k_node = KnossosModule.skeleton.add_node(
                [n_coord[1] + 1, n_coord[0] + 1, n_coord[2] + 1], k_tree,
                n_proeprties)
            KnossosModule.skeleton.set_radius(k_node.node_id(),
                                              skel["diameters"][ii] / 2)
            nx_knossos_id_map[ii] = k_node.node_id()

        # add edges
        for nx_src, nx_tgt in skel["edges"]:
            KnossosModule.skeleton.add_segment(nx_knossos_id_map[nx_src],
                                               nx_knossos_id_map[nx_tgt])

        # enable signals again
        if signal_block:
            KnossosModule.knossos_global_skeletonizer.blockSignals(
                signalsBlocked)
            KnossosModule.knossos_global_skeletonizer.resetData()
        print('Skel down and to K took {}'.format(time.time()-start))
        return


def mesh_loader(gate_obj, ssv_id, tree_id, obj_type, color):
    start = time.time()
    mesh = gate_obj.syconn_gate.get_ssv_obj_mesh(ssv_id, obj_type)
    print("Download time:", time.time() - start)
    start = time.time()
    if len(mesh[0]) > 0:
        KnossosModule.skeleton.add_tree_mesh(tree_id, mesh[1], mesh[2],
                                             mesh[0],
                                             [], 4, False)
        KnossosModule.skeleton.set_tree_color(tree_id,
                                              QtGui.QColor(*color))
    print("Loading {}-mesh time (pure KNOSSOS): {:.2f} s".format(
        obj_type, time.time() - start))


def mesh_loader_threaded(gate_obj, ssv_id, tree_id, obj_type, color):
    gate_obj.syconn_gate.add_ssv_obj_mesh_to_down_queue(ssv_id, obj_type)


def mesh_to_K(gate_obj, ssv_id, tree_id, obj_type, color):
    mesh = gate_obj.syconn_gate.get_ssv_obj_mesh_from_results_store(ssv_id, obj_type)
    if len(mesh[0]) > 0:
        KnossosModule.skeleton.add_tree_mesh(tree_id, mesh[1], mesh[2],
                                             mesh[0],
                                             [], 4, False)
        KnossosModule.skeleton.set_tree_color(tree_id,
                                              QtGui.QColor(*color))


def lz4stringtoarr(string, dtype=np.float32, shape=None):
    """
    Converts lz4 compressed string to 1d array. Moved here to circumvent
    a syconn dependency.

    Parameters
    ----------
    string : str
    dtype : np.dtype
    shape : tuple

    Returns
    -------
    np.array
        1d array
    """
    if string == "":
        return np.zeros((0, ), dtype=dtype)
    try:
        arr_1d = np.frombuffer(decompress(string), dtype=dtype)
    except Exception as e:
        print(str(e) + "\nString length:" + str(len(string)))
        return np.zeros((0,), dtype=dtype)
    if shape is not None:
        arr_1d = arr_1d.reshape(shape)
    return arr_1d


def int2str_label_converter(label, gt_type):
    """
    Converts integer label into semantic string.

    Parameters
    ----------
    label : int
    gt_type : str
        e.g. spgt for spines, axgt for cell compartments or ctgt for cell type

    Returns
    -------
    str
    """
    if type(label) is list:
        if len(label) != 1:
            raise ValueError('Multiple labels given.')
        label = label[0]
    if gt_type == "axgt":
        if label == 1:
            return "axon"
        elif label == 0:
            return "dendrite"
        elif label == 2:
            return "soma"
        else:
            return "N/A"
    elif gt_type == "spgt":
        if label == 1:
            return "head"
        elif label == 0:
            return "neck"
        elif label == 2:
            return "shaft"
        elif label == 3:
            return "other"
        else:
            return "N/A"
    elif gt_type == 'ctgt':
        if label == 1:
            return "MSN"
        elif label == 0:
            return "EA"
        elif label == 2:
            return "GP"
        elif label == 3:
            return "INT"
        else:
            return "N/A"
    elif gt_type == 'ctgt_v2':
        l_dc_inv = dict(STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6)
        l_dc = {v: k for k, v in l_dc_inv.items()}
        try:
            return l_dc[label]
        except KeyError:
            print('Unknown label "{}"'.format(label))
            return "N/A"
    elif gt_type == 'ctgt_v2_old':
        l_dc_inv = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, FS=6, TAN=7)
        l_dc_inv["?"] = 8
        l_dc = {v: k for k, v in l_dc_inv.items()}
        # Do not distinguish between FS and INT/?
        l_dc[8] = "INT"
        l_dc[6] = "INT"
        try:
            return l_dc[label]
        except KeyError:
            print('Unknown label "{}"'.format(label))
    else:
        raise ValueError("Given ground truth type is not valid.")


if __name__=='__main__':
    A = main_class()
