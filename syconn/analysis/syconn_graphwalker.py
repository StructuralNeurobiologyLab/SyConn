from PythonQt import QtGui, Qt, QtCore
import KnossosModule
import os
import time
import json
import networkx as nx
from networkx.readwrite import json_graph
import heapq
import requests
import re
import zmq
import copy
import datetime
from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials
import json
from multiprocessing.pool import ThreadPool
import tempfile
from Queue import Queue
from threading import Thread
import cStringIO as StringIO
import shutil

#KNOSSOS_PLUGIN	Version	2
#KNOSSOS_PLUGIN	Description	Presegmentation correction workflow plugin

########################################################################################################################
# helper functions follow - these can be useful outside of the plugin scope
########################################################################################################################

class NeuroDockInteraction():
    def __init__(self, server='localhost', port=19998, timeout=4000):
        self.server = server
        self.port = port
        self.timeout = timeout
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.sock.connect("tcp://{0}:{1}".format(server, port))

    def send(self, msg):
        self.sock.send(msg)

    def receive(self):
        return self.sock.recv()

    def get_decision(self, sv_id1, sv_id2, coord):
        query_str = '{0} {1} {2} {3} {4}'.format(sv_id1, sv_id2, coord[0], coord[1], coord[2])
        #print('Querying Neurodocker with: ' + query_str)
        self.send(query_str)
        try:
            probability = float(self.receive())
            #print('Got from Neurodocker: ' + str(probability))
        except zmq.Again:
            probability = 0.
            print('Response from Neurodocker timed out, resetting socket.')
            self.reset_socket()

        return probability

    def reset_socket(self):
        self.sock.close()
        self.sock = self.context.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.sock.connect("tcp://{0}:{1}".format(self.server, self.port))


class UndoState(object):
    def __init__(self,
                 curr_edge,
                 curr_heap,
                 gw_object):

        self.curr_edge = copy.deepcopy(curr_edge)
        self.curr_heap = copy.deepcopy(curr_heap)
        self.gw_object = gw_object
        return


class AAMInteraction(object):
    def __init__(self, server):

        self.server = server
        self.session = requests.Session()

    def login(self, username, password):
        post_txt = '<login><username>%s</username><password>%s</password></login>' % (username, password, )
        r = self.session.post(self.server + 'api/2/login', post_txt)
        return r

    def logout(self):
        r = self.session.get(self.server + 'api/2/logout')
        return r

    def session_state(self):
        r = self.session.get(self.server + 'api/2/session')
        print(r)
        print(r.content)

    def submit(self, json_gw='', worktime = 0, gw_log = ''):

        post = {
            'json_gw' : json_gw,
            'worktime' : worktime,
            'log' : gw_log, # .encode('zlib_codec') compression and encoding trouble
            'csrfmiddlewaretoken': self.session.cookies['csrftoken'], }

        r = self.session.post(self.server + 'graphwalker/submit_task', post)

        if r.status_code == 201:
            return True
        else:
            return False

    def start_new_rec(self, provided_sv_id=0):
        """
        Starts a new reconstruction on the backend side. A sv_id can be provided, but the backend decides finally
        whether this id is still available and provides the new seed id, as well as the unique reconstruction id.

        :param provided_sv_id: int
        :return: response
        """

        post_data = {
            'provided_id' : provided_sv_id,
            'csrfmiddlewaretoken': self.session.cookies['csrftoken']}

        # posts the data in form format
        r = self.session.post(self.server + 'graphwalker/new_task', data = post_data)

        return r

    def restart_task(self):
        """
        Restarts the currently active reconstruction in the AAM.

        :return:
        """

        post_data = {
            'csrfmiddlewaretoken': self.session.cookies['csrftoken']}

        # posts the data in form format
        r = self.session.post(self.server + 'graphwalker/restart_task', data = post_data)

        return r

    def get_rec(self, rec_id):
        """
        Get a GraphWalker reconstruction from the AAM backend.

        :param rec_id: int
        :return:
        """
        post_data = {
            'rec_id' : rec_id,
            'csrfmiddlewaretoken': self.session.cookies['csrftoken']}

        # posts the data in form format
        r = self.session.post(self.server + 'graphwalker/get_task', data = post_data)

        if r.status_code == 200:
            gw = GraphWalkerReconstruction()
            gw.from_json(r.text)

        return gw


    def get_mesh(self, sv_id):
        """
        Requests a normalized mesh from the AAM backend for a sv_id.

        :param sv_id:
        :return:
        """

        post_txt = '<mesh><sv_id>%s</sv_id></mesh>' % sv_id
        r = self.session.post(self.server + 'graphwalker/get_mesh', post_txt)

        if r.status_code != 200:
            print(r)
            print(r.content)
            return

        fname_ex = 'filename=([^;]+);'
        m = re.search(fname_ex, r.headers['content-disposition'])
        m.groups()[0], r.content

        mesh = None

        return mesh

    def get_skeleton(self, sv_id):
        """
        Fetch a networkx skeleton from the AAM.

        :param sv_id: int
        :return: networkx graph
        """

        post_data = {
            'sv_id' : sv_id,
            'csrfmiddlewaretoken': self.session.cookies['csrftoken']}

        try:
            r = self.session.post(self.server + 'graphwalker/get_skeleton', data = post_data)

            if r.status_code == 200:
                decoded_json = json.loads(r.text)
                nxg = json_graph.node_link_graph(decoded_json)
        except:
            print('Could not get skeleton - contact your supervisor.')
            nxg = None

        return nxg


    def get_sv_children(self, sv_id):
        """
        Fetch an edge list with additional properties of sv children from the AAM.

        :param sv_id: int
        :return: iterable of edges
        """

        post_data = {
            'sv_id' : sv_id,
            'csrfmiddlewaretoken': self.session.cookies['csrftoken']}

        try:
            r = self.session.post(self.server + 'graphwalker/get_SV_children', data = post_data)
            decoded = json.loads(r.text)
        except:
            # SV children could not be retrieved or are corrupted.
            print('Could not get SV children - contact your supervisor.')
            decoded = None

        return decoded

##################### tmp duplicate definitation of graphwalker class


def store_undo(fct):
    def decorator(self):
        gw_rec = self.current_rec_to_gw()
        curr_state = UndoState(self.current_edge, self.decision_heap, gw_rec)
        self.undo_stack.append(curr_state)
        self.log('store_undo: len(self.undo_stack): {0}'.format(len(self.undo_stack)))

        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0) # limits the undo stack

        self.redo_stack = []  # empty redo stack after normal action

        fct(self)

    return decorator



class GraphWalkerReconstruction(object):
    """
    Class that represents a GraphWalker reconstruction. This is a lightweight representation of
    the part of the dataset supervoxel graph and the only thing necessary together with the
    mesh or skeleton representations that can be retrieved from a supervoxel id to create
    a visual representation of a reconstruction.
    """
    def __init__(self, json_str = ''):

        if json_str:
            self.from_json(json_str)
        else:
            self.rec_graph = nx.Graph()
            self.split_graph = nx.Graph()
            self.sv_rec_start_id = 0
            self.split_svs = set()
            self.bad_svs = set()
            self.ignore_svs = set()
            self.meta_info = []
            self.type = '' # type e.g. Glia, Neuron, ...
            self.rec_id = 0
            self.username = ''
            self.bad_sv_split_locations = dict()

        return

    def from_json(self, json_str):
        """
        Create GraphWalker instance from a json serialized str and return instance.

        :param json_str:
        :return: GraphWalker
        """
        decoded_dict = json.loads(json_str)

        self.rec_graph = json_graph.node_link_graph(decoded_dict['rec_nxg']) # graph from a json serializable object
        self.split_graph = json_graph.node_link_graph(decoded_dict['split_nxg']) # graph from a json serializable object

        self.sv_rec_start_id = decoded_dict['sv_rec_start_id']
        self.split_svs = set(decoded_dict['split_sv'])
        self.bad_svs = set(decoded_dict['bad_svs'])
        self.ignore_svs = set(decoded_dict['ignore_svs'])
        self.meta_info = decoded_dict['meta_info']
        self.rec_id = decoded_dict['rec_id']
        self.username = decoded_dict['username']
        self.type = decoded_dict['type']
        self.bad_sv_split_locations = decoded_dict['bad_sv_split_locations']


        return self

    def to_json(self):
        """
        Serialize reconstruction to json for transport. Packs everything into a dictionary that is then dumped to
        json.

        :return: str; json
        """

        return json.dumps(self.to_dict())

    def to_dict(self):
        """
        Serialize to dict.

        :return:
        """

        encoded_dict = dict()
        encoded_dict['rec_nxg'] = json_graph.node_link_data(self.rec_graph)# provides a json serializable object
        encoded_dict['split_nxg'] = json_graph.node_link_data(self.split_graph)# provides a json serializable object
        encoded_dict['sv_rec_start_id'] = self.sv_rec_start_id # sv id where the user started
        encoded_dict['split_sv'] = list(self.split_svs) # list of all sv's where the user decided for split
        encoded_dict['bad_svs'] = list(self.bad_svs) # set of all sv's that the user tagged as being bad; no sets in json
        encoded_dict['meta_info'] = self.meta_info
        encoded_dict['rec_id'] = self.rec_id # int that represents reconstruction in neo4j db
        encoded_dict['username'] = self.username
        encoded_dict['type'] = self.type
        encoded_dict['ignore_svs'] = list(self.ignore_svs) # currently _ignored_ ...
        encoded_dict['bad_sv_split_locations'] = self.bad_sv_split_locations

        return encoded_dict


########################################################################################################################
# plugin class and related functions follow
########################################################################################################################


def set_position(jump_pos):
    pos = KnossosModule.knossos.getPosition()
    KnossosModule.knossos.setPosition(jump_pos)

def unselect_all_objects():
    selected_obj = KnossosModule.segmentation.getSelectedObjectIndices()
    for cur_obj in selected_obj:
        KnossosModule.segmentation.unselectObject(cur_obj)


class main_class(QtGui.QDialog):
    def __init__(self, parent=KnossosModule.knossos_global_mainwindow):
        #Qt.QApplication.processEvents()
        super(main_class, self).__init__(parent, Qt.Qt.WA_DeleteOnClose)
        try:
            exec(KnossosModule.scripting.getInstanceInContainerStr(__name__) + " = self")
        except KeyError:
            # Allow running from __main__ context
            pass
        self.start_logging()
        self.decision_heap = []
        heapq.heapify(self.decision_heap)

        self.mesh_download_queue = Queue()
        self.init_mesh_download_queue_worker()
        self.mesh_download_done = Queue()

        self.split_svs = set()
        self.bad_svs = set()
        self.bad_sv_split_locations = dict() # for each bad sv a list of coordinates in separate neurons / ecs

        self.manually_added_svs = []

        self.bm_connector = BrainmapsInteraction()

        # holds the currently looked at edge in the same format as the edge list of the decision heap
        self.current_edge = None

        # stores the currently active segmentation object ID (not subobject ID!)
        self.active_obj_id = None

        # create active reconstruction tree and query tree
        #self.create_knossos_rec_trees()

        # nx graph that stores the svs and the merge decisions
        self.rec_graph = nx.Graph()

        # nx graph that stores the split decisions
        self.split_graph = nx.Graph()

        self.current_mode = 'proofreading'

        #self.auto_ND = True
        self.undo_stack = []
        self.redo_stack = []

        # provides a cache of meshes
        self.skel_cache = dict()
        self.mesh_cache = dict()



        # a root is required for efficient graph splitting
        self.sv_rec_start_id = None

        # init Neurodock connector
        self.NDock_connector = NeuroDockInteraction()

        KnossosModule.segmentation.setRenderOnlySelectedObjs(True)

        self.build_gui()

        # terrible hack here. multiprocessing instead of threads might be one option,
        # but even better would be proper multithreading support from PythonQT, which
        # would release the GIL as soon as the execution returns to the C++ code.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.release_gil_hack)
        self.timer.start(5)

        self.timer2 = QtCore.QTimer()
        self.timer2.timeout.connect(self.rebuild_callback_check)
        self.timer2.start(100)

        # done 1004060033426, 562707771591, 729605685610, 943082298690, 563471356725, 564262213035, 1157349768165, 831898835234, 936782720421, 811091137812, 1004060033426

        return


    ################################### methods interacting with the KnossosModule (i.e. KNOSSOS) follow

    def release_gil_hack(self):
        time.sleep(0.01)
        return

    @store_undo
    def manual_add_sv_with_edge(self):
        """
        Retrieve from KNOSSOS the selected sv; This is used for manual rec graph node adding in case
        the automatic segmentation lacks an edge between the reconstruction and this sv.

        :return:
        """

        selected_ids = KnossosModule.segmentation.selectedObjects()
        if len(selected_ids) == 1:
            # Exactly one object needs to be selected for this to work
            # Get all subobject - must be exactly one
            selected_sv_ids = KnossosModule.segmentation.subobjectIdsOfObject(selected_ids[0])
            if len(selected_sv_ids) == 1:
                sv_to_add = selected_sv_ids[0]
                # check whether sv is in ignore_sv
                if not sv_to_add in self.ignore_svs:
                    # populate decision heap with SV id
                    self.add_to_dec_heap(self.make_heap_list_from_children(sv_to_add))

                    #nxskel = self.get_skel_by_id_from_db(sv_to_add)
                    #self.add_nxskel_to_active_reconstruction_tree(nxskel, sv_to_add)

                    self.rec_graph.add_node(sv_to_add)
                    self.rec_graph.node[sv_to_add]['auto'] = False

                    # manually added svs are currently not connected to the rec graph - this might cause
                    # trouble somewhere downstream... keep in mind
                    #self.rec_graph.add_edge(self.current_edge[1][0], self.current_edge[1][1])

                    self.ignore_svs.add(sv_to_add)
                    self.manually_added_svs.append(sv_to_add)

                    self.active_obj_id_label.setText('Active obj ID: %d' % (sv_to_add), )
                    self.current_edge_label.setText('Query SV ID: %d' % (0), )

                    seg_objects = KnossosModule.segmentation.objects()
                    for obj in seg_objects:
                        KnossosModule.segmentation.removeObject(obj)

                    if len(self.decision_heap) > 0:
                        # This means that the heap got repopulated
                        self.mode_combo.setCurrentIndex(0) # switch to proofreading again
                        #self.info_label.setText('Proofreading can.')

                        self.next_SV_edge()
                    else:
                        # The heap is still exhausted, stay in review mode
                        #self.rebuild_knossos_skel_from_rec_graph()
                        self.rebuild_merge_list_from_rec_graph()
                        self.rebuild_mesh_from_rec_graph()

                    self.log('manual_add_sv_with_edge: sv_to_add: {0}'.format(sv_to_add))

        return

    def highlight_query_svs(self):
        """
        Highlight the query supervoxels in the dataviewports.
        This is independent of showing the query mesh or skeleton.

        :return:
        """
        KnossosModule.segmentation.setRenderOnlySelectedObjs(True)

        # delete current query object
        try:
            KnossosModule.segmentation.removeObject(1)
            KnossosModule.segmentation.removeObject(2)
        except:
            pass

        if self.current_edge:
            KnossosModule.segmentation.createObject(1, self.current_edge[1][1], self.current_edge[1][2])
            KnossosModule.segmentation.selectObject(1)
            # query object should be red
            KnossosModule.segmentation.changeColor(1, QtGui.QColor(255, 0, 0, 255))

            KnossosModule.segmentation.createObject(2, self.current_edge[1][0], self.current_edge[1][2])

            # one could cache this, not necessary to rebuild at every step, but no noticable performance effect so far
            for sv_id in self.rec_graph.nodes():
                try:
                    KnossosModule.segmentation.addSubobject(2, sv_id)
                except:
                    pass

            KnossosModule.segmentation.selectObject(2)
            # other object should be black
            KnossosModule.segmentation.changeColor(2, QtGui.QColor(0, 0, 0, 255))

        KnossosModule.segmentation.setRenderOnlySelectedObjs(True)

        return

    def show_query_skel(self):
        """
        Adds the current query skeleton to KNOSSOS and jumps to the query position.

        :return:
        """
        if self.current_edge:
            # add current query skel
            #self.log('Show query skel: Heap len: %d\n' % (len(self.decision_heap)))
            self.current_edge_label.setText('Query SV ID: %d' % (self.current_edge[1][1]), )
            nxskel = self.get_skel_by_id_from_db(self.current_edge[1][1])

            self.nxskel_to_query_tree(nxskel, self.current_edge[1][1])
            set_position(self.current_edge[1][2])
        return

    def show_query(self):
        """
        Adds the current query skeleton to KNOSSOS and jumps to the query position.

        :return:
        """
        if self.current_edge:
            # add current query skel
            #self.log('Show query skel: Heap len: %d\n' % (len(self.decision_heap)))
            self.current_edge_label.setText('Query SV ID: %d' % (self.current_edge[1][1]), )
            self.active_obj_id_label.setText('Active obj ID: %d' % (self.current_edge[1][0]), )

            #nxskel = self.get_skel_by_id_from_db(self.current_edge[1][1])

            #self.nxskel_to_query_tree(nxskel, self.current_edge[1][1])
            set_position(self.current_edge[1][2])
        return

    def create_knossos_rec_trees(self):
        """
        Delete and re-create the KNOSSOS skeleton tree used for the workflow.

        :return:
        """
        KnossosModule.skeleton.delete_tree(1)
        KnossosModule.skeleton.delete_tree(2)
        KnossosModule.skeleton.add_tree(1)
        KnossosModule.skeleton.set_tree_color(1, QtGui.QColor(0,0,0,255))
        KnossosModule.skeleton.add_tree(2)
        KnossosModule.skeleton.set_tree_color(2, QtGui.QColor(255,0,0,255))
        return

    def mesh_download_queue_worker(self):
        while True:
            # this is blocking and therefore fine
            sv_id = self.mesh_download_queue.get()

            if not self.mesh_cache.has_key(sv_id):
                #print('Mesh queue worker: Fetching mesh {0}\n'.format(sv_id))
                ## get mesh
                mesh = self.bm_connector.get_mesh(sv_id)
                #print('Mesh queue worker: Done mesh {0}\n'.format(sv_id))
                # put in cache
                self.mesh_cache[sv_id] = mesh
                self.mesh_download_done.put(sv_id)

            self.mesh_download_queue.task_done()

        return

    def cleanup_mesh_cache(self):
        """
        Checks whether all elements in the mesh cache are part of the
        current graphwalker reconstruction - if not, remove them.
        """
        # currently not implemented

        return

    def init_mesh_download_queue_worker(self):
        """
        Initialize mesh queue daemon workers.

        :return:
        """
        # 5 daemon workers is enough; the download itself is further parallelized on fragment level
        for i in range(5):
            worker = Thread(target=self.mesh_download_queue_worker)
            worker.setDaemon(True)
            worker.start()

        return

    def rebuild_callback_check(self):
        """

        :return:
        """

        while not self.mesh_download_done.empty():
            sv_to_check = self.mesh_download_done.get()
            self.log('rebuild_callback_check: download of sv {0} done.'.format(sv_to_check))

            if self.rec_graph.has_node(sv_to_check): # render sv as black, already part of the reconstruction
                self.mesh_to_knossos(sv_to_check)
                KnossosModule.skeleton.set_tree_color(sv_to_check, QtGui.QColor(245,245,245,255))
            elif sv_to_check == self.current_edge[1][1]: # render sv red, currenty query sv
                self.mesh_to_knossos(sv_to_check)
                KnossosModule.skeleton.set_tree_color(sv_to_check, QtGui.QColor(255,0,0,255))

        return

    def update_mesh_colors(self):
        # iterate over all existing meshes and update the colors

        for sv_id in self.rec_graph.nodes():
            # check the current color
            tree = KnossosModule.skeleton.find_tree_by_id(sv_id)
            if tree:
                if tree.color().red() != 245:
                    #try:
                    KnossosModule.skeleton.set_tree_color(sv_id, QtGui.QColor(245,245,245,255))
                    #except:
                    #    pass

        if self.current_mode == 'proofreading':
            if self.current_edge:
                tree = KnossosModule.skeleton.find_tree_by_id(self.current_edge[1][1])
                if tree:
                    if tree.color().red() != 255:
                        KnossosModule.skeleton.set_tree_color(self.current_edge[1][1], QtGui.QColor(255, 0, 0, 255))

        return

    def prefetch_meshes(self):
        # get sv children of current edge
        # see whether any of the fetched children would land at the top of the heap
        # if yes, add this guy to the mesh_download_queue
        # if not, add the currently second element to the mesh_download_queue
        # no prefetching done currently

        sv_to_prefetch = set()

        prefetch_heap = copy.deepcopy(self.decision_heap)

        next_edge_in_case_of_split = heapq.heappop(prefetch_heap)
        #print 'split: ' +str(next_edge_in_case_of_split)

        sv_to_prefetch.add(next_edge_in_case_of_split[1][1]) # in case of a split, this sv will be presented

        heapq.heappush(next_edge_in_case_of_split, prefetch_heap) # reverse termporary change

        # we simulate a merge decision
        self.add_to_dec_heap(self.make_heap_list_from_children(self.current_edge[1][1]), prefetch_heap)
        next_edge_in_case_of_merge = heapq.heappop(prefetch_heap)
        sv_to_prefetch.add(next_edge_in_case_of_merge[1][1])

        #print('Adding to prefetch queue: {0}\n'.format(sv_to_prefetch))

        for sv_id in sv_to_prefetch:
            if not self.mesh_cache.has_key(sv_id):
                #print('Adding {0} sv to queue\n'.format(sv_id))
                self.mesh_download_queue.put(sv_id)
                self.log('prefetch_meshes: sv {0} put on download queue.'.format(sv_id))

        return

    def rebuild_mesh_from_rec_graph(self):
        #start = time.time()

        # get list of all sv_ids that are currently shown
        trees = KnossosModule.skeleton.trees()
        ids_in_k = set([tree.tree_id() for tree in trees])

        # compare with list in rec_graph
        ids_in_rec_graph = set(self.rec_graph.nodes())

        # add missing ones to knossos, delete if not needed anymore
        ids_to_add = ids_in_rec_graph - ids_in_k
        ids_to_del = ids_in_k - ids_in_rec_graph

        [KnossosModule.skeleton.delete_tree(sv_id) for sv_id in ids_to_del]
        [self.mesh_to_knossos(sv_id) for sv_id in ids_to_add]

        if self.current_mode == 'proofreading':
            if self.current_edge:
                tree = KnossosModule.skeleton.find_tree_by_id(self.current_edge[1][1])
                if not tree:
                    self.mesh_to_knossos(self.current_edge[1][1])

        elif self.current_mode == 'review' or self.current_mode == 'task':
            if self.current_edge:
                tree = KnossosModule.skeleton.find_tree_by_id(self.current_edge[1][1])
                if tree:
                    KnossosModule.skeleton.delete_tree(self.current_edge[1][1])

        # adjust colors of all if necessary
        self.update_mesh_colors()

        #print('Rebuild mesh took: {0}'.format(time.time()-start))

        return

    def mesh_to_knossos(self, sv_id):

        start = time.time()
        if self.mesh_cache.has_key(sv_id):
            #print('Showing mesh {0}'.format(sv_id))
            indices, vertices = self.mesh_cache[sv_id]
            KnossosModule.skeleton.add_tree_pointcloud(sv_id, vertices, [], indices, [], KnossosModule.GL_TRIANGLES,
                                                       True)
        else:
            self.mesh_download_queue.put(sv_id)
            #print('Mesh {0} does not exist in cache, put again on queue.'.format(sv_id))
            self.log('mesh_to_knossos: sv_id: {0} does not yet exist in cache, put again on queue'.format(sv_id))

        self.log('mesh_to_knossos: sv_id: {0} took: {1}'.format(sv_id, time.time()-start))

        return

    def rebuild_merge_list_from_rec_graph(self):
        """
        Recreates the KNOSSOS merge list from scratch from the graphwalker datastructures.

        :return:
        """
        self.log('rebuild_merge_list_from_rec_graph')
        signalsBlocked = KnossosModule.knossos_global_skeletonizer.blockSignals(True)

        try:
            KnossosModule.segmentation.removeObject(1)
            KnossosModule.segmentation.removeObject(2)
        except:
            pass

        if self.current_mode == 'proofreading' and self.current_edge:
            KnossosModule.segmentation.createObject(1, self.current_edge[1][1], self.current_edge[1][2])
            KnossosModule.segmentation.selectObject(1)
            # query object should be red
            KnossosModule.segmentation.changeColor(1, QtGui.QColor(255, 0, 0, 255))

            KnossosModule.segmentation.createObject(2, self.current_edge[1][0], self.current_edge[1][2])
            # one could cache this, not necessary to rebuild at every step
            for sv_id in self.rec_graph.nodes():
                try:
                    KnossosModule.segmentation.addSubobject(2, sv_id)
                except:
                    pass

            KnossosModule.segmentation.selectObject(2)
            # query object should be black
            KnossosModule.segmentation.changeColor(2, QtGui.QColor(0, 0, 0, 255))

        else:
            ids = list(self.rec_graph.nodes())
            if len(ids) > 0:
                first_id = ids[0]
                KnossosModule.segmentation.createObject(1, first_id, (1,1,1))
                for sv_id in self.rec_graph.nodes():
                    try:
                        KnossosModule.segmentation.addSubobject(1, sv_id)
                    except:
                        pass
                KnossosModule.segmentation.selectObject(1)
                # query object should be black
                KnossosModule.segmentation.changeColor(1, QtGui.QColor(0, 0, 0, 255))

        KnossosModule.segmentation.setRenderOnlySelectedObjs(True)

        KnossosModule.knossos_global_skeletonizer.blockSignals(signalsBlocked)
        KnossosModule.knossos_global_skeletonizer.resetData()

        return

    def rebuild_knossos_skel_from_rec_graph(self):
        """
        Visualize the currently active reconstruction graph as skeleton in KNOSSOS from scratch.
        This is required after a mode switch for example, or after a reconstruction is retrieved
        from the backend.

        :return:
        """

        signalsBlocked = KnossosModule.knossos_global_skeletonizer.blockSignals(True)

        self.create_knossos_rec_trees()

        if self.current_mode == 'proofreading':
            for sv_id in self.rec_graph.nodes():
                # fetch skeleton again
                nxskel = self.get_skel_by_id_from_db(sv_id)
                self.nxskel_to_knossos_tree(nxskel, 1, sv_id, signal_block = False)
            self.show_query()

        elif self.current_mode == 'review':
            for sv_id in self.rec_graph.nodes():
                # fetch skeleton again
                nxskel = self.get_skel_by_id_from_db(sv_id)

                if not self.rec_graph.node[sv_id].has_key('auto'):
                    #print('rec graph missing auto: {0}'.format(sv_id))
                    self.log('rebuild_knossos_skel_from_rec_graph: Error: Missing auto for sv_id: {0}. Fixed by setting to auto.'.format(sv_id))
                    self.rec_graph.node[sv_id]['auto'] = True

                if self.rec_graph.node[sv_id]['auto'] == False:
                    self.nxskel_to_knossos_tree(nxskel, 1, sv_id, signal_block = False)
                else:
                    self.nxskel_to_knossos_tree(nxskel, 2, sv_id, signal_block = False)

        elif self.current_mode == 'task':
            for sv_id in self.rec_graph.nodes():
                # fetch skeleton again
                nxskel = self.get_skel_by_id_from_db(sv_id)
                self.nxskel_to_knossos_tree(nxskel, 1, sv_id, signal_block = False)

        KnossosModule.knossos_global_skeletonizer.blockSignals(signalsBlocked)
        KnossosModule.knossos_global_skeletonizer.resetData()

        return

    ################################################## methods for user interaction follow

    def current_rec_to_gw(self):
        gw_instance = GraphWalkerReconstruction()

        gw_instance.rec_graph = copy.deepcopy(self.rec_graph)
        gw_instance.split_graph = copy.deepcopy(self.split_graph)
        gw_instance.username = copy.deepcopy(self.username_line_edit.text)
        gw_instance.rec_id = copy.deepcopy(self.current_rec_id)
        gw_instance.bad_svs = copy.deepcopy(self.bad_svs)
        gw_instance.split_svs = copy.deepcopy(self.split_svs)
        gw_instance.sv_rec_start_id = self.sv_rec_start_id
        gw_instance.type = self.cell_type_combo.currentText
        gw_instance.ignore_svs = copy.deepcopy(self.ignore_svs)
        gw_instance.bad_sv_split_locations = copy.deepcopy(self.bad_sv_split_locations)

        return gw_instance

    def gw_to_current_rec(self, gw_instance):

        self.rec_graph = gw_instance.rec_graph
        self.split_graph = gw_instance.split_graph
        self.bad_svs = gw_instance.bad_svs
        self.current_rec_id = gw_instance.rec_id
        self.split_svs = gw_instance.split_svs
        self.sv_rec_start_id = gw_instance.sv_rec_start_id
        self.ignore_svs = gw_instance.ignore_svs
        self.cell_type_combo.setCurrentIndex(self.cell_type_combo.findText(gw_instance.type))
        self.bad_sv_split_locations = gw_instance.bad_sv_split_locations

        return

    def start_logging(self):
        """
        Initiate logging to a memory buffer. The log file is later submitted
        to the graphwalker backend.

        :return:
        """

        self.log_handle = StringIO.StringIO()
        self.log('start_logging')

        return

    def log(self, to_log):
        """
        Writes string to currently used log buffer.

        :param to_log: str
        :return:
        """
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f: ')
        self.log_handle.write(time_stamp + to_log + '\n')

        return

    def login_backend_connection(self):
        """
        Establishes a connection to the AAM backend with the provided user credentials.

        :return:
        """

        # get username and password from the GUI
        # keep a reference to the backend connector for interaction with it
        self.AAM_connector = AAMInteraction(self.server_line_edit.text)

        response = self.AAM_connector.login(self.username_line_edit.text, self.password_line_edit.text)
        # parse response properly

        self.info_label.setText(response.content)

        self.log('login_backend_connection: ' + response.content)

        return

    def logout_backend_connection(self):
        """
        Log out from the AAM backend and clear username and password field.

        :return:
        """

        response = self.AAM_connector.logout()

        self.username_line_edit.setText('')
        self.password_line_edit.setText('')

        self.info_label.setText(response.content)
        self.log('logout_backend_connection: ' + response.content)

        return

    def start_new_rec_clicked(self):
        """
        Asks the backend for a new task (i.e. supervoxel id).

        :return:
        """
        self.start_logging()
        self.log('start_new_rec_clicked')
        response = self.AAM_connector.start_new_rec()#new_sv_id)

        if response.status_code == 200:

            gw = GraphWalkerReconstruction()
            gw.from_json(response.text)

            # reset all plugin internal datastructures from the last task
            self.reset_plugin_datastructures()
            self.current_rec_id = gw.rec_id

            self.initiate_proofreading(gw.sv_rec_start_id)

            self.mode_combo.setCurrentIndex(self.mode_combo.findText('Proofreading'))
            self.info_label.setText('Started new reconstruction, ID ' + str(self.current_rec_id) + '; sv ID: ' + str(gw.sv_rec_start_id))
        else:
            self.info_label.setText('Failed to start new reconstruction')
            self.log('start_new_rec_clicked: Failed to restart new reconstruction: ' + response.content)

        return

    def submit_rec_to_backend_clicked(self):
        """
        Sends the currently active reconstruction as a final task to the AAM backend.

        :return:
        """

        self.log('submit_rec_to_backend_clicked')

        # Use KNOSSOS time tracking
        annotation_time = KnossosModule.knossos.annotation_time()/3600000.

        # Interact with the AAM connector object for submission. Time is in hours.
        if self.AAM_connector.submit(self.current_rec_to_gw().to_json(),
                                     worktime=annotation_time,
                                     gw_log=self.log_handle.getvalue()):
            self.info_label.setText('Submitted reconstruction ID ' + str(self.current_rec_id) + str(' succesfully.'))
        else:
            self.log('submit_rec_to_backend_clicked: submission failed.')
            self.info_label.setText('Submission failed. Try again.')

        return

    def load_rec_from_backend_clicked(self):
        """
        Asks the AAM backend for a reconstruction with a given reconstruction id or sv_id. This can be used by
        administrators to look at annotations made by users.

        :return:
        """

        self.reset_plugin_datastructures()
        gw_rec = self.AAM_connector.get_rec(int(str(self.load_task_line_edit.text)))

        #self.info_label.setText('Error loading reconstruction ID ' + str(self.load_task_line_edit.text))

        self.gw_to_current_rec(gw_rec)

        self.current_rec_id = int(str(self.load_task_line_edit.text))
        #self.rebuild_knossos_skel_from_rec_graph()
        self.info_label.setText('Loaded reconstruction ID ' + str(self.load_task_line_edit.text) + str(' succesfully.'))

        return

    def restart_rec_clicked(self):
        """

        :return:
        """
        self.start_logging()
        self.log('restart_rec_clicked')
        response = self.AAM_connector.restart_task()

        if response.status_code == 200:
            gw = GraphWalkerReconstruction()
            gw.from_json(response.text)

            # reset all plugin internal datastructures from the last task
            self.reset_plugin_datastructures()

            self.current_rec_id = gw.rec_id

            self.mode_combo.setCurrentIndex(self.mode_combo.findText('Proofreading'))

            self.initiate_proofreading(gw.sv_rec_start_id)
            self.info_label.setText('Restarted reconstruction, ID ' + str(self.current_rec_id) + '; sv ID: ' + str(gw.sv_rec_start_id))

        else:
            self.info_label.setText('Failed to start new reconstruction, check log.')
            self.log('restart_rec_clicked: Failed to restart new reconstruction: ' + response.content)

        return


    def build_gui(self):
        self.setWindowFlags(Qt.Qt.Window)
        layout = QtGui.QGridLayout()
        layout.setSpacing(10)

        # Window layout
        #layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        self.merge_button = QtGui.QPushButton('Merge')
        self.split_button = QtGui.QPushButton('Split')
        self.bad_button = QtGui.QPushButton('Bad SV')
        self.graph_split_button = QtGui.QPushButton('Graph split')
        self.add_selected_sv_button = QtGui.QPushButton('Add selected SV')
        self.mode_combo = QtGui.QComboBox()
        self.stop_button = QtGui.QPushButton('Stop')
        self.undo_button = QtGui.QPushButton('Undo')
        self.redo_button = QtGui.QPushButton('Redo')

        self.login_button = QtGui.QPushButton('Login')
        self.logout_button = QtGui.QPushButton('Logout')

        self.new_task_button = QtGui.QPushButton('New task')
        self.submit_task_button = QtGui.QPushButton('Submit task')
        self.restart_task_button = QtGui.QPushButton('Restart active task')
        #self.skip_task_line_edit = QtGui.QLineEdit('Skip reason')
        self.server_line_edit = QtGui.QLineEdit()
        self.username_line_edit = QtGui.QLineEdit()
        self.password_line_edit = QtGui.QLineEdit() # set echo mode to QLineEdit::PasswordEchoOnEdit

        self.gui_auto_agglo_line_edit = QtGui.QLineEdit()
        self.gui_auto_agglo_line_edit.setText('0')

        self.current_edge_label = QtGui.QLabel('Query SV ID: n/a')
        self.active_obj_id_label = QtGui.QLabel('Active obj ID: n/a')
        self.task_time_label = QtGui.QLabel('Task time: n/a')
        self.info_label = QtGui.QLabel('')
        self.ask_ND_button = QtGui.QPushButton('Ask ND')

        #self.info_label.setTextFormat(Qt.Qt.RichText)
        #self.info_label.setSizePolicy(Qt.QSizePolicy.Fixed, Qt.QSizePolicy.Fixed)
        self.cell_type_combo = QtGui.QComboBox()

        self.cell_type_combo.addItem('Neuron')
        self.cell_type_combo.addItem('Glia')
        self.cell_type_combo.addItem('Other')


        layout.addWidget(self.mode_combo, 1, 0, 1, 2)
        layout.addWidget(QtGui.QLabel("Auto agglomeration threshold"), 2, 0)
        layout.addWidget(self.gui_auto_agglo_line_edit, 2, 1)
        #layout.addWidget(self.ask_ND_button, 2, 1,2)

        layout.addWidget(self.undo_button, 3, 0)
        layout.addWidget(self.redo_button, 3, 1)

        layout.addWidget(self.merge_button, 4, 0)
        layout.addWidget(self.split_button, 5, 0)

        layout.addWidget(self.graph_split_button, 4, 1)
        layout.addWidget(self.add_selected_sv_button, 5, 1)
        layout.addWidget(self.bad_button, 6, 0)
        #layout.addWidget(QtGui.QLabel("Cell type"), 5, 0)

        layout.addWidget(self.cell_type_combo, 6, 1)
        #layout.addWidget(self.stop_button, 5, 0, 1, 2)

        layout.addWidget(self.current_edge_label, 8, 0)
        layout.addWidget(self.active_obj_id_label,  8, 1)

        layout.addWidget(self.new_task_button, 9, 0)
        layout.addWidget(self.submit_task_button, 9, 1)

        #layout.addWidget(QtGui.QLabel("Server: "), 8, 0)
        #layout.addWidget(self.server_line_edit, 8, 1)
        layout.addWidget(QtGui.QLabel("Username: "), 10, 0)
        layout.addWidget(self.username_line_edit, 10, 1)
        layout.addWidget(QtGui.QLabel("Password: "), 11, 0)
        layout.addWidget(self.password_line_edit, 11, 1)
        layout.addWidget(self.login_button, 12, 0)
        layout.addWidget(self.logout_button, 12, 1)

        layout.addWidget(self.restart_task_button, 13, 0)
        #layout.addWidget(self.skip_task_line_edit, 12, 1)
        #layout.addWidget(self.task_time_label, 13, 0, 1, 2)
        layout.addWidget(self.info_label, 14, 0, 1, 2)

        self.mode_combo.addItem('Proofreading')
        self.mode_combo.addItem('Review')
        self.mode_combo.addItem('Task management')

        self.mode_combo.currentIndexChanged.connect(self.mode_combo_changed)
        self.merge_button.clicked.connect(self.merge_SVs_clicked)
        self.split_button.clicked.connect(self.split_SVs_clicked)
        self.bad_button.clicked.connect(self.bad_SV_clicked)
        self.graph_split_button.clicked.connect(self.manual_graph_split)
        self.stop_button.clicked.connect(self.cleanup)
        self.new_task_button.clicked.connect(self.start_new_rec_clicked)
        self.submit_task_button.clicked.connect(self.submit_rec_to_backend_clicked)
        self.restart_task_button.clicked.connect(self.restart_rec_clicked)
        self.login_button.clicked.connect(self.login_backend_connection)
        self.logout_button.clicked.connect(self.logout_backend_connection)
        self.ask_ND_button.clicked.connect(self.ask_ND)
        self.add_selected_sv_button.clicked.connect(self.manual_add_sv_with_edge)
        self.undo_button.clicked.connect(self.undo_clicked)
        self.redo_button.clicked.connect(self.redo_clicked)

        # test server
        #self.server_line_edit.setText('http://192.168.233.125:8000/')

        # aam production server
        self.server_line_edit.setText('http://62.75.143.11:8000/')

        # There is an enum for that, but I don't know how to get it working from Python
        # ... QtGui.QLineEdit.EchoMode
        self.password_line_edit.setEchoMode(3) # 3 means password

        self.mode_combo_changed(2)
        self.mode_combo.setCurrentIndex(self.mode_combo.findText('Task management'))

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('GraphWalker v2')

        # set up global shortcuts for the plugin
        # merge shortcut
        mw = KnossosModule.knossos_global_mainwindow
        merge_action = QtGui.QAction(mw)
        merge_action.setShortcut(QtGui.QKeySequence(Qt.Qt.Key_M))
        merge_action.triggered.connect(self.merge_SVs_clicked)
        mw.addAction(merge_action)

        # split shortcut
        split_action = QtGui.QAction(mw)
        split_action.setShortcut(QtGui.QKeySequence(Qt.Qt.Key_Comma))
        split_action.triggered.connect(self.split_SVs_clicked)
        mw.addAction(split_action)

        # production
        self.username_line_edit.setText('')
        self.password_line_edit.setText('')


        self.show()

        self.build_bad_sv_dialog()
        self.bad_sv_dialog.hide()

        return


    def mode_combo_changed(self, index):


        KnossosModule.segmentation.setRenderOnlySelectedObjs(True)
        # change gui elements to reflect work mode
        if index == 0:
            self.current_mode = 'proofreading'
            self.graph_split_button.setDisabled(True)
            self.split_button.setEnabled(True)
            self.merge_button.setEnabled(True)
            self.gui_auto_agglo_line_edit.setEnabled(True)

            #self.rebuild_knossos_skel_from_rec_graph()
            self.rebuild_merge_list_from_rec_graph()
            self.rebuild_mesh_from_rec_graph()

            self.bad_button.setDisabled(False)
            self.restart_task_button.setDisabled(True)
            self.add_selected_sv_button.setDisabled(True)
            self.new_task_button.setDisabled(True)
            self.login_button.setDisabled(True)
            self.logout_button.setDisabled(True)
            self.server_line_edit.setDisabled(True)
            self.password_line_edit.setDisabled(True)
            self.password_line_edit.setDisabled(True)
            self.username_line_edit.setDisabled(True)
            self.submit_task_button.setDisabled(True)
            self.cell_type_combo.setDisabled(True)

            #self.skip_task_line_edit.setDisabled(True)

        elif index == 1:
            self.current_mode = 'review'
            self.graph_split_button.setEnabled(True)
            self.split_button.setDisabled(True)
            self.merge_button.setDisabled(True)
            self.gui_auto_agglo_line_edit.setDisabled(True)
            self.bad_button.setDisabled(False)

            #self.rebuild_knossos_skel_from_rec_graph()
            self.rebuild_merge_list_from_rec_graph()
            self.rebuild_mesh_from_rec_graph()

            self.restart_task_button.setDisabled(True)
            self.add_selected_sv_button.setDisabled(False)
            self.new_task_button.setDisabled(True)
            self.login_button.setDisabled(True)
            self.logout_button.setDisabled(True)
            self.server_line_edit.setDisabled(True)
            self.password_line_edit.setDisabled(True)
            self.username_line_edit.setDisabled(True)
            self.submit_task_button.setDisabled(True)
            self.cell_type_combo.setDisabled(False)
            #self.skip_task_line_edit.setDisabled(True)

        elif index == 2:
            self.current_mode = 'task'
            self.graph_split_button.setDisabled(True)
            self.split_button.setDisabled(True)
            self.merge_button.setDisabled(True)
            self.gui_auto_agglo_line_edit.setDisabled(True)
            self.add_selected_sv_button.setDisabled(True)
            self.restart_task_button.setDisabled(False)
            self.bad_button.setDisabled(True)

            #self.rebuild_knossos_skel_from_rec_graph()
            self.rebuild_merge_list_from_rec_graph()
            self.rebuild_mesh_from_rec_graph()

            self.new_task_button.setDisabled(False)
            self.login_button.setDisabled(False)
            self.logout_button.setDisabled(False)
            self.server_line_edit.setDisabled(False)
            self.password_line_edit.setDisabled(False)
            self.username_line_edit.setDisabled(False)
            self.submit_task_button.setDisabled(False)
            self.cell_type_combo.setDisabled(True)
            #self.skip_task_line_edit.setDisabled(False)
        self.log('mode_combo_changed: was changed to mode: {0}'.format(self.current_mode))
        return

    @store_undo
    def manual_graph_split(self):
        """
        Performs a graph split of the current reconstruction graph at the
        supervoxel ID that is specified by the comment in the currently active
        skeleton node.

        :return:
        """

        try:
            clicked_id = KnossosModule.skeleton.last_clicked_point_cloud_id()
            #clicked_id = int(KnossosModule.skeleton.active_node().comment())
            self.log('manual_graph_split: triggered on sv id: %d' % (clicked_id))
        except:
            self.info_label.setText('Please click on a supervoxel')
            return

        ids_to_remove = []
        ids_to_remove.append(clicked_id)

        # split rec graph and perform cc
        self.rec_graph.remove_nodes_from(ids_to_remove)

        sgraphs = nx.connected_component_subgraphs(self.rec_graph)
        # find sub graph that contains the root, this is the new rec_graph
        for g in sgraphs:
            if self.sv_rec_start_id in g.nodes():
                self.rec_graph = g
            else:
                # add all id's to ids_to_remove, they need to be removed from decision heap after
                ids_to_remove.extend(g.nodes())

        # These nodes get lost in the graph splitting process
        for manually_added_sv in self.manually_added_svs:
            if manually_added_sv != clicked_id:
                self.rec_graph.add_node(manually_added_sv)

        #self.rebuild_knossos_skel_from_rec_graph()
        self.rebuild_merge_list_from_rec_graph()
        self.rebuild_mesh_from_rec_graph()

        id_remove_set = set(ids_to_remove)

        # All IDs can be added to ignore_svs but NOT to split_svs. Splits are only recorded for
        # actual edges in the graph. We cannot be sure that the graph split decision is really
        # a valid split decision. It might split a valid subgraph (that in itself should not belong
        # to the cell the annotator is currently working on). Ignore is safe since it will only prevent
        # the ids from showing up against for this cell, but for other, independent cells.
        for id_to_remove in ids_to_remove:
            self.ignore_svs.add(id_to_remove)

        self.log('manual_graph_split: IDs to remove: %s' % (str(id_remove_set)))
        # remove now unnecessary id's from decision heap
        self.log('manual_graph_split: Heap len before: %d\n' % (len(self.decision_heap)))
        self.decision_heap = [dec_edge for dec_edge in self.decision_heap if not (dec_edge[1][0] in id_remove_set)
                              or (dec_edge[1][1] in id_remove_set)]
        heapq.heapify(self.decision_heap)
        self.log('manual_graph_split: Heap len after: %d\n' % (len(self.decision_heap)))

        # remove the current edge in proofreading mode, but only if it emerged from the removed component
        if self.current_edge[1][0] in id_remove_set or self.current_edge[1][1] in id_remove_set:
            self.log('manual_graph_split: Removed current edge')
            if len(self.decision_heap) > 0:
                self.current_edge = heapq.heappop(self.decision_heap)
            else:
                self.current_edge = None
                self.mode_combo.setCurrentIndex(1)
                self.info_label.setText('No more SVs on heap.')
                self.log('manual_graph_split: SV heap exhausted.')
        else:
            self.log('manual_graph_split: Did not remove current edge')

        return


    def get_skel_by_id_from_db(self, skel_id):
        """
        Fetch skeletons (networkx graph instances) from client cache or from backend

        :param skel_id:
        :return:
        """
        start = time.time()
        if self.skel_cache.has_key(skel_id):
            return self.skel_cache[skel_id]

        # skeleton is not in client cache, fetch from backend; get_skeleton returns already a nx graph instance
        reskel = self.AAM_connector.get_skeleton(skel_id)

        if not reskel:
            self.log('get_skel_by_id_from_db: Could not get skeleton with id {0} from backend.'.format(skel_id))
        else:
            # add reskel to cache
            self.skel_cache[skel_id] = reskel
            self.log('get_skel_by_id_from_db: fetching skel from cache / db took: ' + str(time.time()-start))

        return reskel

    def add_to_dec_heap(self, new_decisions, dec_heap = None):
        """
        Add decision edges to the decision heap.

        :param new_decisions: edge
        :return:
        """

        if not dec_heap:
            dec_heap = self.decision_heap

        for dec in new_decisions:
            heapq.heappush(dec_heap, dec)
        return

    def reset_plugin_datastructures(self):
        """
        Cleans up all plugin datastructures from last proofreading, to make graphwalker
        ready for next proofreading task.

        :return:
        """
        self.log('reset_plugin_datastructures')
        self.rec_graph = nx.Graph()
        self.decision_heap = []
        heapq.heapify(self.decision_heap)

        self.bad_svs = set()
        self.split_svs = set()
        self.ignore_svs = set()
        self.skel_cache = dict()
        self.current_rec_id = 0
        self.active_obj_id = 0
        self.sv_rec_start_id = 0
        self.current_edge = None
        self.active_obj_id = None
        self.sv_rec_start_id = None
        self.undo_stack = []
        self.redo_stack = []
        self.bad_sv_split_locations = dict()

        # reset knossos annotation time; in ms
        KnossosModule.knossos.set_annotation_time(0)

        # delete all trees
        trees = KnossosModule.skeleton.trees()
        ids_in_k = [tree.tree_id() for tree in trees]
        [KnossosModule.skeleton.delete_tree(tree_id) for tree_id in ids_in_k]
        self.current_mode = 'proofreading' # one of 'proof', 'review', 'final'

        return

    def initiate_proofreading(self, new_id):
        """
        Starts the proofreading workflow, at a given supervoxel ID:
        - Logging started
        - First supervoxel children fetched from backend
        - Next query supervoxel shown to annotator

        :param new_id: int
        :return:
        """
        # 867132839421 - google skel 7 for development
        # hardcoded start for google skel7

        self.active_obj_id = new_id
        self.sv_rec_start_id = new_id

        # the start sv should not be considered for later merge queries
        self.ignore_svs.add(self.active_obj_id)

        # populate decision heap with start SV id
        self.add_to_dec_heap(self.make_heap_list_from_children(new_id))
        self.log('initiate_proofreading: len(self.decision_heap): ' + str(len(self.decision_heap)))

        #nxskel = self.get_skel_by_id_from_db(self.active_obj_id)

        # add the first sv to the mesh downloader

        self.mesh_download_queue.put(self.active_obj_id)
        self.log('initiate_proofreading: sv {0} put on download queue.'.format(self.active_obj_id))

        #print('Added first sv {0} to mesh download queue\n'.format(self.active_obj_id))
        #self.add_nxskel_to_active_reconstruction_tree(nxskel, new_id)
        self.rec_graph.add_node(self.sv_rec_start_id)
        self.rec_graph.node[self.sv_rec_start_id]['auto'] = False

        self.active_obj_id_label.setText('Active obj ID: %d' % (self.active_obj_id), )
        self.current_edge_label.setText('Query SV ID: %d' % (0), )
        self.next_SV_edge()
        return

    @store_undo
    def merge_SVs_clicked(self):
        """
        Manual merge of current query SV. Undo-able action.

        :return:
        """
        self.log('merge_SVs_clicked')
        self.merge_curr_edge()
        self.next_SV_edge()
        return

    @store_undo
    def split_SVs_clicked(self):
        """
        Manual split of current query SV. Undo-able action.

        :return:
        """
        self.log('split_SVs_clicked')
        # store decision in merge graph
        #self.m_graph[self.current_edge[1][0]][self.current_edge[1][1]]['decision'] = False
        self.split_svs.add(self.current_edge[1][1])
        self.ignore_svs.add(self.current_edge[1][1])

        #print('Split graph edge [0][1]: {0} [1][1]: {1}'.format(self.current_edge[1][0], self.current_edge[1][1]))
        self.split_graph.add_edge(self.current_edge[1][0], self.current_edge[1][1])
        #self.split_graph.edge[self.current_edge[1][0], self.current_edge[1][1]]['split'] = True

        self.info_label.setText('Split supervoxel')

        tree = KnossosModule.skeleton.find_tree_by_id(self.current_edge[1][1])
        if tree:
            KnossosModule.skeleton.delete_tree(self.current_edge[1][1])

        self.next_SV_edge()
        return

    def load_state(self, state_to_load):
        """
        Loads an undo / redo state to graphwalker and sets the plugin datastructures.

        :param state_to_load:
        :return:
        """

        self.gw_to_current_rec(state_to_load.gw_object)
        self.current_edge = state_to_load.curr_edge
        self.decision_heap = state_to_load.curr_heap
        self.active_obj_id = self.current_edge[1][0]

        #self.rebuild_knossos_skel_from_rec_graph()
        self.rebuild_merge_list_from_rec_graph()
        self.rebuild_mesh_from_rec_graph()

        self.show_query()

        #if self.current_edge:
        #    set_position(self.current_edge[1][2])
        #    self.current_edge_label

        return

    def undo_clicked(self):
        """
        Trigger undo-actions.

        :return:
        """
        if len(self.undo_stack) > 0:

            # store the current state for redo
            gw_rec = self.current_rec_to_gw()
            curr_state = UndoState(self.current_edge, self.decision_heap, gw_rec)
            self.redo_stack.append(curr_state)

            to_undo = self.undo_stack.pop()
            #self.redo_stack.append(to_undo)
            self.load_state(to_undo)
            #print('Current rec graph node len after:{0}'.format(len(self.rec_graph.nodes())))
            self.log('undo_clicked: Last state restored')
            self.info_label.setText('Undo triggered')
        return

    def redo_clicked(self):
        """
        Trigger redo-actions.

        :return:
        """

        if len(self.redo_stack) > 0:

            # store the current state for undo again
            gw_rec = self.current_rec_to_gw()
            curr_state = UndoState(self.current_edge, self.decision_heap, gw_rec)
            self.undo_stack.append(curr_state)

            to_redo = self.redo_stack.pop()
            #self.undo_stack.append(to_redo)
            self.load_state(to_redo)
            self.log('redo_clicked: Redo state restored')
            self.info_label.setText('Redo triggered')
        return

    @store_undo
    def bad_SV_clicked(self):
        """
        Bad SVs are defined by:
        - SV spans multiple cells
        - SV spans into ECS

        :return:
        """
        # determine mode
        if self.current_mode == 'proofreading':
            self.current_bad_sv = self.current_edge[1][1]

            if self.current_bad_sv in self.bad_svs:
                self.info_label.setText('Query supervoxel is already in bad svs.')
                return

        elif self.current_mode == 'review':
            try:
                self.current_bad_sv = KnossosModule.skeleton.last_clicked_point_cloud_id()
                if self.current_bad_sv in self.bad_svs:
                    self.info_label.setText('Query supervoxel is already in bad svs.')
                    return
            except:
                self.info_label.setText('Please select a supervoxel first.')
                return

        # highlight the selected mesh in red
        bad_sv = KnossosModule.skeleton.find_tree_by_id(self.current_bad_sv)
        if bad_sv:
            KnossosModule.skeleton.set_tree_color(self.current_bad_sv, QtGui.QColor(255, 0, 0, 255))

        # highlight in data viewports

        try:
            KnossosModule.segmentation.removeObject(1)
            KnossosModule.segmentation.removeObject(2)
            KnossosModule.segmentation.removeObject(3)
        except:
            pass

        KnossosModule.segmentation.createObject(3, self.current_bad_sv, (1,1,1))
        KnossosModule.segmentation.selectObject(3)
        KnossosModule.segmentation.changeColor(3, QtGui.QColor(255, 0, 0, 255))

        self.bad_sv_split_locations[str(self.current_bad_sv)] = []
        self.bad_sv_dialog.show()

        return

    def build_bad_sv_dialog(self):

        self.bad_sv_dialog = QtGui.QDialog(KnossosModule.knossos_global_mainwindow, Qt.Qt.WA_DeleteOnClose)

        self.bad_sv_dialog.setWindowFlags(Qt.Qt.Window)
        self.bad_sv_dialog.setWindowFlags(self.bad_sv_dialog.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.bad_sv_dialog.setWindowFlags(self.bad_sv_dialog.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)

        #self.bad_sv_dialog.setModal(True)
        layout = QtGui.QGridLayout()
        layout.setSpacing(10)
        self.bad_sv_dialog.setLayout(layout)

        self.bad_sv_dialog_done_button = QtGui.QPushButton('Done')
        self.bad_sv_dialog_cancel_button = QtGui.QPushButton('Cancel')
        self.bad_sv_dialog_add_position_button = QtGui.QPushButton('Add position')
        self.bad_sv_info_label = QtGui.QLabel('No locations added so far.')

        self.bad_sv_dialog.setGeometry(300, 300, 200, 200)
        self.bad_sv_dialog.setWindowTitle('Select split locations')

        layout.addWidget(self.bad_sv_dialog_add_position_button, 1, 0, 1, 2)
        layout.addWidget(self.bad_sv_dialog_cancel_button, 2, 0)
        layout.addWidget(self.bad_sv_dialog_done_button, 2, 1)
        layout.addWidget(self.bad_sv_info_label, 3, 0, 1, 2)

        self.bad_sv_dialog_add_position_button.clicked.connect(self.bad_sv_dialog_add_position_button_clicked)
        self.bad_sv_dialog_cancel_button.clicked.connect(self.bad_sv_dialog_cancel_button_clicked)
        self.bad_sv_dialog_done_button.clicked.connect(self.bad_sv_dialog_done_button_clicked)

        self.bad_sv_dialog.show()
        return

    def bad_sv_dialog_add_position_button_clicked(self):
        cur_pos = KnossosModule.knossos.getPosition()
        curr_pos_sv_id = KnossosModule.knossos.readOverlayVoxel(cur_pos)

        if curr_pos_sv_id != self.current_bad_sv:
            self.bad_sv_info_label.setText('Only positions inside the red area can be added!')
            return

        # for each bad sv a list of coordinates in separate neurons / ecs
        self.bad_sv_split_locations[str(self.current_bad_sv)].append(cur_pos)

        self.log('bad_sv_dialog_add_position_button_clicked: Added location: {0}'.format(cur_pos))

        info_str = '\n'.join(map(str, self.bad_sv_split_locations[str(self.current_bad_sv)]))
        self.bad_sv_info_label.setText(info_str)

        return

    def bad_sv_dialog_cancel_button_clicked(self):

        self.bad_sv_split_locations[str(self.current_bad_sv)] = []

        self.bad_sv_info_label.setText('No locations added so far.')
        self.current_bad_sv = None
        self.bad_sv_dialog.hide()
        self.update_mesh_colors()

        try:
            KnossosModule.segmentation.removeObject(3)
        except:
            pass

        self.rebuild_merge_list_from_rec_graph()

        return

    def bad_sv_dialog_done_button_clicked(self):

        # only if at least two locations were added!
        if len(self.bad_sv_split_locations[str(self.current_bad_sv)]) > 1:
            self.log('bad_sv_dialog_done_button_clicked: added {0} to bad sv list.'.format(self.current_bad_sv))
            self.info_label.setText('Added {0} to bad SVs'.format(self.current_bad_sv))
            self.bad_sv_info_label.setText('No locations added so far.')
            self.bad_svs.add(self.current_bad_sv)
            self.bad_sv_dialog.hide()
            self.update_mesh_colors()
            try:
                KnossosModule.segmentation.removeObject(3)
            except:
                pass

            self.rebuild_merge_list_from_rec_graph()


        else:
            self.bad_sv_info_label.setText('Add at least 2 locations first.')

        return

    def cleanup(self):
        #if self.logging:
        #    self.log_handle.close()#
        self.log('cleanup')
        self.info_label.setText('Stopped')
        return

    def log_heap(self):
        for e in self.decision_heap:
            self.log(str(e) + '\n')
        self.log('\n\n')

    def ask_ND(self):
        """
        Queries the neurodock backend for a merge decision.

        :return:
        """

        start = time.time()
        prob = self.NDock_connector.get_decision(self.current_edge[1][0], self.current_edge[1][1], self.current_edge[1][2])
        #print('ND probability for merger: {0}'.format(prob))
        self.log('ask_ND: neurodock returned merge probability {0} in {1}s\n'.format(prob, time.time()-start))
        print('ND prob: {0}, took {1}'.format(prob, time.time()-start))

        if prob > 0.8:
            print('ND is sure, merge!')
            #self.merge_curr_edge()
            return True
        elif prob < 0.2:
            print('ND is sure, split')
            return False
        else:
            print('ND is unsure, decide yourself!')
            return False

    def merge_curr_edge(self):
        """
        Merges currently displayed query sv to reconstruction graph.

        :return:
        """

        self.log('merge_curr_edge: merging query SV {0}'.format(self.current_edge[1][1]))
        self.ignore_svs.add(self.current_edge[1][1])

        self.add_to_dec_heap(self.make_heap_list_from_children(self.current_edge[1][1]))
        #nxskel = self.get_skel_by_id_from_db(self.current_edge[1][1])
        #self.add_nxskel_to_active_reconstruction_tree(nxskel, self.current_edge[1][1])
        #self.rebuild_mesh_from_rec_graph()

        # this graph holds only the current reconstructions

        if not self.rec_graph.node.has_key(self.current_edge[1][0]):
            # This is a so far not understood bug, but its impact seems small
            self.log('merge_curr_edge: Error missing sv_id in rec_graph: {0}'.format(self.current_edge[1][0]))
            self.rec_graph.node[self.current_edge[1][0]]['auto'] = True

        self.rec_graph.add_edge(self.current_edge[1][0], self.current_edge[1][1])
        self.rec_graph.edge[self.current_edge[1][0]][self.current_edge[1][1]]['manual merge'] = True

        self.rec_graph.node[self.current_edge[1][1]]['auto'] = False
        self.info_label.setText('Merged supervoxel')

        self.update_mesh_colors()

        return

    def auto_agglomerate(self, node_limit = 0):
        """
        Optional auto-agglomeration feature that agglomerates all merge decisions
        up to a given node_limit

        :param node_limit: int
        :return:
        """

        self.log('auto_agglomerate: Started, node_limit: {0} Heap len: {1}'.format(node_limit,
                                                                                    len(self.decision_heap)))
        start = time.time()
        # automatically agglomerate to a certain size
        #self.current_edge = heapq.heappop(self.decision_heap)
        self.current_edge = heapq.heappop(self.decision_heap)
        agglomerated_nodes = 0
        agglomerated_skels = []

        self.prefetch_meshes()

        nxskel = self.get_skel_by_id_from_db(self.current_edge[1][1])

        # do not auto agglomerate too large pieces - this makes it confusing to the annotator
        while agglomerated_nodes + len(nxskel.nodes()) < node_limit and len(self.decision_heap) > 0:# and self.ask_ND():

            agglomerated_skels.append((nxskel, self.current_edge[1][1]))
            self.ignore_svs.add(self.current_edge[1][1])
            self.add_to_dec_heap(self.make_heap_list_from_children(self.current_edge[1][1]))

            if not self.rec_graph.node.has_key(self.current_edge[1][0]):
                # This is a so far not understood bug, but its impact seems small
                self.log('auto_agglomerate: Missing sv_id in rec_graph: {0}'.format(self.current_edge[1][0]))
                self.rec_graph.node[self.current_edge[1][0]]['auto'] = True

            self.rec_graph.add_edge(self.current_edge[1][0], self.current_edge[1][1])
            self.rec_graph.edge[self.current_edge[1][0]][self.current_edge[1][1]]['manual merge'] = False

            self.rec_graph.node[self.current_edge[1][1]]['auto'] = True
            agglomerated_nodes += len(nxskel.nodes())

            self.current_edge = heapq.heappop(self.decision_heap)
            nxskel = self.get_skel_by_id_from_db(self.current_edge[1][1])
            self.mesh_download_queue.put(self.current_edge[1][1])


        # take care of the signal block here, many signal block operations are expensive
        #signalsBlocked = KnossosModule.knossos_global_skeletonizer.blockSignals(True)

        #for agglo_skel in agglomerated_skels:
        #    self.add_nxskel_to_active_reconstruction_tree(agglo_skel[0], agglo_skel[1], signal_block = False)

        #KnossosModule.knossos_global_skeletonizer.blockSignals(signalsBlocked)
        #KnossosModule.knossos_global_skeletonizer.resetData()
        self.log('auto_agglomerate: Done, added ' + str(len(agglomerated_skels)) + ' SVs + ' + ' took in total ' + str(time.time()-start))
        return

    def next_SV_edge(self):
        """
        Present the next query supervoxel to the annotator. Auto-agglomeration
        is triggered here as well.

        :return:
        """
        # auto-agglomerate as far as "possible"
        if len(self.decision_heap) == 0:
            #print('heap exhausted')
            # This means that the no more edges are are in the SV graph. Switch to review mode.
            self.mode_combo.setCurrentIndex(1)
            self.info_label.setText('No more SVs on heap.')
            self.log('next_SV_edge: SV heap exhausted.')
            self.current_edge = None
            return

        self.auto_agglomerate(int(str(self.gui_auto_agglo_line_edit.text)))

        self.active_obj_id = self.current_edge[1][0]
        self.active_obj_id_label.setText('Active obj ID: %d' % (self.active_obj_id), )

        self.show_query()
        self.highlight_query_svs()
        #print('Added sv {0} to mesh download queue: '.format(self.current_edge[1][1]))
        self.mesh_download_queue.put(self.current_edge[1][1])
        self.log('next_SV_edge: sv {0} put on download queue.'.format(self.current_edge[1][1]))
        self.update_mesh_colors()

        return

    def nxskel_to_knossos_tree(self, nxskel, knossos_tree_id, sv_id, signal_block = True):
        # disable knossos signal emission first - O(n^2) otherwise

        if not nxskel:
            return

        if signal_block:
            signalsBlocked = KnossosModule.knossos_global_skeletonizer.blockSignals(True)
        k_tree = KnossosModule.skeleton.find_tree_by_id(knossos_tree_id)
        # add nodes
        nx_knossos_id_map = dict()
        for nx_node in nxskel.nodes():
            nx_coord = nxskel.node[nx_node]['position']
            #newsk_node.from_scratch(newsk_anno, nx_coord[1]+1, nx_coord[0]+1, nx_coord[2]+1, ID=nx_node)
            k_node = KnossosModule.skeleton.add_node([nx_coord[1]+1, nx_coord[0]+1, nx_coord[2]+1], k_tree)
            nx_knossos_id_map[nx_node] = k_node.node_id()
            # possible to add this to node property instead of abusing the node comments?
            KnossosModule.skeleton.set_comment(nx_knossos_id_map[nx_node], str(sv_id))

        # add edges
        for nx_src, nx_tgt in nxskel.edges():
            KnossosModule.skeleton.add_segment(nx_knossos_id_map[nx_src], nx_knossos_id_map[nx_tgt])

        # enable signals again
        if signal_block:
            KnossosModule.knossos_global_skeletonizer.blockSignals(signalsBlocked)
            KnossosModule.knossos_global_skeletonizer.resetData()
        return

    def add_nxskel_to_active_reconstruction_tree(self, nxskel, sv_id, signal_block = True):
        # active tree is always tree with id 1 by convention
        start = time.time()
        self.nxskel_to_knossos_tree(nxskel, 1, sv_id, signal_block = signal_block)
        if nxskel:
            self.log('add_nxskel_to_active_reconstruction_tree: took: ' + str(time.time()-start) + ' num nodes: ' + str(len(nxskel.nodes())) + '\n')
        return

    def nxskel_to_query_tree(self, nxskel, sv_id):
        start = time.time()
        self.flush_query_tree()
        self.nxskel_to_knossos_tree(nxskel, 2, sv_id)
        if nxskel:
            self.log('nxskel_to_query_tree: took: ' + str(time.time()-start) + ' num nodes: ' + str(len(nxskel.nodes())) + '\n')
        return

    def flush_query_tree(self):
        """
        Clear the KNOSSOS query tree, by convention tree 2

        :return:
        """

        KnossosModule.skeleton.delete_tree(2)
        KnossosModule.skeleton.add_tree(2)
        KnossosModule.skeleton.set_tree_color(2, QtGui.QColor(255,0,0,255))
        return

    def make_heap_list_from_children(self, sv_id):
        """
        Fetch a list of sv children from the backend and convert to the proper format for the plugin.
        Example edge (heap) list [(score, (sv_id, trg_id, coord), ...]
        score: merge score
        sv_id: parent sv
        trg_id: child sv
        coord: (x,y,z) iterable of int

        :param sv_id: int
        :return: iterable of edges
        """
        # entries of heap list are of form: (score, (id1, id2, coord))
        heap_list = []
        start = time.time()
        # fetch children from backend
        heap_list = self.AAM_connector.get_sv_children(sv_id)
        # this prevents that sv's that are already in the reconstruction or sv's that were split are considered again
        heap_list = [el for el in heap_list if not (el[1][1] in self.ignore_svs)]

        self.log('make_heap_list_from_children: query took: ' + str(time.time()-start))
        return heap_list

if __name__=='__main__':
    A = main_class()