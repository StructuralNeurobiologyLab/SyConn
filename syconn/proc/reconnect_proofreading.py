# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.handler.basics import flatten
from multiprocessing import JoinableQueue as Queue
from multiprocessing import Process
from knossos_utils import skeleton
from knossos_utils import skeleton_utils as su
from knossos_utils import knossosdataset as kds
kds._set_noprint(True)
import zipfile
import os
import time
import copy
import networkx as nx
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import glob
from networkx.readwrite import json_graph
#import brainmaps as bm
import colorsys
import random
import json
import re
from collections import Counter
from collections import defaultdict
import sys
import traceback

node_to_sv_mappings_todo_queue = Queue()
node_to_sv_mappings_done_queue = Queue()


def load_best_bet_rag(path = '/mnt/j0126/areaxfs_v10/RAGs/v4b_20180214_nocb_merges.txt'):
    """
    Loads the current default "best bet" supervoxel graph / rag of the data set.
    :param path: str
    :return: networkx graph
    """
    appr_g = nx.read_edgelist(path, delimiter=',', nodetype=int)

    print('Done loading best bet RAG from {0}'.format(path))

    return appr_g

def analyze_j0126_reconnector_task(path_to_kzip, make_plots = False, verbose=False):
    task_dict = read_j0126_reconnector_task(path_to_kzip)
    done_tasks = [task for task in task_dict['reconn_done_list']]
    #times = [t['t'] for t in done_tasks]
    times = []
    k_annos = []
    src_coords = []
    skel_times = []
    all_nxgs = []
    src_ids = []
    for done_task in done_tasks:
        #try:
        k_annos.append(nx_skel_to_knossos_skel(json_graph.node_link_graph(json.loads(done_task['skel_json']))))
        skel_times.append(get_skel_times_from_nx_skel(json_graph.node_link_graph(json.loads(done_task['skel_json']))))
        all_nxgs.append(json_graph.node_link_graph(json.loads(done_task['skel_json'])))
        times.append(done_task['t'])
        src_coords.append((done_task['src_crd_1'], done_task['src_crd_2']))
        src_ids.append(done_task['src_id'])
        #except Exception as e:
        #    print('Could not parse reconnect with src_id {0}'.format(done_task['src_id']))
    #[nx_skel_to_knossos_skel(json_graph.node_link_graph(json.loads(done_task['skel_json']))) for
     #          done_task in done_tasks]

    stats_dict = dict()
    stats_dict['task'] = os.path.basename(path_to_kzip)

    mobj = re.search(r'^.*-.*-(?P<annotator>.*)-\d{8}-\d{6}-final.*$', stats_dict['task'])
    if mobj:
        try:
            stats_dict['annotator'] = mobj.group('annotator')
        except:
            print('Could not extract annotator name')
            stats_dict['annotator'] = 'unknown'

    stats_dict['src_coords'] = src_coords

    stats_dict['src_ids'] = src_ids
    stats_dict['times'] = times
    skel_lengths = [k_anno.physical_length()/1000. for k_anno in k_annos]
    stats_dict['skel_lengths'] = skel_lengths

    stats_dict['annotator'] = [stats_dict['annotator']]*len(skel_lengths)

    #skel_times = [get_skel_times_from_nx_skel(json_graph.node_link_graph(json.loads(done_task['skel_json']))) for
    #           done_task in done_tasks]
    stats_dict['skel_times'] = skel_times

    times_no_skel = [t['t'] for t, ts in zip(done_tasks, skel_times) if ts < 0.1]
    stats_dict['times_no_skel'] = times_no_skel
    # get all skel node timestamps
    #all_nxgs = [json_graph.node_link_graph(json.loads(done_task['skel_json'])) for
    #           done_task in done_tasks]
    all_timestamps = []
    for nxg in all_nxgs:
        these_ts = []
        for n in nxg.nodes():
            try:
                these_ts.append(nxg.node[n]['ts'] / 1000.)
            except KeyError:
                continue
        all_timestamps.append(these_ts)

    t_minus_skel = [t - st for t, st in zip(times, skel_times)]

    stats_dict['t_minus_skel'] = t_minus_skel

    skel_speed = [sl/st for sl, st in zip(skel_lengths, skel_times) if st > 0.]
    stats_dict['skel_speed'] = skel_speed
    stats_dict['mean_task_time'] = np.mean(times)
    stats_dict['k_annos'] = k_annos



    #print('cum skel time {0} for 100 reconnects'.format(sum(skel_times)))
    #print('cum task time {0} for 100 reconnects'.format(sum(times)))
    #print('cum task time {0} for no skeleton traced tasks'.format(sum(times_no_skel)))
    if verbose:
        print('Mean skel speed [um/s] {0} median {1} std {2}'.format(np.mean(skel_speed), np.median(skel_speed), np.std(skel_speed)))
        print('Mean skel times {0} median {1} std {2}'.format(np.mean(skel_times), np.median(skel_times), np.std(skel_times)))
        print('Mean skel length {0} median {1} std {2}'.format(np.mean(skel_lengths), np.median(skel_lengths), np.std(skel_lengths)))
        print('Mean task time {0} median {1} std {2}'.format(np.mean(times), np.median(times), np.std(times)))
        print('Mean task time no skeleton {0} median {1} std {2}'.format(np.mean(times_no_skel), np.median(times_no_skel), np.std(times_no_skel)))
        print('Mean t_minus_skel time {0} median {1} std {2}'.format(np.mean(t_minus_skel), np.median(t_minus_skel), np.std(t_minus_skel)))

    if make_plots:
        plt.figure()
        for this_skel_ts in all_timestamps:
            plt.plot(this_skel_ts, [1.]*len(this_skel_ts))
        plt.title('all node timestamps')
        plt.xlabel('ts in s')

        plt.figure()
        plt.hist(times, bins=100)
        plt.xlabel('t [s]')
        plt.title('hist of total times')

        plt.figure()
        plt.hist(times_no_skel, bins=100)
        plt.xlabel('t [s]')
        plt.title('hist of total times no skeleton')

        plt.figure()
        plt.hist(skel_lengths, bins=100)
        plt.xlabel('skel length [um]')
        plt.title('hist of skeleton lengths')

        plt.figure()
        plt.hist(skel_times, bins=100)
        plt.xlabel('skel time [s]')
        plt.title('hist of skeleton times')

    #print('cum time over 20s {0}, under 20s {1}'.format(sum([t for t in times if t >20.]), sum([t for t in times if t <20.])))
    return stats_dict


def read_j0126_reconnector_task(path_to_kzip):
    import json

    with zipfile.ZipFile(path_to_kzip, 'r') as kzip_fh:
        task_dict = json.loads(kzip_fh.read('reconnect_json'))

    return task_dict

def nx_skel_to_knossos_skel(nxg):
    this_anno = skeleton.SkeletonAnnotation()
    this_anno.scaling = [10., 10., 20.]
    nodes_to_remove = []
    for n in nxg.nodes():
        #print(nxg.node[n])
        try:
            coord = map(int, nxg.node[n]['coord'].replace('(','').replace(')', '').replace('L','').split(','))
        except KeyError:
            #print('nx node has no coord defined, skipping.')
            nodes_to_remove.append(n)
            continue

        node = skeleton.SkeletonNode()
        node.from_scratch(this_anno, *coord, ID=n)
        this_anno.addNode(node)

    [nxg.remove_node(n) for n in nodes_to_remove]
    #if len(nodes_to_remove):
    #    print('Had to remove {0} nodes without coord'.format(len(nodes_to_remove)))

    # add edges to skeleton file
    for e in nxg.edges():
        this_anno.addEdge(this_anno.node_ID_to_node[e[0]], this_anno.node_ID_to_node[e[1]])

    return this_anno

def get_skel_times_from_nx_skel(nxg):
    nodes_to_remove = []
    ts = []
    for n in nxg.nodes():

        try:
            ts.append(int(nxg.node[n]['ts']))
        except KeyError:
            nodes_to_remove.append(n)
            continue

    [nxg.remove_node(n) for n in nodes_to_remove]
    #if len(nodes_to_remove):
    #    print('Had to remove {0} nodes without time stamp'.format(len(nodes_to_remove)))

    try:
        skel_t = (max(ts) - min(ts))/1000.
    except:
        skel_t = 0.
    return skel_t



def update_RAG_with_reconnects(reconnect_folder = '/mnt/j0126/areaxfs_v10/reconnect_tasks/final_tasks/',
                               path_to_skeletons='/mnt/j0126/areaxfs_v10/reconnect_tasks/traced_skeletons/',
                               path_to_reconn_rags='/mnt/j0126/areaxfs_v10/reconnect_tasks/resulting_ssv_rags/'):

    """
    Applies the reconnect skeleton tasks to an existing RAG by adding edges. Requires a Knossos segmentation
    dataset, from which the segmentation IDs are collected.

    :param reconnect_folder:
    :param path_to_skeletons:
    :param path_to_reconn_rags:
    :return:
    """


    # load rag
    rag = load_best_bet_rag()

    # load all reconnects
    kzips = glob.glob(reconnect_folder + '*.k.zip')

    parsing_errors = []
    task_dicts = []
    for kzip in kzips:
        try:
            task_dicts.append(analyze_j0126_reconnector_task(kzip))
            print('Successfully parsed task {0}'.format(kzip))
        except:
            parsing_errors.append(kzip)
            print('Error parsing task {0}'.format(kzip))

    #all_recon_tasks = flatten([t['k_annos'] for t in task_dicts])

    all_recon_tasks = []
    for task_dict in task_dicts:
        all_recon_tasks.extend(task_dict['k_annos'])

    all_src_ids = []
    for task_dict in task_dicts:
        all_src_ids.extend(task_dict['src_ids'])

    all_src_coords = []
    for task_dict in task_dicts:
        all_src_coords.extend(task_dict['src_coords'])


    print('Got in total {0} tasks'.format(len(all_recon_tasks)))
    # filter the skeletons that do not reconnect anything, defined by less than 5 reconnect nodes

    positive_reconnects = [a for a in all_recon_tasks if len(a.getNodes()) > 5]
    print('Got in total {0} reconnects > 5 nodes'.format(len(positive_reconnects)))
    print('Total parsing errors: {0}'.format(len(parsing_errors)))

    #return positive_reconnects
    # contains additional edges for RAG from manual reconnects
    additional_edges = []

    workers = init_node_to_sv_id_workers()
    total_reconnects = len(positive_reconnects)
    recon_cnt = 0.
    rag_extension = []
    start = time.time()
    #print(all_recon_tasks)
    #print(all_src_coords)
    #print(all_src_ids)

    unmapped_nodes_cnt = 0

    for reconnect_anno, src_coords, src_id in zip(all_recon_tasks, all_src_coords, all_src_ids):
        #print(src_coords)

        #print(src_id)

        if len(reconnect_anno.getNodes()) < 5:
            continue

        recon_cnt += 1.
        print('Reconnects done: {0}%'.format(recon_cnt/total_reconnects * 100.))
        mapped_nodes = []


        node1 = skeleton.SkeletonNode()
        node1.from_scratch(reconnect_anno, *src_coords[0])
        node1.setPureComment('source 1')
        reconnect_anno.addNode(node1)

        node2 = skeleton.SkeletonNode()
        node2.from_scratch(reconnect_anno, *src_coords[1])
        node2.setPureComment('source 2')
        reconnect_anno.addNode(node2)

        reconnect_anno.addEdge(node1, node2)

        # connect source 1 with the closest node an annotator made if there is
        # one
        kd_tree = su.KDtree(reconnect_anno.getNodes(),
                            [n.getCoordinate() for n in reconnect_anno.getNodes()])

        nodes, dists = kd_tree.query_k_nearest([src_coords[0]], k = 3,
                                                  return_dists = True)
        for node, dist in zip(nodes, dists):
            # link the source 1 node with the first annotator placed node in the
            # skeleton; query k = 3 is necessary to ensure that one of the hits
            # is an annotator created node!
            if not 'source' in node.getPureComment():
                reconnect_anno.addEdge(node1, node)
                break

        orig_reconnect_anno = copy.deepcopy(reconnect_anno)

        #reconnect_anno.interpolate_nodes(max_node_dist_scaled=200)

        # push nodes onto queue
        anno_nodes = reconnect_anno.getNodes()
        all_node_cnt = len(anno_nodes) + 10
        for skel_node in reconnect_anno.getNodes():
            node_to_sv_mappings_todo_queue.put(skel_node.getCoordinate())

        # push the seed coordinates onto the queue - this is a hack, make sure
        # that they are mapped > 5 times, see below

        [node_to_sv_mappings_todo_queue.put(src_coords[0]) for i in range(5)]
        [node_to_sv_mappings_todo_queue.put(src_coords[1]) for i in range(5)]

        # wait for all nodes to be mapped
        done_nodes = 0
        while done_nodes < all_node_cnt:
            node_coord, sv_id = node_to_sv_mappings_done_queue.get()
            mapped_nodes.append((node_coord, sv_id))
            #done_node.setPureComment(str(sv_id))
            done_nodes += 1
            #print('\r\x1b[K Nodes done: {0} from {1} total'.format(done_nodes, all_node_cnt), end='')
            #time.sleep(0.01)

        all_mapped_sv_ids = [el[1] for el in mapped_nodes]

        # after interpolation, a new kd tree is needed
        kd_tree = su.KDtree(reconnect_anno.getNodes(), [n.getCoordinate() for n in reconnect_anno.getNodes()])

        #print('Len reconnect anno nodes: {0}'.format(len(reconnect_anno.getNodes())))
        #print('Len mapped nodes: {0}'.format(
        #    len(mapped_nodes)))

        for mapped_node in mapped_nodes:
            #print('mapped_nodes[1]: {0}'.format(mapped_node[0]))
            #anno_node = kd_tree.query_ball_point(mapped_node[0], radius=10.)
            anno_node, dist = kd_tree.query_k_nearest([mapped_node[0]], return_dists=True)
            #anno_node = anno_node[0]
            #dist = dist[0]
            #if dist > 0.:
                #print('anno_node: {0}'.format(anno_node))
                #print('dist: {0}'.format(dist))
            # temp storage for mapped sv_id
            anno_node.sv_id = mapped_node[1]


        # count sv_ID occurences and replace infrequent ones with 0
        very_likely = []
        keep_ids = dict()
        #keep_ids[0] = False
        #print('Type sv_id 0 {0}'.format(type(0)))
        unique_ids, counts = np.unique(all_mapped_sv_ids, return_counts=True)
        for sv_id, cnt in zip(unique_ids, counts):
            #print('Type sv_id {0}'.format(type(sv_id)))
            if sv_id != 0:
                if cnt > 4:
                    keep_ids[sv_id] = True
                    very_likely.append(sv_id)
                else:
                    keep_ids[sv_id] = False
            else:
                keep_ids[sv_id] = False

        # prune skeleton using keep_ids to remove nodes over the background
        # and also nodes with sv_ids that were mapped to sv_ids that were too
        # infrequent
        #all_nodes = list(reconnect_anno.getNodes())

        nx_g = su.annotation_to_nx_graph(reconnect_anno)

        n_o_i = list({k for k, v in nx_g.degree() if v > 1})
        # delete in-between nodes that should not be included
        #print('noi {0}'.format(n_o_i))
        for node in n_o_i:
            #print('node.sv_id: {0}'.format(node.sv_id))
            try:
                if keep_ids[node.sv_id] == False:
                    # remove this node, relinking it to neighbors
                    neighbors = list(nx_g[node].keys())
                    #print('Found neighbors {0}'.format(neighbors))
                    # connect all neighbors, take first (arbitrary)
                    if len(neighbors) > 1:
                        src_neighbor = neighbors[0]
                        for neighbor in neighbors[1:]:
                            reconnect_anno.addEdge(src_neighbor, neighbor)
                            nx_g.add_edge(src_neighbor, neighbor)

                    reconnect_anno.removeNode(node)
                    nx_g.remove_node(node)
            except AttributeError:
                unmapped_nodes_cnt += 1
                print('Node {0} of src_id {1} without sv_id.'.format(node, src_id))

        n_o_i = list({k for k, v in nx_g.degree() if v == 1})
        # delete end nodes that should not be included
        for node in n_o_i:
            try:
                if keep_ids[node.sv_id] == False:
                    reconnect_anno.removeNode(node)
                    nx_g.remove_node(node)
            except AttributeError:
                unmapped_nodes_cnt += 1
                print('Node {0} of src_id {1} without sv_id.'.format(node, src_id))

        for n in reconnect_anno.getNodes():
            try:
                n.setPureComment('{0} sv id: {1}'.format(n.getPureComment(),
                                                        n.sv_id))
            except AttributeError:
                n.setPureComment('{0} sv id: {1}'.format(n.getPureComment(),
                                                        'not mapped'))

        # convert the skeleton to nx graph by iterating over the edges of the
        # skeleton annotation; the result is a nx graph representing the
        # topology of the skeleton, but consisting of sv_ids as nodes;

        topo_nx_g = nx.Graph()
        edges = reconnect_anno.getEdges()
        for src_node in list(edges.keys()):
            for trg_node in edges[src_node]:
                try:
                    if src_node.sv_id != trg_node.sv_id:
                        topo_nx_g.add_edge(src_node.sv_id, trg_node.sv_id)
                except AttributeError:
                    pass

        #rag_extension.append(very_likely)
        #very_likely = []

        #unique_likely, counts = np.unique(all_mapped_sv_ids, return_counts=True)
        #for sv_id, cnt in zip(unique_likely, counts):
        #    if sv_id != 0:
        #        if cnt > 3:
        #            very_likely.append(sv_id)
        # list of list of svs that belong together according to the reconnect tracing
        rag_extension.append(topo_nx_g)

        # write topo_nx_g to file for later bidirectionality analysis


        # write annotation with mergelist as kzip to folder
        #for src_id, anno, src_coords in zip(all_recon_tasks['src_ids'],
        #                                    all_recon_tasks['k_annos'],
        #                                    all_recon_tasks['src_coords']):

        skel_obj = skeleton.Skeleton()

        #this_anno = skeleton.SkeletonAnnotation()
        #this_anno.scaling = [10.,10., 20.]

        skel_obj.add_annotation(orig_reconnect_anno)
        orig_reconnect_anno.setComment('tracing')

        skel_obj.add_annotation(reconnect_anno)
        reconnect_anno.setComment('sv topo')
        outfile = path_to_skeletons + 'reconnect_{0}.k.zip'.format(src_id)


        #print('Writing {0}'.format(outfile))
        #skel_paths.append(outfile)
        skel_obj.to_kzip(outfile)

        # add mergelist to the kzip
        buff = ''
        buff += '{0} 0 0 '.format('1')
        for sv_id in very_likely:
            buff += '{0} '.format(sv_id)
        buff += '\n0 0 0\n\n\n'

        with zipfile.ZipFile(outfile, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('mergelist.txt', buff)

        outfile = path_to_reconn_rags + 'ssv_rag_{0}.csv'.format(src_id)
        nx.write_edgelist(topo_nx_g, outfile, delimiter=',', data=False)

    for worker in workers:
        worker.terminate()

    added_rag_edges = 0
    print('Extending global rag')
    for this_ssv_rag in rag_extension:
        new_edges = this_ssv_rag.edges()
        rag.add_edges_from(new_edges)
        added_rag_edges += len(new_edges)

    print('Done extending global rag')
    print('Added in total {0} edges to the rag'.format(added_rag_edges))

    # create merge list with reconnects only for testing
    #for this_ext in rag_extension:
    #    if len(this_ext) > 1:
    #        last_id = this_ext[0]
    #        for this_id in this_ext[1:]:
    #            if not rag.has_edge(last_id, this_id):
    #                rag.add_edge(last_id, this_id)
    #                added_rag_edges.append((last_id, this_id))

    nx_rag_to_knossos_mergelist(rag, path_to_mergelist='/mnt/j0126/areaxfs_v10/RAGs/v4b_20180214_nocb_merges_reconnected_knossos_mergelist.txt')


    nx.write_edgelist(rag, '/mnt/j0126/areaxfs_v10/RAGs/v4b_20180214_nocb_merges_reconnected.txt',
                      delimiter=',', data=False)

    print('Total number unmapped nodes: {0}'.format(unmapped_nodes_cnt))

    print('Mapping {0} took {1}'.format(total_reconnects, (time.time() - start)))
    return


def init_node_to_sv_id_workers():
    """
    Initialize queue daemon workers.
    :return:
    """
    workers = []
    for i in range(32):
        #worker = Thread(target=node_to_sv_id_queue_worker)
        worker = Process(target=node_to_sv_id_queue_worker, args=[node_to_sv_mappings_todo_queue,
                                                                  node_to_sv_mappings_done_queue])
        worker.daemon = True
        worker.start()
        workers.append(worker)

    return workers


def node_to_sv_id_queue_worker(node_to_sv_mappings_todo_queue,
                               node_to_sv_mappings_done_queue,
                               use_brainmaps=False,
                               kd_seg_path='/mnt/j0126_cubed/'):

    if use_brainmaps == True:
        bmi = bm.BrainmapsInteraction(json_key=bm.service_account)
        volume_id = 'j0126_13_v4b_cbs_ext0_fixed'
        project_id = '611024335609'
        dataset_id = 'j0126'

        # volume_id = 'nov2015_mask135_0945_1000_gala_0.95'
        # project_id = '611024335609'
        # dataset_id = 'j0126'
    else:
        kd = kds.KnossosDataset()
        kd.initialize_from_knossos_path(kd_seg_path, cache_size=10)
    while True:
        # this is blocking and therefore fine
        # skel_node = node_to_sv_mappings_todo_queue.get()
        corner_xyz = node_to_sv_mappings_todo_queue.get()
        # corner_xyz = skel_node.getCoordinate()
        # hack to reverse xy
        # corner_xyz = [corner_xyz[0], corner_xyz[1], corner_xyz[2]]

        corner_xyz[0], corner_xyz[1], corner_xyz[2] = corner_xyz[0] - 1, corner_xyz[1] - 1, corner_xyz[2] - 1
        oob = False
        if corner_xyz[0] <= 0: oob = True  # corner_xyz[0] = 1; oob = True
        if corner_xyz[1] <= 0: oob = True  # corner_xyz[1] = 1; oob = True
        if corner_xyz[2] <= 0: oob = True  # corner_xyz[2] = 1; oob = True

        if corner_xyz[0] > 10500: oob = True  # corner_xyz[0] = 10500; oob = True
        if corner_xyz[1] > 10770: oob = True  # corner_xyz[1] = 10770; oob = True
        if corner_xyz[2] > 5698: oob = True  # corner_xyz[2] = 5698; oob = True

        # if corner_xyz[0] > 5000: corner_xyz[0] = 5000
        # if corner_xyz[1] > 5000: corner_xyz[1] = 5000
        # if corner_xyz[2] > 5000: corner_xyz[2] = 5000

        attempt = 0
        succes = False
        while not succes:
            try:
                if use_brainmaps:
                    node_sample = bmi.get_subvolume_chunk(project_id=project_id,
                                                          dataset_id=dataset_id,
                                                          volume_id=volume_id,
                                                          corner_z=corner_xyz[0],
                                                          corner_y=corner_xyz[1],
                                                          corner_x=corner_xyz[2],
                                                          size_z=2,
                                                          size_y=2,
                                                          size_x=2,
                                                          subvol_format='RAW_SNAPPY',
                                                          gzip=0)
                else:
                    if not oob:
                        node_sample = kd.from_overlaycubes_to_matrix((1, 1, 1), corner_xyz, verbose=False,
                                                                     show_progress=False, nb_threads=1)
                succes = True
            except Exception as e:

                print("".join(traceback.format_exception(*sys.exc_info())))
                print(str(e))
                attempt += 1
                time.sleep(10)
                print('Retrying subvol download, attempt: {0}, subvol: {1}, reinitializing bmi'.format(attempt,
                                                                                                       (corner_xyz)))
                if use_brainmaps == True:
                    bmi = bm.BrainmapsInteraction(json_key=bm.service_account)

        if not oob:
            # find most frequent id

            node_sample = flatten(node_sample.tolist())
            #print(node_sample)
            # print('got node sample {0}'.format(node_sample))
            from collections import Counter
            cnt = Counter(node_sample)
            sv_id = cnt.most_common()[0][0]
        else:
            sv_id = 0

        # sv_id = np.argmax(np.bincount(node_sample.reshape([node_sample.size]).astype(np.int64)))
        corner_xyz[0], corner_xyz[1], corner_xyz[2] = corner_xyz[0] + 1, corner_xyz[1] + 1, corner_xyz[2] + 1
        node_to_sv_mappings_done_queue.put((corner_xyz, sv_id))
        node_to_sv_mappings_todo_queue.task_done()

    return

def nx_rag_to_knossos_mergelist(rag, path_to_mergelist):
    """
    Converts a networkx rag to knossos merge list format.
    :param rag:
    :param path_to_mergelist:
    :return:
    """

    if not rag:
        rag = load_best_bet_rag()

    all_ccs = nx.connected_components(rag)

    with open(path_to_mergelist, 'w') as ml_fh:
        buff = ''
        for obj_cnt, cc in enumerate(all_ccs):
            buff += '{0} 0 0 '.format(obj_cnt)
            for sv_id in cc:
                buff += '{0} '.format(sv_id)
            buff += '\n0 0 0\n\n\n'
        ml_fh.write(buff)

    return