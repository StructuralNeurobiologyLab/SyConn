import copy
import numpy as np
import os
import random
import re
from collections import Counter
from scipy import spatial
from scipy.cluster import hierarchy

import networkx as nx
from newskeleton import SkeletonNode, NewSkeleton, SkeletonAnnotation

import syconn.new_skeleton.NewSkeletonUtils as nsu
from syconn.utils.basics import euclidian_distance, FloatCoordinate


class KDtree:
    """
    scipy based KD-tree wrapper class that allows efficient spatial searches
    for arbitrary python objects.
    """
    
    def __init__(self, nodes, coords=None):
        # spatial.cKDtree is reported to be 200-1000 times faster
        # however, query_ball_point is only included in very recent scipy
        # packages, not yet shipped
        # with Ubuntu 12.10 (and a manual scipy installation can be  messy)

        self.lookup = list(nodes)

        if not coords:
            self.coords = np.array([node.getCoordinate_scaled()
                                    for node in nodes])

            self.tree = spatial.cKDTree(self.coords)
        else:
            self.coords = coords
            # the ordering of coords and nodes must be the same!!!
            self.tree = spatial.cKDTree(self.coords)
            
        return

    def __str__(self):
        return ', '.join([str(x) for x in self.lookup])

    # enables efficient pickling of cKDtree
    def __getstate__(self):
        return (self.lookup, self.coords)

    # enables efficient pickling of cKDtree
    def __setstate__(self, state):
        self.lookup, coords = state
        self.tree = spatial.cKDTree(self.coords)

    def query_k_nearest(self, coords, k=1, return_dists=False):
        # This function was written to replace queryNN which is still kept
        # for backwards compatibility
        dists, obj_lookup_IDs = self.tree.query(np.array(coords), k=k)
        try:
            if not return_dists:
                return [self.lookup[ID] for ID in obj_lookup_IDs.tolist()]
            else:
                return [self.lookup[ID] for ID in obj_lookup_IDs.tolist()],dists
        except AttributeError:
            if not return_dists:
                return self.lookup[obj_lookup_IDs]
            else:
                return self.lookup[obj_lookup_IDs], dists

    def query_nearest_node(self, coords):
        if isinstance(coords[0], (int, float)):
            coords = [coords]
        return self.queryNN(coords)

    def queryNN(self, coords):
        # element num 1 contains the array indices for our lookup table
        nodes = [self.lookup[i] for i in
            list(self.tree.query(np.array(coords)))[1].tolist()]
            
        return nodes

    def query_ball_point(self, coord, radius):
        listOfNodeLists = self.tree.query_ball_point(np.array(coord), radius)
        resultList = []        
        for listOfNodes in listOfNodeLists:
            if type(listOfNodes) is list:
                resultList.append([self.lookup[i] for i in listOfNodes])
            else:
                resultList.append(self.lookup[listOfNodes])
        
        # this would make the list flat:
        # resultList = [item for sublist in resultList for item in sublist]

        return resultList

    def query_ball_tree(self, other, radius):
        results = self.tree.query_ball_tree(other.tree, radius)

def avg_annotation_inter_node_distance(anno, outlier_filter=2000.):
    """
    Calculates the average inter node distance for an annotation. Candidate
    for inclusion into the skeleton annotation object.

    :param anno: annotation object
    :param outlier_filter: float, ignore distances higher than value
    :return: float
    """
    edge_cnt = 0
    total_length = 0.

    for from_node, to_node in anno.iter_edges():
        this_dist = euclidian_distance(from_node.getCoordinate_scaled(),
                           to_node.getCoordinate_scaled())
        if this_dist < outlier_filter:
            total_length += this_dist
            edge_cnt += 1

    if edge_cnt:
        avg_dist = total_length / float(edge_cnt)
        return avg_dist
    else:
        print('No edges in current annotation, cannot calculate inter node '
              'distance.')
        return

def annoToKDtree(annotations):
    """
    Uses scipy kd-trees. Scaling must be set.
    Only nodes are added to the tree, no edges currently. Creates
    one kd-tree for every annotation.
    
    """

    if not isinstance(annotations, (set, list)):
        annotations = [annotations]

    trees = [KDtree(anno.getNodes()) for anno in annotations]
    
    return trees


def annosToKDtree(annotations):
    """

    Uses scipy kd-trees. Scaling must be set.
    Only nodes are added to the tree, no edges currently. Inserts many
    annotations into a single kd-tree.
    
    """
    nodes = []
    for anno in annotations:
        nodes.extend([node for node in anno.getNodes()])
    tree = KDtree(nodes)
    return tree
    
    

def euclNodeDist(node1, node2):
    n1 = np.array(node1.getCoordinate_scaled())
    n2 = np.array(node2.getCoordinate_scaled())
    return np.linalg.norm(n1-n2)

def getAnnoWithMostNodes(skeleton):
    """Many users submit nml files with multiple trees (annotations) that
    were created accidentally.
    This functions checks how many nodes each annotation contains
    and returns the annotation that contains most. Returns none in case
    there are trees with more than 1 node in each."""
    annos = skeleton.getAnnotations()
    # most probable anno is the annotation with the most nodes of all
    annosSortedByNodeNums = zip([len(anno.getNodes()) for anno in annos], list(annos))
    annosSortedByNodeNums.sort()
    mostProbableAnno = list(annosSortedByNodeNums.pop(-1))[1]
    if len(mostProbableAnno.getNodes()) == 0:
        mostProbableAnno = None
    #if annosSortedByNodeNums:
    #    for numNodes, anno in annosSortedByNodeNums:
    #        if numNodes > 1:
    #            mostProbableAnno = None
    #            break

    return mostProbableAnno

#def loadNMLsInDir(directory):
#    """Loads all .nml files inside a directory."""
#    annotations = []
#    failedFiles = []
#
#    allNMLfiles = [file for file in os.listdir(directory) if file.lower().endswith('.nml')]    
#    
#    for nmlfile in allNMLfiles:
#       
#        
#        #anno = getAnnoWithMostNodes(skeletonObj)
#        if anno:
#            anno.scaling = (10,10,20)
#            anno.filename = nmlfile 
#            annotations.append(anno)
#        else:
#            failedFiles.append(nmlfile)
#            print 'failed to extract a single plausible annotaton: ' + nmlfile    
#
#    return (annotations)


def loadj0126ConsensusNMLsInDir(directory):
    """Loads all .nml files inside a directory and give warnings."""
    annotations = []
    annodict = dict()

    allNMLfiles = [file for file in os.listdir(directory)
                   if file.lower().endswith('.nml')]
    
    for nmlfile in allNMLfiles:
        print 'loading ' + nmlfile
        annos = loadj0126NML(os.path.join(directory, nmlfile))
        
        # test the number of annotations, must be 1
        if len(annos) > 1:
            # check if only one annotation actually contains nodes
            
            thisanno = None
            for currAnno in annos:
                if len(currAnno.getNodes()) > 0:
                    if not thisanno:
                        thisanno = currAnno
                    else:
                        raise Exception('File ' + nmlfile +
                                        ' contains more than'
                                        ' one annotation with nodes.')
                        print 'File ' + nmlfile +\
                              ' contains more than one annotation with nodes.'
            anno = thisanno 
        else:
            anno = annos[0]


        nxG = annoToNXGraph(anno)
        anno.pathLen = nxG[0].size(weight='weight') / 1000 # to microns
        anno.numBranchNodes = len(list({k for k, v
                                        in nxG[0].degree().iteritems()
                                        if v > 2}))
        anno.branchDensity = anno.numBranchNodes / anno.pathLen  

        # only a single connected component allowed                
        currNxg = annoToNXGraph(anno)[0]
        if nx.number_connected_components(currNxg) > 1:
            print 'File ' + nmlfile + ' contains more than' \
                                      ' one connected component.'
            for nodesInC in nx.connected_components(currNxg):
                print 'This connected component contains ' +\
                      str(len(nodesInC)) + '  nodes.'
                #for node in nodesInC:
                #    print str(node.getID())
                    
            raise Exception('File ' + nmlfile +
                            ' contains more than one connected component.')
            
        annotations.append(anno)
        annodict[anno.seedID] = anno
 
    return (annotations, annodict)

def find_and_validate_contacts(dir_1, dir_2, scaling=(9.0, 9.0, 21.0)):
    files = dict()
    files['dir_1'] = glob.glob(dir_1 + '/*.nml') + glob.glob(dir_1 + '/*.k.zip')
    files['dir_2'] = glob.glob(dir_2 + '/*.nml') + glob.glob(dir_2 + '/*.k.zip')

    annotations = dict()

    print('Loading files.')
    for cur_dir, cur_files in files.iteritems():
        for cur_f in cur_files:
            print cur_f
            s = NewSkeleton()
            s = s.fromNml(cur_f, scaling=scaling)
            for cur_a in s.getAnnotations():
                # Not exactly correct but does the job of removing file name
                # extension in the usual case

                if not cur_a.getNodes():
                    continue

                cur_a.tag = os.path.basename(cur_f.replace('.k.zip', '').replace(
                    '.nml', ''))
                annotations.setdefault(cur_dir, []).append(cur_a)

    print('Analyzing contacts.')
    contacts = find_contacts_between(
        annotations['dir_1'],
        annotations['dir_2'],
        lumping=lambda x: x.tag,
        cluster_filter=True)


    # print('Writing validation output.')
    # s_out = NewSkeleton()
    # for pair_l, partners in contacts.iteritems():
    #     for pair_r in partners:
    #         pair_annotation = SkeletonAnnotation()
    #         pair_annotation.comment = '%s -> %s' % (
    #             pair_l.annotation.tag,
    #             pair_r.annotation.tag, )
    #         pair_l_coord = pair_l.getCoordinate()
    #         pair_r_coord = pair_r.getCoordinate()
    #         node_l = SkeletonNode().from_scratch(
    #             pair_annotation,
    #             pair_l_coord[0],
    #             pair_l_coord[1],
    #             pair_l_coord[2],)
    #         node_l.setComment('TODO')
    #         pair_annotation.addNode(node_l)
    #         node_r = SkeletonNode().from_scratch(
    #             pair_annotation,
    #             pair_r_coord[0],
    #             pair_r_coord[1],
    #             pair_r_coord[2],)
    #         pair_annotation.addNode(node_r)
    #         pair_annotation.addEdge(node_l, node_r)
    #         s_out.add_annotation(pair_annotation)
    #
    # s_out.toNml('validate_contacts.nml')


def find_contacts_between(from_annotations, to_annotations,
                          spotlight_radius=1000,
                          lumping=None, cluster_filter=True):
    """
    Extension of findContactSites below. This one only finds contacts
    between from_annotations and to_annotations.

    Parameters
    ----------

    one_to_one : boolean
        If true, only return one match per node / other annotation pair.

    exclusion : function
        If one_to_one is true, then this function is applied to all
        SkeletonNode object matched from a given coordinate and for every
        unique result of exclusion, only one of the nodes is retained.
        E.g., the default lambda x: x.annotation, which is used when
        exclusion is None, will cause only one node per matching annotation
        to be retained per coordinate.

    cluster_filter:
        Cluster matches based on their centroids. The result depends strongly
        on the threshold parameter to fclusterdata. Only the shortest-distance
        match will be returned for every cluster.

    Returns
    -------
        contact_sites : dict of list
            node in from_annotations -> list of nodes in to_annotations


    """

    from_lumped = dict()
    to_lumped = dict()

    for cur_from in from_annotations:
        cur_key = lumping(cur_from)
        from_lumped.setdefault(cur_key, []).append(cur_from)
    for cur_to in to_annotations:
        cur_key = lumping(cur_to)
        to_lumped.setdefault(cur_key, []).append(cur_to)

    cluster_filtered_contact_per_pair = dict()
    for cur_from_tag, cur_from in from_lumped.iteritems():
        cur_from_kd = annosToKDtree(cur_from)

        for cur_to_tag, cur_to in to_lumped.iteritems():
            cluster_filtered_contact = dict()
            print('%s -> %s' % (cur_from_tag, cur_to_tag))

            cur_to_kd = annosToKDtree(cur_to)

            contact_sites = cur_from_kd.query_ball_tree(cur_to_kd,
                                                        spotlight_radius)

            if not contact_sites:
                continue

            if cluster_filter:
                flat_contacts = []
                for cur_left, cur_rights in contact_sites.iteritems():
                    for cur_right in cur_rights:
                        flat_contacts.append([cur_left, cur_right])

                flat_centroids = []
                for i, cur_contact in enumerate(flat_contacts):
                    cur_centroid = cur_contact[0].getCoordinate_scaled()
                    #cur_centroid = [x + y for x, y in zip(
                    #    cur_contact[0].getCoordinate_scaled(),
                    #    cur_contact[1].getCoordinate_scaled())]
                    flat_centroids.append(cur_centroid)

                if len(flat_centroids) == 1:
                    # Do not cluster, wouldn't work.
                    cluster_filtered_contact.setdefault(
                        flat_contacts[0][0], []).append(flat_contacts[0][1])
                    continue

                centroids_np = np.array(flat_centroids)
                #import ipdb
                #ipdb.set_trace()
                print centroids_np.shape
                clusters = hierarchy.fclusterdata(centroids_np, 300,
                                                  criterion='distance')
                #print len(set(clusters))

                cluster_id_to_shortest = dict()
                cluster_id_to_length = dict()
                for i, cur_cluster in enumerate(clusters):
                    cur_dst = euclidian_distance(
                        flat_contacts[i][0].getCoordinate_scaled(),
                        flat_contacts[i][1].getCoordinate_scaled())

                    if not cur_cluster in cluster_id_to_length:
                        cluster_id_to_length[cur_cluster] = 1000000000
                        cluster_id_to_shortest[cur_cluster] = None

                    if cur_dst < cluster_id_to_length[cur_cluster]:
                        cluster_id_to_length[cur_cluster] = cur_dst
                        cluster_id_to_shortest[cur_cluster] = [
                            flat_contacts[i][0],
                            flat_contacts[i][1]]

                for cur_pair in cluster_id_to_shortest.itervalues():
                    cluster_filtered_contact.setdefault(cur_pair[0], []).append(
                        cur_pair[1]
                    )

                cluster_filtered_contact_per_pair[
                    '%s-%s' % (cur_from_tag, cur_to_tag, )] =  \
                    cluster_filtered_contact

                print('Writing validation output.')
                s_out = NewSkeleton()
                pair_annotation = SkeletonAnnotation()
                pair_annotation.comment = '%s -> %s' % (
                    cur_from_tag,
                    cur_to_tag, )
                pair_annotation.color = (0.0, 0.0, 1.0, 1.0)
                s_out.add_annotation(pair_annotation)
                for cur_to_anno in cur_to:
                    cur_to_anno.setComment(cur_to_tag)
                    cur_to_anno.color = (1.0, 0.0, 0.0, 1.0)
                    s_out.add_annotation(cur_to_anno)
                for cur_from_anno in cur_from:
                    cur_from_anno.setComment(cur_from_tag)
                    cur_from_anno.color = (0.0, 1.0, 0.0, 1.0)
                    s_out.add_annotation(cur_from_anno)
                for pair_l, partners in cluster_filtered_contact.iteritems():
                    for pair_r in partners:
                        pair_l_coord = pair_l.getCoordinate()
                        pair_r_coord = pair_r.getCoordinate()
                        # The randint here is a horrible workaround around
                        # the sorry state of NewSkeleton
                        node_l = SkeletonNode().from_scratch(
                            pair_annotation,
                            pair_l_coord[0],
                            pair_l_coord[1],
                            pair_l_coord[2],
                            ID=random.randint(5000000, 6000000))
                        node_l.setComment('TODO')
                        pair_annotation.addNode(node_l)
                        node_r = SkeletonNode().from_scratch(
                            pair_annotation,
                            pair_r_coord[0],
                            pair_r_coord[1],
                            pair_r_coord[2],
                            ID=random.randint(5000000, 6000000))
                        pair_annotation.addNode(node_r)
                        pair_annotation.addEdge(node_l, node_r)

                s_out.toNml('%s-%s.nml' % (cur_from_tag, cur_to_tag, ))

    return cluster_filtered_contact_per_pair


def findContactSites(annotations, spotlightRadius=1000):
    """
    Finds node-pairs between different annotations with maximum distance 
    spotlightRadius.
    :annotations: List of annotations
    :spotlightRadius: search radius for kd-tree in nm
    :return: Dictionary containing contact-node-pairs for each annotation. Key is indice of annotation in list "annotations".
    """
    # Insert all annotations into a single kd-tree for fast spatial lookups
    kdtree = annosToKDtree(annotations)

    contactSites = dict()
    for j, anno in enumerate(annotations):
        annoNodes = list(anno.getNodes())
        coords = [node.getCoordinate_scaled() for node in annoNodes]
        # foundNodes is a list of node lists, each node lists corresponds to
        # a single query from an element in coords
        foundNodes = kdtree.query_ball_point(coords, spotlightRadius)
        
        # contact sites contains contact-node-pairs for each annotation
        anno_ContactSites = []
        for i, nodes in enumerate(foundNodes):
            # add anno node as first node in list and nearby nodes of other annos
            new_nodes = [node for node in nodes if node.annotation != anno]
            anno_ContactSites.append([annoNodes[i]] + new_nodes)
        contactSites[j] = anno_ContactSites
    return contactSites

def loadj0126NMLbyRegex(regex):
    
    archiveDir = '/mnt/fs/singvogel/skeletons/Consensus/'

    # extract all nml files of directory
    extension = '.nml'

    # list of all tracer subdirs
    allTracerDirs = [name for name in os.listdir(archiveDir) if os.path.isdir(os.path.join(archiveDir, name))]

    regObj = re.compile(regex, re.IGNORECASE)

    allNMLfiles = [file for file in os.listdir(archiveDir) if file.lower().endswith(extension)]

    matches = {}
    
    for nml in allNMLfiles:
        # perform re matching and copy file to targetDir if sucessful
        mObj = regObj.search(nml)
        if mObj:
            print 'Found nml: ', nml 
            annos = loadj0126NML(nml)
            matches[annos[0].seedID] = annos
    
    return matches


def load_j0251_nml(path_to_file, merge_all_annos=False):
    annos = load_jk_NML(pathToFile=path_to_file,
                        ds_id = 'j0251',
                        scaling=(10.,10.,25.),
                        dataset_dims=[1,270000,1,270000,1,387350],
                        remove_empty_annotations=True)
    return annos

def load_j0256_nml(path_to_file, merge_all_annos=False):
    annos = load_jk_NML(pathToFile=path_to_file,
                        ds_id = 'j0256',
                        scaling=(11.,11.,29.),
                        dataset_dims=[1,151050,1,151050,1,71874],
                        remove_empty_annotations=True)
    return annos


def loadj0126NML(path_to_file, merge_all_annos=False):
    #print('Loading: ' + os.path.basename(path_to_file))
    annos = load_jk_NML(pathToFile=path_to_file,
                        ds_id = 'j0126',
                        merge_all_annos=merge_all_annos,
                        scaling=(9., 9., 20.),
                        dataset_dims = [1, 108810, 1, 106250, 1, 115220],
                        remove_empty_annotations=True)
    return annos


def load_jk_NML(pathToFile,
                ds_id,
                scaling,
                dataset_dims,
                remove_empty_annotations,
                merge_all_annos=False):
    """
    
    Loads a NML file and add some specific attributes to the returned
    skeleton annotation objects.        
    
    """
    
    annos = []
    skeletonObj = NewSkeleton()
    skeletonObj.fromNmlcTree(pathToFile, scaling=scaling)

    filename = os.path.basename(pathToFile)
    seed = ''
    tracer = ''

    # old wiki format:
    mobj=re.search(ds_id + '-(?P<seed>.*)-(?P<tracer>[A-Za-z]*)\.\d{3}\.nml$',
                   filename)
    if mobj:
        tracer = mobj.group('tracer')
        seed = mobj.group('seed')

    # hdbrain format:
    mobj=re.search(ds_id+'.*-(?P<seed>.*)-(?P<tracer>[A-Za-z]*)-\d{8}-\d{6}'
                   '.*((nml)|(k.zip))$', filename)
    if mobj:
        tracer = mobj.group('tracer')
        seed = mobj.group('seed')


    if merge_all_annos and len(skeletonObj.annotations) > 1:
        for anno_to_merge in skeletonObj.annotations[1:]:
            skeletonObj.annotations[0] =\
                nsu.merge_annotations(skeletonObj.annotations[0], anno_to_merge)
    num_nodes_total = len(skeletonObj.getNodes())
    for anno in skeletonObj.annotations:
        if remove_empty_annotations:
            if len(anno.getNodes()) == 0:
                continue

        if filename == None:
            raise Exception('Filename none')

        if skeletonObj.getSkeletonTime() >= 0 and skeletonObj.getIdleTime() \
                >= 0:
            anno.avg_node_time = ((skeletonObj.getSkeletonTime() - \
                         skeletonObj.getIdleTime()) / 1000.)

            anno.avg_node_time /= num_nodes_total
        else:
            anno.avg_node_time = 0.


        anno.scaling = scaling
        anno.filename = filename
        anno.seedID = seed
        anno.color = (1.0, 0.0, 0.0)
        anno.datasetDims = dataset_dims
        anno.username = tracer
        annos.append(anno)
        
    skeletonObj.scaling = scaling
    
    return annos
    
def getNMLannos(filename):
    skel = ns.NewSkeleton()
    skel.fromNml(filename)
    annos = skel.getAnnotations()
    annotations = []
    for anno in annos:
        anno.scaling = (10,10,20)
    
        anno.filename = filename
        annotations.append(anno)
    
    return annotations
    
def get_all_comment_nodes(annos):
    """

    Returns a list of all nodes with a comment for given annotations.

    """

    if type(annos) == list:
        nodes = []
        for anno in annos:
            nodes.extend([node for node in anno.getNodes() if node.getPureComment()])
    else:
        nodes = [node for node in annos.getNodes() if node.getPureComment()]

    return nodes
    
def get_all_node_comments(annos):
    """
    
    Returns a list of strings (all node comments) for given annotations.

    """
    
    nodes = get_all_comment_nodes(annos)
    return [node.getPureComment() for node in nodes]    

def annotation_matcher(annotations,
                      spotlightRadius=400,
                      samplesize=400,
                      thres=0.25,
                      visual_inspection=False,
                      skip_if_too_small=True,
                      write_match_group_nmls = '',
                      write_to_file=''):

    """

    Algorithm to group annotations that might be the same
    neuronal structure. Based on a spatial search of a
    subpopulation of nodes of each annotation in all other annotations.
    Uses the scipy.spatial kd-tree implementation of fast spatial searches and
    networkx connected components for the grouping of the annotations.

    Parameters
    ----------

    annotations :    list of NewSkeleton.annotation objects
    spotlightRadius: int [nm] that defines the radius of a sphere  around a
                    currently
                    searched node used to find nodes in other annotations
                    500 works well, be careful, not easy to tweak.
    samplesize:     int of how many nodes to randomly sample from each
                    annotation for
                    the matching; higher numbers slows the matching down, but
                    makes the result more accurate
    skip_if_too_small: bool that defines whether to skip an annotation if it
                       has less than samplesize nodes. If False (default),
                       all available nodes will be used.
    thres:          float [0,1]; fraction of nodes of samplesize that need to
                    have a match in another annotation for both annotations to
                    be reported as probable match partners; only reciprocal
                    matches are currently accepted.

    visualize_matches: bool, uses matplotlib and mayavi for debugging

    write_to_file: str of path to filename for the output or empty to not write
                   a file

    Returns
    -------

    groups: list of lists of annotation objects that are probably the same
            biological structure



    """

    # import "difficult" libraries in this namespace only, to prevent trouble
    if visual_inspection:
        import matplotlib.pyplot as plt
        #from Visualization import plotting as skelplot

    if skip_if_too_small:
        annotations = [
                x for x in annotations if len(x.getNodes()) >= 100]

    print('Starting with kd-tree construction')
    # Insert all annotations into a single kd-tree for fast spatial lookups
    kdtree = annosToKDtree(annotations)
    print('Done with kd-tree construction, starting with matching')
    # Query x random nodes of each annotation against the kd-tree
    # (query_ball_point search).
    annoMatchGroup = dict()
    already_processed_same = dict()
    for anno in annotations:
        if already_processed_same.has_key(anno):
            print('Already processed, skipping ' + anno.filename)
            continue
        try:
            coords = [node.getCoordinate_scaled() for
                      node in random.sample(anno.getNodes(), samplesize)]
        except ValueError:
            coords = [node.getCoordinate_scaled()
                      for node in
                      random.sample(anno.getNodes(),len(anno.getNodes()))]
            #print 'Anno ' + anno.filename +\
            #      ' contained less nodes than sample size, using only ' +\
            #      str(len(anno.getNodes())) + ' nodes.'
        #raise()
        # foundNodes is a list of node lists, each node lists corresponds to
        # a single query from an element in coords
        foundNodes = kdtree.query_ball_point(coords, spotlightRadius)

        # ignore identical annotations
        foundNodes_identical = kdtree.query_ball_point(coords, 1.)

        annoMatchGroup[anno] = dict()
        matchSets = [] # list of match sets
        for nodeList in foundNodes:
            matchSets.append(set([node.annotation for node in nodeList]))

        candidate_same = []
        for nodeList in foundNodes_identical:
            for node in nodeList:
                candidate_same.append(node.annotation)
        anno_node_len = len(anno.getNodes())
        same_annos = dict()
        candidates = Counter(candidate_same).most_common()
        for cand, num in candidates:
            if num > 100:
                if anno_node_len == len(cand.getNodes()):
                    same_annos[cand] = True
                    already_processed_same[cand] = True
                    #if not cand.filename == anno.filename:
                    #    print(cand.filename + ' and ' + anno.filename + '
                    # are the same')



        # for src_node, matched_nodes in zip(coords, foundNodes):
        #     for match_node in matched_nodes:
        #         if anno != match_node.anno:

                
        # builds a dictionary with an entry for each other anno that was
        # close to at least one of the matched nodes

        for matchSet in matchSets:
            for match in matchSet:
                #if same_annos.has_key(match):
                #    print 'same match'
                if not same_annos.has_key(match):
                #if anno != match:
                    if anno_node_len < samplesize:
                        this_samplesize = anno_node_len
                    else:
                        this_samplesize = samplesize
                    if annoMatchGroup[anno].has_key(match):
                        annoMatchGroup[anno][match] += (1.0 / this_samplesize)
                    else:
                        annoMatchGroup[anno][match] = (1.0 / this_samplesize)

    # possible to write this nesting in a more pythonic way?
    matchGraph = nx.Graph()
    print('Done with kd-queries, starting with grouping')

    # some variables to spot problems easier:
    all_matches = []
    weak_matches_not_included = []
    weak_matches_possibly_included = []

    for anno in annoMatchGroup.keys():
        for match in annoMatchGroup[anno].keys():
            if visual_inspection:
                all_matches.append(annoMatchGroup[anno][match])

            if annoMatchGroup[anno][match] > 0.2 and\
                annoMatchGroup[anno][match] <= thres:
                # magic number 0.05, this really depends on your data :(
                weak_matches_not_included.append((anno, match))

                #print('weak match: ' + str(annoMatchGroup[anno][match])+ ' '+\
                #      anno.seedID + ' -> ' + match.seedID)

            if annoMatchGroup[anno][match] >= thres and\
                annoMatchGroup[anno][match] < thres + 0.1:
                 # magic number 0.2, this really depends on your data:(
                weak_matches_possibly_included.append((anno, match))


            if annoMatchGroup[anno][match] > thres:
                if annoMatchGroup.has_key(match):
                    if annoMatchGroup[match].has_key(anno):
                        #if annoMatchGroup[match][anno] > thres:
                            # reciprocal match found that satisfies threshold
                        matchGraph.add_edge(match, anno)
    print('Done with match graph construction, starting with connected comp')
    # perform connected components to get the annotations that probably
    # belong together
    groups = list(nx.connected_components(matchGraph))
    #groups = list(nx.find_cliques(matchGraph))

    # add annotations without a reciprocal partner for completeness
    for anno in annotations:
        if not matchGraph.has_node(anno):
            # add only one out of many equal annotations
            #if already_processed_same.has_key(anno):
            #print('Already processed, skipping ' + anno.filename)
            #    continue
            groups.append([anno])

    #

    # error checking / visualization codes follows
    if visual_inspection:
        # visualize weak matches for easier error detection and plot match histo
        plt.figure()
        plt.hist(all_matches, 100, (0.0, 1.0))
        plt.ylabel('number of matches')
        plt.xlabel('match quality (fraction of nodes with match)')
        plt.title('Match quality histogram')

        # # visualize weak matches directly, one by one
        # for src_anno, match_anno in weak_matches_not_included:
        #     print('Now we will show weak matches that were not accepted as'
        #           ' matches. Only different skeletons should show up.')
        #     src_anno.color = (1.0, 0.0, 0.0)
        #     match_anno.color = (0.0, 0.0, 1.0)
        #     skelplot.visualize_annotation(src_anno)
        #     skelplot.add_anno_to_mayavi_window(match_anno)
        #     print('currently shown weak match: ' +\
        #           str(annoMatchGroup[src_anno][match_anno]) + ' ' +\
        #           src_anno.seedID + ' ,red -> ' + match_anno.seedID + ' ,blue')
        #     raw_input('Enter for next weak match (put mayavi win on top)')
        #
        # for src_anno, match_anno in weak_matches_possibly_included:
        #     print('Now we show weak matches that were possibly (reciprocity'
        #           ' test can still exclude them) accepted as'
        #           ' matches. Only equal skeletons should show up.')
        #     src_anno.color = (1.0, 0.0, 0.0)
        #     match_anno.color = (0.0, 0.0, 1.0)
        #     skelplot.visualize_annotation(src_anno)
        #     skelplot.add_anno_to_mayavi_window(match_anno)
        #     print('currently shown weak match: ' +\
        #           str(annoMatchGroup[src_anno][match_anno]) + ' ' +\
        #           src_anno.seedID + ' ,red -> ' + match_anno.seedID + ' ,blue')
        #     raw_input('Enter for next weak match (put mayavi win on top)')


    if write_match_group_nmls:
        print('Done with grouping, starting to write out match group nmls')
        for cnt, group in enumerate(groups):
            skel_obj = NewSkeleton()

            colors=skelplot.get_different_rgba_colors(len(group),rgb_only=True)

            for cnt2, anno in enumerate(group):
                anno.setComment(anno.username + ' ' + anno.seedID)
                anno.color = colors[cnt2]
                skel_obj.add_annotation(anno)

            skel_obj.toNml(write_match_group_nmls+'group_' + str(cnt) + '.nml')

    if write_to_file:
        outfile = open(write_to_file, 'w')
        for group in groups:
            for anno in group:
                if anno.username and anno.seedID:
                    outfile.write(anno.username + ':' + anno.seedID + ' ')
                else:
                    outfile.write(anno.filename + ' ')
            outfile.write('\n')
        outfile.close()
    
    return groups, matchGraph


def prune_stub_branches(annotations,
                        len_thres=1000.,
                        preserve_annotations=True):
    """
    Removes short stub branches, that are often added by annotators but
    hardly represent true morphology.

    :param annotations:
    :param len_thres:
    :param preserve_annotations:
    :return:
    """


    pruned_annotations = []

    if not type(annotations) == list:
        annotations = [annotations]

    for anno in annotations:
        if preserve_annotations:
            new_anno = copy.deepcopy(anno)
        nx_g = annotation_to_nx_graph(new_anno)
        
        # find all tip nodes in an anno, ie degree 1 nodes
        end_nodes = list({k for k, v in nx_g.degree().iteritems() if v == 1})

        # DFS to first branch node
        for end_node in end_nodes:
            prune_nodes = []
            for curr_node in nx.traversal.dfs_preorder_nodes(nx_g, end_node):
                if nx_g.degree(curr_node) > 2:
                    b_len = nx.shortest_path_length(nx_g, end_node,
                                                    curr_node,
                                                    weight='weight')
                    if b_len < len_thres:
                        # remove this stub, i.e. prune the nodes that were
                        # collected on our way to the branch point
                        for prune_node in prune_nodes:
                            new_anno.removeNode(prune_node)
                        break
                    else:
                        break
                # add this node to the list of nodes that MAY get removed
                # in case a stub is detected later in the loop
                prune_nodes.append(curr_node)
                
        pruned_annotations.append(new_anno)

    if len(pruned_annotations) == 1:
        return pruned_annotations[0]
    else:
        return pruned_annotations


def estimateDisagreementRate(annotations, spotlightRadius):
    """Looks at all nodes >deg 2 and searches for corresponding
    nodes in the other annotations."""

   # nxGs = annoToNXGraph(annotations)
    #bNodes = []
    #eNodes = []

    #for g in nxGs:
        # get a set of all branch nodes deg > 2 for all annotations
    #    bNodes.append({k for k, v in g.degree().iteritems() if v > 2})
        # get a set of all end nodes for each annotation, i.e. deg == 1
    #    eNodes.append({k for k, v in g.degree().iteritems() if v == 1})

    #for bNodes1, eNodes1, nxG, kdT in itertools.izip(bNodes, eNodes, nxGs, kdTs):
    #    for bNode1, bNode2 in itertools.combinations(bNodes1, bNodes2):
            # spatial lookup on all bNodes of a single anno with all bNodes in all other annos
            # all lonely bNodes are stored
    #        if euclDistNodes(bNode1, Bnode2) < spotlightRadius:
    #            if len(bNode1.getChildren()) != len(bNode2.getChildren()):
    #                misbNodes.
                

     #       allSpottedNodes = kdT.query_ball_point(bNode1.getCoordinate_scaled(), spotligthRadius)
            # is there exactly ONE node with SAME deg in the spotlight radius?
       #     [g.degree[node] for node in nodes]
      #      
            
            
    # find lonely e(nd) and b(ranch) nodes
    #kdTs = annoToKDtree(annotations)
    #for bNode in bNodes:
        # for each bNode, query all kdTs with spotlightRadius
     #   for kdT, g in itertools.izip(kdTs, nxGs):
       #     nodes = kdT.query_ball_point(bNode.getCoordinate_scaled(), spotligthRadius)
      #      # is there exactly ONE node with SAME deg in the spotlight radius?
        #    [g.degree[node] for node in nodes]
                       
    return




def genSeedNodes(annotations, reqRedundancy, spotlightRadius):
    """ Returns a set of tuples of seed nodes and tracing radii
    that can be distributed to tracers. If no seed nodes are returned,
    no end nodes were found that had less redundancy than
    reqRedundancy, i.e. the annotation is complete."""
    newAnnos = []
    for anno in annotations:
        anno.nxG = annoToNXGraph(anno)[0]
        anno.kdT = annoToKDtree(anno)[0]
        newAnnos.append(anno)

    annotations = set(newAnnos)

    seedNodes = []

    for anno in annotations:
        # get a set of all end nodes for each annotation, i.e. deg == 1
        anno.eNodes = list({k for k, v in anno.nxG.degree().iteritems() if v == 1})
        anno.lonelyENodes = dict()
        # remove non-lonely eNodes of current anno: the tracing is complete if lonelyENodes remains empty
        for eNode in anno.eNodes:
            anno.lonelyENodes[eNode] = 0
            for otherAnno in annotations.difference(set([anno])):
                if otherAnno.kdT.query_ball_point(eNode.getCoordinate_scaled(), spotlightRadius):
                    anno.lonelyENodes[eNode] += 1
            if anno.lonelyENodes.has_key(eNode):
                if anno.lonelyENodes[eNode] > reqRedundancy:
                    del(anno.lonelyENodes[eNode])
                    break
        #seedNodes.append(anno.lonelyENodes)
    #return seedNodes
        # generate new seed nodes to test the validity (by tracing then) of the lonely end nodes of the current anno
        for leNode in anno.lonelyENodes.keys():
            stopSearch = 0
            # find first branch node
            for currNode in nx.traversal.dfs_preorder_nodes(anno.nxG, leNode):
                if stopSearch:
                    break
                if anno.nxG.degree(currNode) > 2:
                    # close to node of another annotation?
                    foundNodesAllOthers = []
                    for otherAnno in annotations.difference(set([anno])):
                        foundNodesThisAnno = otherAnno.kdT.query_ball_point(currNode.getCoordinate_scaled(), spotlightRadius)
                        if foundNodesThisAnno:
                            foundNodesAllOthers.extend(foundNodesThisAnno)
                    if foundNodesAllOthers:
                       # just take the first element, which one is arbitrary
                        seedNodes.append(foundNodesAllOthers[0])
                        stopSearch = 1
                        break
                    
                        # tdItem: remove lonely end nodes that are covered
                        # by the same seed point ( remove branch node from
                        #graph, DFS, find all end nodes in DFS, remove those
                        #from initial lonely end node list for this anno)
    
# round 1: find all lonely stop nodes in all available annotations;
# -> DFS from lonely stop nodes to first encountered branch node in proximity to one OTHER annotation; remove all lonely stop nodes before the branch node (remove found branch node and do another DFS from the stop node, remove all other encountered stop nodes)
# -> create a seed node at the position of the OTHER annotation node close to the branch node and distribute; limit the radius for tracing to max(shortest path length seed node to all stop nodes in not found branch)

# redundancy can be set by defining a lonelyness criterion for stop nodes (i.e. <2 in proximity means lonely)

# disadvantages: possible that wrong seed points are picked?

# round 2: if the missed branch was correct, the lonely stop node will be gone now. otherwise: more rounds? accept lonely stop node as consequence of false branching after x rounds.

    return seedNodes

def getNodesByCommentRegex(regex, annotations):
    '''


    '''

    matchingNodes = []
    allNodes = []
    regobj = re.compile(regex, re.IGNORECASE)


    # accept list of annotations and single annotation object
    if type(annotations) == list:
        for annotation in annotations:
            allNodes.extend(annotation.getNodes())
    else:
        allNodes = annotations.getNodes()

    matchingNodes = [node for node in allNodes
                     if regobj.search(node.getPureComment())]

    return matchingNodes
    
def getAnnosByCommentRegex(regex, annotations):
    regobj = re.compile(regex, re.IGNORECASE)
    matchingAnnos = [anno for anno in annotations if anno.getComment()] # cumbersome to prefilter here...
    matchingAnnos = [anno for anno in matchingAnnos if regobj.search(anno.getComment())]
    return matchingAnnos

def annotation_to_nx_graph(annotation):
    """
    Creates a network x graph representation of an annotation object.
    :param annotation: NewSkeleton.annotation object
    :return: networkx graph object
    """

    nxG = nx.Graph()
    for node in annotation.getNodes():
        nxG.add_node(node)
        for child in node.getChildren():
            nxG.add_edge(node, child, weight=node.distance_scaled(child))

    return nxG


def write_anno(a, out_fname):
    """
    Write SkeletonAnnotation to file.
    """

    s = NewSkeleton()
    s.add_annotation(a)
    s.toNml(out_fname)


def nx_graph_to_annotation(G, scaling=None):
    """
    Turn a NetworkX graph into a SkeletonAnnotation. Nodes in the graph are
    assumed to be SkeletonNode instances, but only the edges present in the
    NetworkX graph are used, potentially existing NewSkeleton style edges are
    ignored.

    Parameters
    ----------

    G : NetworkX graph
        ... where the nodes are SkeletonNode instances


    Returns
    -------

    a : SkeletonAnnotation instance
        Fresh SkeletonAnnotation, with newly created SkeletonNode objects
    """

    a = SkeletonAnnotation()
    a.scaling = scaling
    new_node_mapping = dict()

    for cur_n in G.nodes_iter():
        x, y, z = cur_n.getCoordinate()
        cur_n_copy = SkeletonNode()
        cur_n_copy.from_scratch(a, x, y, z)
        new_node_mapping[cur_n] = cur_n_copy
        a.addNode(cur_n_copy)

    for n_1, n_2 in G.edges_iter():
        a.addEdge(new_node_mapping[n_1], new_node_mapping[n_2])

    return a


def nodes_to_NX_graph(nodes):
    """
    Takes an iterable of nodes and creates a networkx graph. The nodes must
    be part of a skeleton annotation object. This is useful to get the path
    length of a subset of a skeleton annotation, defined by nodes.
    :param nodes: iterable of SkeletonNode objects
    :return: networkx graph
    """
    nx_G = nx.Graph()

    for n in nodes:
        nx_G.add_node(n)
        for child in n.getChildren():
            nx_G.add_edge(n, child, weight=n.distance_scaled(child))

    return nx_G

def annoToNXGraph(annotations, merge_annotations_to_single_graph=False):
    """

    Creates a weighted networkx graph from an annotation.
    Weights are the euclidian distance between nodes. Node objects
    are preserved. The annotation.scaling parameter is required.

    """
    graphs = []
    if not type(annotations) == list:
        annotations = [annotations]

    if merge_annotations_to_single_graph:
        nxG = nx.Graph()
        all_nodes = []
        for anno in annotations:
            all_nodes.extend(anno.getNodes())
        for node in all_nodes:
            nxG.add_node(node)
            for child in node.getChildren():
                try:
                    nxG.add_edge(node, child,
                                 weight=node.distance_scaled(child))
                except:
                    print 'Phantom child node, annotation' \
                          'object inconsistent'
        graphs = nxG

        # ugly code duplication here, don't look at it! ;)

    else:
        for anno in annotations:
            nxG = nx.Graph()
            nodes = anno.getNodes()
            for node in nodes:
                nxG.add_node(node)
                for child in node.getChildren():
                    try:
                        nxG.add_edge(node, child,
                                     weight=node.distance_scaled(child))
                    except:
                        print 'Phantom child node, annotation' \
                              'object inconsistent'
            graphs.append(nxG)

    return (graphs)

def shortestPathBetNodes(annotation, node1, node2):
    """Shortest path length between 2 nodes in units of scaling / 1000"""
    graph = annoToNXGraph(annotation)
    return nx.dijkstra_path_length(graph, node1, node2)/1000.
    
def genZColumnGridSkeleton():
    """
    
    Generates a nml file with a skeleton grid, based on dense columns
    in z direction to help further annotations. Each column is
    an individual tree, labeled with a comment. This allows the annotator
    to hide columns that are already done.
    
    Configure variables directly in the code, at the top.
    
    """
    
    skeleton = NewSkeleton()
    
    # configuration variables follow

    nml_path = '/mnt/hgfs/E/column_grid.nml'
    
    # in voxels
    spacing_x = 100
    spacing_y = 100
    spacing_z = 20
    
    skeleton.experimentName = 'j0251'
    scaling = (10, 10, 25)
    
    cube_min = (13250, 13550, 7647) # min coord of grid in bounding box
    cube_mini_boxes_per_dim = (5, 5, 10)
    
    node_radius = 0.5

    # configuration end

    skeleton.scaling = (scaling[0], scaling[1], scaling[2])
    node_id = 1

    for x in range(cube_min[0], cube_min[0] +
            cube_mini_boxes_per_dim[0] * spacing_x, spacing_x):
        for y in range(cube_min[1], cube_min[1] +
            cube_mini_boxes_per_dim[1] * spacing_x, spacing_x):

            # create a new tree for each column to help the annotators

            cur_anno = SkeletonAnnotation()
            cur_anno.scaling = (scaling[0], scaling[1], scaling[2])
          
            for z in range(cube_min[2], cube_min[2] +
                    cube_mini_boxes_per_dim[2] * spacing_z, spacing_z):
                                
                # gen nodes
                node1 = SkeletonNode()
                node2 = SkeletonNode()                
                node3 = SkeletonNode()                 
                node4 = SkeletonNode()                
                
                # link with annotation                
                node1.from_scratch(cur_anno, x, y, z,
                                   1, 1, 0, node_id, node_radius)
                node_id += 1
                node2.from_scratch(cur_anno, x+spacing_x, y, z,
                                   1, 1, 0, node_id, node_radius)
                node_id += 1                                   
                node3.from_scratch(cur_anno, x+spacing_x, y+spacing_y, z,
                                   1, 1, 0, node_id, node_radius)   
                node_id += 1
                node4.from_scratch(cur_anno, x, y+spacing_y, z,
                                   1, 1, 0, node_id, node_radius)                
                node_id += 1
                
                cur_anno.addNode(node1) 
                cur_anno.addNode(node2)        
                cur_anno.addNode(node3) 
                cur_anno.addNode(node4) 
                
                # connect nodes
                cur_anno.addEdge(node1, node2) 
                cur_anno.addEdge(node2, node3) 
                cur_anno.addEdge(node3, node4) 
                cur_anno.addEdge(node4, node1)
            
            skeleton.annotations.add(cur_anno)
            
    skeleton.toNml(nml_path)
    
    return

def setAnnotationStats(annos):
    """Sets the following stats:
    anno.pathLen in um
    anno.numBranchNodes
    """
    newAnnos = []
    for anno in annos:
        nxG = annoToNXGraph(anno)
        anno.pathLen = nxG[0].size(weight='weight') / 1000
        anno.numBranchNodes = len(list({k for k, v in nxG[0].degree().iteritems() if v > 2}))
        anno.branchDensity = anno.numBranchNodes / anno.pathLen
        # avgBranchLen
        #anno.avgBranchLen = 
        newAnnos.append(anno)
    
    return newAnnos

def genj0126SkelObj():
    skel = NewSkeleton()
    skel.scaling = (10,10,20)
    skel.experimentName = 'j0126'
    
    return skel

def annotationCanonizer(annos):
    """Returns the anno with the most nodes for annos
    that were identified to be the same and all other
    unique annotations."""
    canonized = []
    grouped = annotationMatcher(annos, 5000, samplesize=50, thres=0.5)
    for group in grouped:
        if type(group) == list:
            group.sort(key=lambda x: x.pathLen, reverse=True)
            canonized.append(group[0]) # use the largest anno for now
        else:
            canonized.append(group)
    return canonized
    
def annosToNMLFile(annos, filename):
    skel = ns.NewSkeleton()
    skel.scaling = (9,9,20)
    skel.experimentName = 'j0126'
    
    currBaseID = 1
    for anno in annos:
        # find highest node ID in this annotation
        ids = [node.getID() for node in anno.getNodes()]
        ids.sort()
        currBaseID += (ids[-1] + 1)
        anno.setNodeBaseID(currBaseID)
        skel.annotations.add(anno)
        print currBaseID
    
    skel.toNml(filename)
    return

class EnhancedAnnotation():
    """
    Representation of an annotation that additionally includes a NetworkX
    graph representation and KD-Tree (from scipy spatial module)
    representation of the same annotation.
    """

    def __init__(self, annotation, interpolate=False):
        """
        Generate the KD trees for spatial lookups and NetworkX graph.

        Parameters
        ----------

        annotation : SkeletonAnnotation object

        interpolate : bool or positive float
            Control how self.kd_tree_spatial will interpolate nodes in the
            annotation.
            If False, use only the already existing nodes. If True, interpolate
            with a resolution of approximately 1 voxel.
            If positive float, interpolate with approximately that resolution.
        """

        # todo support interpolate again
        #
        #self.kd_tree_spatial = annoToKDtree(annotation,
        #        interpolate=interpolate)[0]
        self.kd_tree_spatial = annoToKDtree(annotation)[0]
        self.kd_tree_nodes = annoToKDtree(annotation)[0]
        self.graph = annoToNXGraph(annotation)[0]
        self.annotation = annotation

        for cur_n in self.annotation.getNodes():
            coord = cur_n.getCoordinate()
            cur_n.annotation = self

    def __getattr__(self, name):
        # Hack to fake inheritance-like behavior from SkeletonAnnotation.
        # Actually inheriting would be complicated because we would need
        # an extra conversion function from SkeletonAnnotation to
        # EnhancedAnnotation.
        return getattr(self.annotation, name)

    def get_shortest_path(self, from_node, to_node):
        """
        Return EnhancedAnnotation object representing the segment defined
        by the shortest path between from_node and to_node.

        Parameters
        ----------
        
        from_node : SkeletonNode object
            Defines the beginning of the path
        to_node : SkeletonNode object
            Defines the end of the path
        """

        # Construct new SkeletonAnnotation object containing only the
        # shortest path between from_node and to_node
        #

        new_a = SkeletonAnnotation()
        path = nx.dijkstra_path(self.graph, from_node, to_node)
        new_a.add_path(path)

        new_ea = EnhancedAnnotation(new_a)

        return new_ea

    def __contains__(self, node):
        """
        Return True if the annotation contains a node at the position of node,
        False otherwise.
        """

        n = get_closest_node(node, self)

        dst = euclidian_distance(n.getCoordinate_scaled(),
                node.getCoordinate_scaled())

        if dst < 1.0:
            return True
        else:
            return False

def get_closest_node(location, annotation, cutoff=None):
    """
    Return the node in annotation that is closest to location.

    Parameters
    ----------

    location : SkeletonNode object or list of numeric type
        If SkeletonNode object, node for which to find closest match in
        annotation.
        If list of numeric type, coordinate for which to find closest
        match in annotation.

    annotation : EnhancedAnnotation or SkeletonAnnotation object.
        Annotation in which to search for matching nodes. Spatial lookup
        will be faster when using EnhancedAnnotation.

    cutoff : float or None
        If float, return closest node only if its distance to location is
        smaller than cutoff. If None, ignore.

    Returns
    -------

    closest_node : SkeletonNode object or None
        Closest node to location in annotation or None if there is no node in
        annotation or closest node in annotation is futher away from location
        than cutoff.
    """

    try:
        location = location.getCoordinate_scaled()
    except AttributeError:
        pass

    try:
        closest_node = annotation.kd_tree_nodes.query_nearest_node(location)[0]
    except AttributeError:
        distances = []
        all_nodes = list(annotation.getNodes())
        for cur_n in all_nodes:
            cur_dist = euclidian_distance(location,
                    cur_n.getCoordinate_scaled())
            distances.append(cur_dist)
        try:
            closest_node = all_nodes[distances.index(min(distances))]
        except ValueError:
            # Argument to min is empty
            closest_node = None
    except IndexError:
        closest_node = None

    if cutoff is not None:
        if euclidian_distance(location,
            closest_node.getCoordinate_scaled()) > cutoff:
                closest_node = None

    return closest_node

class ShortestPathSegment():
    """
    Class representing a shortest path between two nodes in an annotation.
    """

    def __init__(self):
        self.annotation = None
        self.from_node = None
        self.to_node = None
        self.path = []

    def __repr__(self):
        return str([str(x) for x in self])
        
    def from_annotation(self, annotation, from_node, to_node):
        """
        Parameters
        ----------

        annotation : EnhancedAnnotation object
            Annotation in which the path is defined

        from_node, to_node : SkeletonNode object
            First and last node in shortest path, respectively.
        """

        if isinstance(annotation, SkeletonAnnotation):
            annotation = EnhancedAnnotation(annotation)

        self.annotation = annotation
        self.from_node = from_node
        self.to_node = to_node

        self.path = nx.dijkstra_path(annotation.graph, from_node, to_node)

    def from_path(self, path, annotation=None):
        self.annotation = annotation
        self.path = path

        try:
            self.from_node = path[0]
            self.to_node = path[len(path) - 1]
        except:
            pass

    def __len__(self):
        return len(self.path)

    def __iter__(self):
        for cur_node in self.path:
            yield cur_node

    def iter_edges(self):
        # zip() truncates based on the shorter list
        for from_node, to_node in zip(self.path, self.path[1:]):
            yield (from_node, to_node)

    def get_node_at_distance(self, dst, exact=False):
        """
        Return node at a specified distant from segment start.

        Parameters
        ----------

        dst : positive float
            Minimum distance, in physical units

        exact : bool
            If True, generate and return a new node that is at exactly the
            distance dst (rounded to integer coordinates). The new node will
            also be added to the ShortestPathSegment.

        Returns
        -------

        distant_node : SkeletonNode object
            Node at minimum distance dst from segment start node. If no node
            is sufficiently distant, return most distant node. This will happen
            even if exact=True is set!

        total_distance : float
            Distance of distant_node from segment start.
        """

        total_distance = 0.
        distant_node = None

        if len(self) == 1:
            return (self.from_node, 0.0)

        for node_1, node_2 in self.iter_edges():
            edge_len = euclidian_distance(node_1.getCoordinate_scaled(),
                    node_2.getCoordinate_scaled())
            if total_distance + edge_len >= dst:
                if exact:
                    delta = FloatCoordinate(node_2.getCoordinate_scaled()) - \
                        FloatCoordinate(node_1.getCoordinate_scaled())
                    edge_len_fraction = (dst - total_distance) / edge_len
                    assert(edge_len_fraction <= 1.)
                    exact_pos = FloatCoordinate(node_1.getCoordinate_scaled()) \
                            + delta * edge_len_fraction
                    exact_pos_dataset_scale = exact_pos / \
                            node_1.annotation.scaling
                    exact_pos_dataset_scale = [round(x) for x in \
                            exact_pos_dataset_scale]
                    distant_node = copy.copy(node_1)
                    distant_node.setCoordinate(exact_pos_dataset_scale)
                    insert_idx = self.path.index(node_2)
                    self.path.insert(insert_idx, distant_node)
                else:
                    distant_node = node_2

                break

            total_distance += edge_len

        if distant_node is None:
            distant_node = node_2

        total_distance = euclidian_distance(
                self.from_node.getCoordinate_scaled(),
                distant_node.getCoordinate_scaled())

        return (distant_node, total_distance)

    def length(self):
        """
        Return total length of segment over all nodes in physical units.
        """

        total_length = 0.

        for node_1, node_2 in self.iter_edges():
            total_length += euclidian_distance(
                    node_1.getCoordinate_scaled(),
                    node_2.getCoordinate_scaled())

        return total_length

    def get_subsegment(self, start_node, end_node):
        """
        Return the subsegment of this segment that lies between two nodes.

        Parameters
        ----------

        start_node : SkeletonNode object
            Node that starts the subsegment. Must come before end_node in the
            order of the segment.

        end_node : SkeletonNode object
            Node that ends the subsegment. Must come after start_node in the
            order of the segment.

        Returns
        -------

        subsegment : ShortestPathSegment instance
            Nodes from start_node to end_node, including them, in order.
            Empty if there is no subsegment limited by the specified
            nodes.

        """

        subsegment = ShortestPathSegment()
        subsegment.from_path(
            self.path[self.path.index(start_node):self.path.index(end_node)+1])

        return subsegment

def remove_all_node_comments(anno):
    all_nodes = list(anno.getNodes())
    for n in all_nodes:
        n.setPureComment('')

    return anno

def get_subsegment_by_distance(annotation, start_node, end_node,
        near_dst, far_dst):
    """
    Return ShortestPathSegment instance corresponding to a subsegment
    delimited by physical distances of a subsegment delimited by start
    and end nodes.

    Parameters
    ----------

    annotation : EnhancedAnnotation instance

    start_node : SkeletonNode instance
        Start node, must be contained in annotation.

    end_node : SkeletonNode instance
        End node, must be contained in annotaton.

    near_dst : positive float
        Distance of subsegment start seen from start_node, towards end_node.

    far_dst : positive float
        Distance of subsegment end seen from start_node, towards end_node.
        Must be larger than near_dst.

    Raises
    ------

    Exception
        If near_dst is larger than far_dst

    Returns
    -------

    subsegment : ShortestPathSegment instance
    """

    if far_dst < near_dst:
        raise Exception('far_dst must be larger or equal to near_dst.')

    segment = ShortestPathSegment()
    segment.from_annotation(annotation, start_node, end_node)

    segment_start_node = segment.get_node_at_distance(near_dst, exact=True)[0]
    segment_end_node = segment.get_node_at_distance(far_dst, exact=True)[0]

    subsegment = segment.get_subsegment(segment_start_node, segment_end_node)

    return subsegment

# useful console snippets:
# annos.sort(key=lambda x: x.numBranchNodes, reverse=True)


