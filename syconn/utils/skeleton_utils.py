import copy
import numpy as np
import os
import random
import re
import zipfile
from collections import Counter
from glob import glob
from scipy import spatial
from scipy.spatial import ConvexHull

import networkx as nx

from syconn.utils.basics import euclidian_distance, FloatCoordinate
from syconn.utils.skeleton import SkeletonAnnotation, Skeleton, SkeletonNode
from syconn.utils.skeleton import integer_checksum



class InvalidFileFormatException(Exception):
    pass


class NonSimpleNML(Exception):
    pass


def neighbour_next(node):
    return node.getNeighbors()


def return_process(node):
    return node


def iter_nodes_dfs(annotation, root):
    """
    Iterate over the nodes in annotation, starting at root, using depth-first
    search.

    Parameters
    ----------

    annotation : SkeletonAnnotation object

    root : SkeletonNode object
        Must be contained in annotation.
    """

    anno_search = AnnotationSearch(annotation, root, None, None)
    anno_dfs = anno_search.search(
        anno_search.dfs_search, neighbour_next, return_process)

    for from_node, cur_node in anno_dfs:
        yield cur_node


class AnnotationSearch(object):
    def __init__(self, annotation, node, f, context):
        self.annotation = annotation
        self.node = node
        self.f = f
        self.context = context
        return

    #
    def search(self, search_f, next_f, process_f):
        to_visit = [(None, self.node)]
        visited = set()

        while len(to_visit) > 0:
            (from_node, cur_node) = to_visit.pop()
            visited.add(cur_node)
            yield process_f((from_node, cur_node))
            next_nodes = next_f(cur_node)
            next_nodes.difference_update(visited)
            search_f(cur_node, to_visit, next_nodes)

    #
    def dfs_search(self, from_node, to_visit, next_nodes):
        for i in next_nodes:
            to_visit.append((from_node, i))
        return

    #
    def bfs_search(self, from_node, to_visit, next_nodes):
        for i in next_nodes:
            to_visit.insert(0, (from_node, i))
        return

    #
    def BFS(self, descend_f):
        unvisited = self.annotation.getNodes().copy()
        cur_to_visit = set([(self.node, None)])
        level = 1
        count = 0
        while len(cur_to_visit) > 0:
            next_to_visit = set()
            for (cur_node, origin_node) in cur_to_visit:
                if cur_node in unvisited:
                    unvisited.remove(cur_node)
                    count += 1
                    self.f(cur_node, origin_node, level, count, self.context)
                    for descend_node in descend_f(cur_node):
                        next_to_visit.add((descend_node, cur_node))
            cur_to_visit = next_to_visit
            level += 1
        return unvisited

    #
    def annnotationNeighborsBFS(self):
        def neighborsDescend(node):
            return node.getNeighbors()

        return self.BFS(neighborsDescend)

    #
    def annnotationChildrenBFS(self):
        def childrenDescend(node):
            return node.getChildren()

        return self.BFS(childrenDescend)

    #
    def annnotationParentsBFS(self):
        def parentsDescend(node):
            return node.getChildren()

        return self.BFS(parentsDescend)

    #
    pass


def filter_nodes(s, criterion):
    matching_nodes = []

    for cur_node in s.getNodes():
        if criterion(cur_node):
            matching_nodes.append(cur_node)

    return matching_nodes


def is_simple_skeleton(s, fail_on_empty=False):
    """
    Returns the annotation if the skeleton contains exactly
    one non-empty annotation. If fail_on_empty is True, return
    the annotation only if there is exactly one non-empty
    annotation and no empty annotations. Return None otherwise.
    """

    if isinstance(s, str):
        skeleton = Skeleton()
        s = skeleton.fromNml(s)

    nonempty_annotation_count = 0
    annotation_count = len(s.getAnnotations())
    for i in s.getAnnotations():
        if len(i.getNodes()) > 0:
            annotation = i
            nonempty_annotation_count += 1

    if fail_on_empty:
        if nonempty_annotation_count == 1 and annotation_count == 1:
            return annotation
        else:
            return None
    else:
        if nonempty_annotation_count == 1:
            return annotation
        else:
            return None


def has_3_4_worktime_bug(filename):
    """
    Knossos 3.4 introduced obfuscated (xor'ed) worktimes. When skeletons
    were created in versions prior to 3.4 and went through multiple
    load-save cycles in 3.4, the work time would alternatingly be
    xor'ed and not xor'ed and the node timestamps became unreliable.

    This works on the assumptions that times are not larger than ca.
    15 days, so we can use the high bits of the xor key to detect
    whether a time has been xor'ed or not.

    This function returns True if the work time in the file can not be
    trusted.

    filename is the path to the nml file.
    """

    def probably_obfuscated(time):
        xor_detection_threshold = 1300000000

        if time >= xor_detection_threshold:
            return True
        else:
            return False

    # Decide quickly whether the file could be affected

    re_3_4_saved = 'lastsavedin version=\"3\.4\"'
    re_3_4_created = 'createdin version=\"3\.4\"'

    with open(filename, 'r') as f:
        header = f.read(2048)

        if re.search(re_3_4_saved, header) and \
                not re.search(re_3_4_created, header):
            # Might be affected
            pass
        else:
            return False

    s = Skeleton()
    s.fromNml(filename)

    # If the times in the header are not obfuscated, we know there
    # is a problem

    if not probably_obfuscated(s.skeleton_time) or \
            not probably_obfuscated(s.skeleton_idletime):
        return True

    # ... but if they are obfuscated, we can't be sure there's no
    # problem. We have to check all the node timestamps to make sure
    # they are consistent. Time stamps should never be obfuscated,
    # so if they are, we know the bug has been triggered.

    for cur_node in s.getNodes():
        if probably_obfuscated(cur_node.getDataElem('time')):
            return True

    return False


def get_nodes_with_comment(s, expression, flags=0):
    """
    Return a list of SkeletonNode objects with comments that match expression.

    Parameters
    ----------

    s : NewSkeleton instance or SkeletonAnnotation instance.

    expression : str

    flags : int
        Regular expression flags for re.search()
    """

    matching_nodes = []

    for cur_node in s.getNodes():
        if re.search(expression, cur_node.getPureComment(), flags):
            matching_nodes.append(cur_node)

    return matching_nodes


def get_nodes_with_token(s, token):
    """
    Return a list of SkeletonNode objects in s with comments containing token.
    A token is defined as a substring of the node comment separated by a
    semicolon and optional whitespace.

    Parameters
    ----------

    s: NewSkeleton instance or SkeletonAnnotation instance.

    token : str

    Return
    ------

    matching_nodes : list of SkeletonNode instances
    """

    matching_nodes = []

    for cur_node in s.getNodes():
        if cur_node.has_comment_token(token):
            matching_nodes.append(cur_node)

    return matching_nodes


def get_annotations_with_comment(s, expression, flags=0):
    """
    Returns a list of SkeletonAnnotation objects with comments that match
    expression.

    Parameters
    ----------

    s : NewSkeleton instance

    expression : str

    flags : int
        Regular expression flags for re.search()
    """

    matching_annos = []

    for cur_anno in s.getAnnotations():
        if re.search(expression, cur_anno.getComment(), flags):
            matching_annos.append(cur_anno)

    return matching_annos


def get_nonempty_annotations(s):
    nonempty = []

    for cur_anno in s.getAnnotations():
        if len(cur_anno.getNodes()) > 0:
            nonempty.append(cur_anno)

    return nonempty


def is_singly_connected(annotation):
    """
    Return True if all nodes in annotation are connected, False otherwise
    (including if there are no nodes).

    Parameters
    ----------

    annotation : SkeletonAnnotation object
    """

    nodes = list(annotation.getNodes())
    if len(nodes) == 0:
        return False

    dfs_nodes = list(iter_nodes_dfs(annotation, nodes[0]))

    if len(dfs_nodes) == len(nodes):
        return True
    else:
        return False


def get_the_nonempty_annotation(s):
    """
    Convenience function that returns the one nonempty annotation from a
    NewSkeleton instance s and raises an Exception if more than one exist in
    s.

    Parameters
    ----------

    s : NewSkeleton instance
    """

    annos = get_nonempty_annotations(s)

    if len(annos) != 1:
        raise NonSimpleNML('Skeleton must contain exactly one non-empty '
                        'annotation.')

    return annos[0]


def get_the_nonempty_connected_annotation(s):
    """
    Convenience function that returns the one nonempty, singly connected
    annotation from NewSkeleton instance s and raises Exception if more than
    one non-empty annotation exists or if that one annotation is not singly
    connected.

    Parameters
    ----------

    s : NewSkeleton instance
    """

    anno = get_the_nonempty_annotation(s)
    if not is_singly_connected(anno):
        raise NonSimpleNML('Skeleton must contain exactly one non-empty, '
                          'singly connected annotation.')

    return anno


def reset_nml_for_heidelbrain_taskfile(f, time=0, current_knossos='3.4.2'):
    """
    Takes an NML and applies some corrections to make it suitable as a regular
    Heidelbrain task file. That is, timestamps are removed or reset
    and the version strings are set to a recent version of knossos. The file
    at f is altered by this function.

    Parameters
    ----------

    f : str
        Path to file

    time : int
        What to set the time to, in milliseconds. This can be used to add
        gratuitous time to a heidelbrain task.

    current_knossos : str
        Knossos version to set createdin and lastsavedin to.
    """

    idle_time = 0

    time_checksum = integer_checksum(time)
    idle_time_checksum = integer_checksum(idle_time)

    with open(f, 'r') as fp:
        cur_text = fp.read()
        cur_text = re.sub('<idleTime.*>', '', cur_text)
        cur_text = re.sub('<time.*>', '', cur_text)
        cur_text = re.sub('time=\"[0-9]*\"', 'time=\"0\"', cur_text)
        cur_text = re.sub('<createdin.*>', '', cur_text)
        cur_text = re.sub('<lastsavedin.*>', '', cur_text)
        cur_text = re.sub('\n\s*\n', '\n', cur_text)
        cur_text = re.sub(
            '</parameters>',
            '\t<time checksum=\"%s\" ms=\"%d\"/>\n'
            '\t<idleTime checksum=\"%s\" ms=\"%d\"/>\n'
            '\t<createdin version=\"%s\"/>\n'
            '\t<lastsavedin version=\"%s\"/>\n'
            '\t</parameters>' % (
                time_checksum, time, idle_time_checksum, idle_time,
                current_knossos, current_knossos,),
            cur_text)

    with open(f, 'w') as fp:
        fp.write(cur_text)


def annotation_from_nodes(nodes, annotation=None, preserve_scaling=True,
                          connect=False):
    """
    Take an iterable of nodes and return an annotation that contains those
    nodes. If the annotation key word parameter is specified, return an
    annotation in which the nodes are connected as in annotation. Otherwise,
    the nodes will not be connected.

    Parameters
    ----------

    nodes : Iterable of SkeletonNode objects

    annotation : SkeletonAnnotation instance or NoneType

    preserve_scaling : bool
        Whether to copy the scaling from annotation to the new annotation.

    connect : bool
        Whether to copy the connectivity from annotation to the new annotation.

    Returns
    -------

    new_anno : SkeletonAnnotation instance
    """

    new_anno = SkeletonAnnotation()

    for cur_n in nodes:
        new_anno.addNode(cur_n)

    if annotation is not None and connect is True:
        for cur_n in nodes:
            try:
                new_anno.edges[cur_n] = annotation.edges[cur_n].copy()
                new_anno.reverse_edges[cur_n] = \
                    annotation.reverse_edges[cur_n].copy()
            except KeyError:
                pass

    if annotation is not None and preserve_scaling is True:
        new_anno.scaling = annotation.scaling

    return new_anno


def split_by_connected_component(annotation):
    """
    Take an annotation and return a list of new annotations that represent
    the connected components in the original annotation.

    Parameters
    ----------

    annotation : SkeletonAnnotation instance

    Returns
    -------

    connected_components : set of SkeletonAnnotation instances
    """

    seen_nodes = set()
    connected_components = set()

    annotation = copy.copy(annotation)

    for cur_n in annotation.getNodes():
        if not cur_n in seen_nodes:
            cur_component = list(iter_nodes_dfs(annotation, cur_n))
            seen_nodes.update(cur_component)
            connected_components.add(
                annotation_from_nodes(
                    cur_component, annotation=annotation, connect=True))

    return connected_components


def get_reachable_nodes(node):
    """
    Return a set of all nodes that are reachable from node (part of the
    connected component that includes node).

    Parameters
    ----------

    node : SkeletonNode instance

    Returns
    -------

    reachable_nodes : set of SkeletonNode instances
    """

    reachable_nodes = set(iter_nodes_dfs(node.annotation, node))

    return reachable_nodes


def merge_annotations(a1, a2):
    """
    Take all the nodes and edges from annotation a2 and add them to
    annotation a1. Annotation a2 remains intact after calling this function.

    Parameters
    ----------

    a1, a2 : SkeletonAnnotation instances

    Returns
    -------

    a1 : SkeletonAnnotation instances
        Same annotation that was passed as a parameter, modified "in place".

    As an additional side effect, this function sets the old_to_new_nodes
    attribute on a1, which is a dict mapping nodes from a2 to nodes in a1
    after the merge.
    """

    a2 = copy.copy(a2)

    for cur_n in a2.getNodes():
        a1.addNode(cur_n)

    a1.reverse_edges.update(a2.reverse_edges)
    a1.edges.update(a2.edges)

    # This is set by __copy__() to allow finding back nodes from the a2
    # annotation in the new merged annotation.
    a1.old_to_new_nodes = a2.old_to_new_nodes

    return a1


def get_largest_annotation(skeleton):
    """
    Return the physically largest annotation from skeleton.

    Parameters
    ----------

    skeleton : NewSkeleton instance

    Returns
    -------

    largest_annotation : SkeletonAnnotation instance or None
        Largest annotation or None if there is no annotation in skeleton.

    largest_annotation_length : float
        Physical length of largest_annotation

    """

    largest_annotation = None
    largest_annotation_length = 0.

    for cur_a in skeleton.getAnnotations():
        if cur_a.physical_length() > largest_annotation_length:
            largest_annotation = cur_a
            largest_annotation_length = cur_a.physical_length()

    return largest_annotation, largest_annotation_length


def extract_main_tracings(source_folder, scaling):
    """
    Go through a set of NML files, extract the largest annotation and save
    only that annotation to a new file.

    Parameters
    ----------

    source_folder : str or list of str
        If str, glob expression yielding the list of NML paths to look at.
        If list of str, the list of NML paths to look at.

    Returns
    -------

    Nothing, writes NML files with the same name as before, except the .nml
    extension is replaced by .main_tracing.nml.
    """

    if isinstance(source_folder, str):
        source_folder = glob.glob(source_folder)

    for cur_f in source_folder:
        print('Processing %s' % (cur_f,))

        s = Skeleton()
        s.fromNml(cur_f, scaling=scaling)
        main_tracing, main_tracing_len = get_largest_annotation(s)

        s = Skeleton()
        s.add_annotation(main_tracing)
        s.toNml(cur_f.replace('.nml', '.main_tracing.nml'))


def get_node_at_position(annotation, position):
    """
    Return a node from annotation located at a given position.

    Parameters
    ----------

    annotation : SkeletonAnnotation instance

    position : Iterable of int with length 3

    Returns
    -------

    node : SkeletonNode instance or None
        Node at position, or None if there is no node at that position.
    """

    node = None

    for cur_n in annotation.getNodes():
        if euclidian_distance(cur_n.getCoordinate(), position) == 0.:
            node = cur_n
            break

    return node


def save_annotations(annotations, fname, scaling=None):
    """
    Save annotations to an NML.

    Parameters
    ----------

    annotations : Iterable of SkeletonAnnotation instances

    fname : str
        Filename to save to

    scaling : Iterable of float or None
        Scaling parameter to set on the NML file.

    Returns
    -------

    Nothing, writes file.
    """

    s = Skeleton()
    s.set_scaling(scaling)
    for cur_a in annotations:
        s.add_annotation(cur_a)
    s.toNml(fname)


def get_nml_str_from_knossos_file(fname):
    """
    Take a knossos file and return the NML contents as str, i.e. open the
    file and read it or extract and read the annotation.xml from a kzip.
    """

    if fname.lower().endswith('.nml'):
        with open(fname, 'r') as fp:
            txt = fp.read()
    elif fname.lower().endswith('.k.zip'):
        zipper = zipfile.ZipFile(fname)

        if not 'annotation.xml' in zipper.namelist():
            raise InvalidFileFormatException(
                'kzip file does not contain annotation.xml')

        txt = zipper.read('annotation.xml')
    else:
        raise InvalidFileFormatException('Expect NML or kzip.')

    return txt


def iter_nodes_regex(fname):
    """
    Iter over nodes in a NML / kzip using regular expressions. Much faster
    than XML parsing.

    Parameters
    ----------

    fname : str
        Path to nml or kzip
    """

    node_id_ex = r'<node id="(?P<node_id>\d+)"'

    txt = get_nml_str_from_knossos_file(fname)

    for cur_match in re.finditer(node_id_ex, txt):
        yield cur_match


def get_max_node_id(fname):
    """
    Return maximal node ID.
    """

    all_node_ids = []
    for cur_match in iter_nodes_regex(fname):
        cur_node_id = int(cur_match.groupdict()['node_id'])
        all_node_ids.append(cur_node_id)

    return max(all_node_ids)


def has_node_id_overflow_problem(fname, cutoff=2**31-1):
    """
    Due to a knossos issue, node IDs can become excessively high.
    This checks whether a file is affected.

    Parameters
    ----------

    fname : str
        Path to nml or kzip

    cutoff : int
        Start warning from this node ID

    Returns
    -------

    problem : boolean
        True if there is a problem
    """

    for cur_match in iter_nodes_regex(fname):
        cur_node_id = cur_match.groupdict()['node_id']
        if int(cur_node_id) > cutoff:
            return True


    return False


def has_node_id_overflow_problem_dir(path, expression='*', cutoff=2**31-1):
    """
    Like has_node_id_overflow_problem, but works on a directory.

    Parameters
    ----------

    path : str
        Path to directory of NMLs and / or .k.zip

    Returns
    -------

    problem : list of str
        List of paths to problematic files

    no_problem : list of str
        List of paths to intact files
    """

    all_nml = glob(path + '/%s.nml' % (expression, ))
    all_kzip = glob(path + '/%s.k.zip' % (expression, ))

    problem = []
    no_problem = []

    for cur_f in all_nml + all_kzip:
        if has_node_id_overflow_problem(cur_f, cutoff):
            problem.append(cur_f)
        else:
            no_problem.append(cur_f)

    return problem, no_problem


def get_node_positions_as_nparray(s, scaling='raw'):
    """
    Return the positions of all nodes in a skeleton as a numpy array.

    Parameters
    ----------

    s : NewSkeleton instance

    scaling : str
        if 'raw', use the raw node coordinates, otherwise, use scaled
        coordinates.
    """

    nodes = s.getNodes()

    nodes_np = np.zeros((len(nodes), 3))

    if scaling=='raw':
        pos_fn = SkeletonNode.getCoordinate
    else:
        pos_fn = SkeletonNode.getCoordinate_scaled

    for i, cur_n in enumerate(nodes):
        cur_pos = pos_fn(cur_n)
        nodes_np[i, 0] = cur_pos[0]
        nodes_np[i, 1] = cur_pos[1]
        nodes_np[i, 2] = cur_pos[2]

    return nodes_np


def get_convex_hull(s, scaling='nm'):
    """
    Return scipy's convex hull of the set of points contained in a skeleton.

    Parameters
    ----------

    s : NewSkeleton instance or str
        If str, path to a file to load skeleton from

    scaling : str
        As in get_node_positions_as_nparray

    Returns
    -------

    hull : scipy.spatial.ConvexHull instance
    """

    if isinstance(s, str):
        skel = Skeleton()
        skel.fromNmlcTree(s)
        s = skel

    nodes_np = get_node_positions_as_nparray(s, scaling)

    hull = ConvexHull(nodes_np)

    return hull


def get_end_nodes(annotation):
    """
    Return set of SkeletonNode objects in annotation that only have one
    neighbor, i.e. that are ends.

    Parameters
    ----------

    annotation : EnhancedAnnotation object
        Annotation in which to search for end nodes
    """

    return set([k for k, v in annotation.graph.degree().iteritems() if v == 1])


def prune_short_end_branches(anno, length, debug_labels=False):
    """
    Remove all end branches that are shorter than length.

    Parameters
    ----------

    anno : SkeletonAnnotation instance

    length : float
        In physical units, up to which length to prune

    debug_labels : boolean
        Whether to add comments to the pruned skeleton for debugging
        purposes. Will label identified end nodes and upstream nodes.
    """

    anno_en = EnhancedAnnotation(anno)
    to_remove = []

    for cur_end_node in get_end_nodes(anno_en):
        cur_end_node.setComment('END')
        upstream_node = None

        for cur_next_node in iter_nodes_dfs(anno_en, cur_end_node):
            if cur_next_node.degree() > 2:
                upstream_node = cur_next_node
                upstream_node.setComment('UPSTREAM')

            if upstream_node is not None:
                sp = ShortestPathSegment()
                sp.from_annotation(anno_en, cur_end_node, upstream_node)

                if sp.length() < length:
                    for cur_node in sp.path:
                        if cur_node == upstream_node:
                            break
                        cur_node.setComment('DEL')
                        to_remove.append(cur_node)
                break

    for cur_n in to_remove:
        anno_en.removeNode(cur_n)


def skeleton_from_single_coordinate(
        coordinate, comment=None, branchpoint=False):
    """
    Return a NewSkeleton object containing exactly one annotation with
    exactly one node at position coordinate.
    This is good for generating seed NML files for specific coordinates.

    Parameters
    ----------

    coordinate : Iterable of int

    comment : str

    branchpoint : bool
        Whether this node is a branchpoint.

    Returns
    -------

    s : NewSkeleton instance
    """

    s = Skeleton()
    anno = SkeletonAnnotation()
    node = SkeletonNode()

    x, y, z = coordinate[0:3]

    node.from_scratch(anno, x, y, z)
    if comment is not None:
        node.setPureComment(comment)

    anno.addNode(node)
    s.add_annotation(anno)
    s.set_edit_position([x, y, z])
    s.active_node = node

    if branchpoint:
        s.branchNodes.append(node)

    return s


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
    annosSortedByNodeNums = zip([len(anno.getNodes()) for anno in annos],
                                list(annos))
    annosSortedByNodeNums.sort()
    mostProbableAnno = list(annosSortedByNodeNums.pop(-1))[1]
    if len(mostProbableAnno.getNodes()) == 0:
        mostProbableAnno = None
    return mostProbableAnno


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

            raise Exception('File ' + nmlfile +
                            ' contains more than one connected component.')

        annotations.append(anno)
        annodict[anno.seedID] = anno
    return annotations, annodict


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
    skeletonObj = Skeleton()
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
                merge_annotations(skeletonObj.annotations[0], anno_to_merge)
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
    skel = Skeleton()
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

    if write_match_group_nmls:
        print('Done with grouping, starting to write out match group nmls')
        for cnt, group in enumerate(groups):
            skel_obj = Skeleton()

            # colors = skelplot.get_different_rgba_colors(len(group),rgb_only=True)

            for cnt2, anno in enumerate(group):
                anno.setComment(anno.username + ' ' + anno.seedID)
                # anno.color = colors[cnt2]
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

    s = Skeleton()
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

    skeleton = Skeleton()

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
    skel = Skeleton()
    skel.scaling = (10,10,20)
    skel.experimentName = 'j0126'

    return skel


def annosToNMLFile(annos, filename):
    skel = Skeleton()
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