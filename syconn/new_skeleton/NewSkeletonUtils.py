import copy
import numpy as np
import re
import zipfile
from glob import glob
from scipy.spatial import ConvexHull

from syconn.new_skeleton.newskeleton import SkeletonAnnotation, NewSkeleton, SkeletonNode
from syconn.new_skeleton.newskeleton import integer_checksum
from syconn.utils.basics import euclidian_distance


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

#
def debugBFSFunc(node, origin_node, level, count, context):
    if origin_node == None:
        origin_str = "None"
    else:
        origin_str = origin_node.getID()
    print "%d\t%s\t%d\t%d" % (node.getID(), origin_str, level, count)
    return

#
class AnnotationCanonizer:
    def __init__(self, annotation):
        self.annotation = annotation
        return

    #
    def validateRoot(self, new_root):
        root = self.annotation.getRoot()
        if new_root == None:
            if root <> None:
                return
        elif root == new_root:
            return
        raise "Root failed!"

    #
    def validateRootNoParent(self):
        if len(self.annotation.getRoot().getParents()) > 0:
            raise "Root Parent failed!"
        return

    #
    def validateSingleParent(self):
        nodes = self.annotation.getNodes().copy()
        nodes.remove(self.annotation.getRoot())
        tmp = [node.getSingleParent() for node in nodes]
        return

    #
    def canonize(self, new_root):
        def canonizeBFSFunc(node, origin_node, level, count, node_children):
            if origin_node <> None:
                node_children.setdefault(origin_node, []).append(node)
            return

        #
        def validateNoneParentBFS(parent_to_child, root):
            none_children = parent_to_child[None]
            if len(children) == 1:
                if children[0] == root:
                    return
            raise "None Parent For Non-Root!"

        #
        if new_root <> None:
            self.annotation.resetRoot(new_root)
        self.validateRoot(new_root)
        root = self.annotation.getRoot()
        parent_to_child = {}
        orphans = AnnotationBFS(self.annotation, root, canonizeBFSFunc,
            parent_to_child).annnotationNeighborsBFS()
        # Clear edges (before removing orphans, to avoid wasting time remove each orphan's edges)
        self.annotation.clearEdges()
        # Remove orphans
        for node in orphans:
            self.annotation.removeNode(node)
        print "Deleted %d Orphans" % len(orphans)
        # Re-edge, descending from root
        for (parent, children) in parent_to_child.items():
            for child in children:
                parent.addChild(child)
        self.validateRootNoParent()
        self.validateSingleParent()
        return

    #
    pass


def skeletonCanonize(skeleton, root_f):
    def noneFunc(annotation):
        return None

    if root_f is None:
        root_f = noneFunc
    for annotation in skeleton.getAnnotations():
        AnnotationCanonizer(annotation).canonize(root_f(annotation))
    return


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
        skeleton = NewSkeleton()
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

    s = NewSkeleton()
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

        s = NewSkeleton()
        s.fromNml(cur_f, scaling=scaling)
        main_tracing, main_tracing_len = get_largest_annotation(s)

        s = NewSkeleton()
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

    s = NewSkeleton()
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
        skel = NewSkeleton()
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

    s = NewSkeleton()
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