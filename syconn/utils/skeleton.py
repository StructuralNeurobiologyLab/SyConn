# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from xml.dom import minidom
import xml.etree.cElementTree as cElementTree
from math import pow, sqrt
from syconn.utils.basics import euclidian_distance
import copy
import tempfile
import unicodedata
import hashlib
import struct
import numpy as np
import h5py
import os
import shutil
import zipfile
from multiprocessing import Pool


class Skeleton:
    """
    Basic class for cell tracings used in Knossos

    Attributes
    ----------
    annotations : set of SkeletonAnnotations
        several tracings of one cell
    scaling : tuple
        scaling of dataset
    """
    def __init__(self):
        # Uninitialized Mandatory
        self.annotations = set()
        # Uninitialized Optional
        self.skeleton_time = None
        self.skeleton_idletime = None
        self.skeleton_comment = None
        self.scaling = None
        self.branchNodes = [] # list of node IDs
        self.active_node = None
        self.edit_position = None
        self.experiment_name = None
        self.version = '4.1.2'
        return

    def set_edit_position(self, edit_position):
        self.edit_position = edit_position

    def reset_all_ids(self):
        """
        Rebases all node IDs, starting at 1 again.
        We should have just one node ID and one annotation ID instead of this
        mess with base and high IDs and only make the node IDs
        knossos-compatible when writing to nmls.
        annotation.high_id : highest node id in this annotation
        annotation.nodeBaseID : lowest node id in this annotation
        """
        cnt_a = 1
        cnt_all_nodes = 1
        for a in self.annotations:
            a.annotation_ID = cnt_a # value currently unused - a nice sign of
            # the clusterfuck
            a.nodeBaseID = cnt_all_nodes
            cnt_n = 1
            for n in a.getNodes():
                n.ID = cnt_n
                cnt_n += 1
                cnt_all_nodes += 1
            a.high_id = cnt_n

    def reset_all_times(self):
        for cur_n in self.getNodes():
            cur_n.setDataElem('time', 0)
        self.skeleton_time = 0
        self.skeleton_idletime = 0

    def getAnnotations(self):
        return self.annotations

    def getNodes(self):
        all_nodes_lst = []
        for cur_annotation in self.getAnnotations():
            all_nodes_lst.extend(list(cur_annotation.getNodes()))
        all_nodes = set(all_nodes_lst)
        return all_nodes

    def getVolumes(self):
        all_volumes = set()
        for cur_annotation in self.getAnnotations():
            all_volumes = all_volumes.union(cur_annotation.getVolumes())

        return all_volumes

    def has_node(self, node):
        for cur_anno in self.getAnnotations():
            if cur_anno.has_node(node):
                return True
        return False

    def add_annotation(self, annotation):
        """
        Add SkeletonAnnotation annotation to the skeleton. Node IDs in
        annotation may not be preserved.
        """

        high_id = self.get_high_node_id()
        annotation.nodeBaseID = high_id + 1

        self.annotations.add(annotation)

    def toSWC(self, path=''):
        """
        SWC is a standard skeleton format, very similar to nml, however,
        it is edge-centric instead of node-centric. Some spec can be found here:
        http://research.mssm.edu/cnic/swc.html
        SWC can also be read by Amira
        :param path: str
        :return:
        """
        with open(path, 'w') as trg_file:
            edges = []
            rev_edges = []
            for a in self.getAnnotations():
                # after this, edges is a list of dictionaries containing the
                # edges: key: node, value: node obj
                edges.append(a.getEdges())
                rev_edges.append(a.getReverseEdges())

            # find a root
            for anno_index, e_dict in enumerate(edges):
                root_set = False
                for src_n in e_dict.keys():
                    src_id = src_n.getUniqueID()
                    if len(e_dict[src_n]) > 0:
                        for trg_n in e_dict[src_n]:
                            trg_x = trg_n.getCoordinate()[0]
                            trg_y = trg_n.getCoordinate()[1]
                            trg_z = trg_n.getCoordinate()[2]
                            trg_r = trg_n.getDataElem('radius')
                            trg_id = trg_n.getUniqueID()

                            n_str = '{0:d} 0 {1:f} {2:f} {3:f} {4:f} {' \
                                    '5:d}'.format(
                                trg_id, trg_x, trg_y, trg_z, trg_r, src_id)

                            trg_file.write(n_str + '\n')
                    if (len(rev_edges[anno_index][src_n]) == 0) and not \
                            root_set:
                        # set the root
                        root_set = True
                        trg_x = src_n.getCoordinate()[0]
                        trg_y = src_n.getCoordinate()[1]
                        trg_z = src_n.getCoordinate()[2]
                        trg_r = src_n.getDataElem('radius')
                        trg_id = src_n.getUniqueID()

                        n_str = '{0:d} 0 {1:f} {2:f} {3:f} {4:f} {5:d}'.format(
                            trg_id, trg_x, trg_y, trg_z, trg_r, -1)

                        trg_file.write(n_str + '\n')

                if not root_set:
                    print('Warning, no root set.')

    def fromSWC(self, path=''):
        """
        SWC is a standard skeleton format, very similar to nml.
        :param path: str
        :return:
        """

    def fromNml(self, filename, use_file_scaling=False, scaling='dataset', comment=None, meta_info_only=False):
        if filename.endswith('k.zip'):
            zipper = zipfile.ZipFile(filename)

            if not 'annotation.xml' in zipper.namelist():
                raise Exception("k.zip file does not contain annotation.xml")

            xml_string = zipper.read('annotation.xml')
	    doc = minidom.parseString(xml_string)
	else:
	    doc = minidom.parse(filename)

        self.fromDom(doc, use_file_scaling, scaling, comment, meta_info_only=meta_info_only)

        return self

    def fromNmlString(self, nmlString, use_file_scaling=False, scaling='dataset', comment=None, meta_info_only=False):
        doc = minidom.parseString(nmlString)
        self.fromDom(doc, use_file_scaling, scaling, comment, meta_info_only=False)

        return self

    def fromDom(self, doc, use_file_scaling=False,  scaling='dataset', comment=None, meta_info_only=False):
        try:
            [self.experiment_name] = parse_attributes(
                doc.getElementsByTagName(
                    "parameters")[0].getElementsByTagName(
                    "experiment")[0], [["name", str]])
        except IndexError:
            self.experiment_name = None

        try_time_slice_version = False
        # Read skeleton time and idle time
        try:
            [self.skeleton_time, skeleton_time_checksum] = parse_attributes(
                    doc.getElementsByTagName("parameters")[0].getElementsByTagName("time")[0],
                    [["ms", int], ["checksum", str]])
            [self.skeleton_idletime, idletime_checksum] = parse_attributes(
                    doc.getElementsByTagName("parameters")[0].getElementsByTagName("idleTime")[0],
                    [["ms", int], ["checksum", str]])

        except IndexError:
            self.skeleton_time = None
            self.skeleton_idletime = None
            try_time_slice_version = True

        if try_time_slice_version:
            # Time slicing version
            self.skeleton_time, skeleton_time_checksum = parse_attributes(
                    doc.getElementsByTagName("parameters")[0].getElementsByTagName("time")[0],
                    [["min", int], ["checksum", str]])
            if self.skeleton_time is None:
                self.skeleton_time, skeleton_time_checksum = parse_attributes(
                        doc.getElementsByTagName("parameters")[0].getElementsByTagName("time")[0],
                        [["ms", int], ["checksum", str]])
                if skeleton_time_checksum != integer_checksum(self.skeleton_time):
                    raise Exception("Checksum mismatch")
            else:
                if skeleton_time_checksum != integer_checksum(self.skeleton_time):
                    raise Exception("Checksum mismatch")
                self.skeleton_time = self.skeleton_time * 60 * 1000
                skeleton_time_checksum = integer_checksum(self.skeleton_time)

            self.skeleton_idletime = 0
            idletime_checksum = integer_checksum(0)

        if use_file_scaling == True:
            self.scaling = parse_attributes(doc.getElementsByTagName("parameters")[0].getElementsByTagName("scale")[0], [["x", float], ["y", float], ["z", float]])
        else:
            if isinstance(scaling, str):
                self.scaling = None
            else:
                self.scaling = scaling

        try:
            [self.last_saved_version] = parse_attributes(doc.getElementsByTagName("parameters")[0].getElementsByTagName("lastsavedin")[0], [["version", str]])
        except IndexError:
            self.last_saved_version = '0'
        try:
            [self.created_version] = parse_attributes(doc.getElementsByTagName("parameters")[0].getElementsByTagName("createdin")[0], [["version", str]])
        except IndexError:
            self.created_version = '0'

        if Version(self.get_version()['saved']) >= Version((3, 4, 2)):
            # Has SHA256 checksums
            if self.skeleton_time is not None and self.skeleton_idletime is not None:
                if skeleton_time_checksum != integer_checksum(self.skeleton_time) or \
                   idletime_checksum != integer_checksum(self.skeleton_idletime):
                       raise Exception('Checksum mismatch!')
            else:
                raise Exception('No time records exist!')

        if meta_info_only:
            return self

        base_id = self.get_high_node_id()

       # Construct annotation. Annotations are trees (called things inside the nml files).
        node_ID_to_node = {}
        annotation_elems = doc.getElementsByTagName("thing")
        for annotation_elem in annotation_elems:
            annotation = SkeletonAnnotation().fromNml(
                    annotation_elem,
                    self,
                    base_id=base_id)
            if comment:
                annotation.setComment(comment)
            self.annotations.add(annotation)
            for node in annotation.getNodes():
                node_ID_to_node[node.getUniqueID()] = node

        # Read node comments
        comment_elems = doc.getElementsByTagName("comment")
        for comment_elem in comment_elems:
            [nodeID, comment] = parse_attributes(comment_elem, [["node",
                int], ["content", unicode]])
            node_ID_to_node[nodeID + base_id].setComment(comment)

        # Read branch points
        branch_elems = doc.getElementsByTagName("branchpoint")
        for branch_elem in branch_elems:
            [nodeID] = parse_attributes(branch_elem, [["id", int]])
            self.branchNodes.append(nodeID)
        return self

    def fromNmlcTree(self, filename, use_file_scaling=False, scaling='dataset',
                     comment=None):
        """Reads nml file with cElementTree Parser
        is capable of parsing patches

        Parameters
        ----------
        filename : str
            path to nml-file or path to k.zip-file (but this might cause
            trouble with other functions since self.filename = *.k.zip)
        use_file_scaling : bool
        scaling : str
        comment : str
        """
        if filename:
            self.filename = filename
            try:
                # this was introduced to identify *.nml files that are really
                #  *.k.zip files that have a wrong file ending.
                zip_test = zipfile.ZipFile(filename)
                is_k_zip = True
            except:
                is_k_zip = False

        if filename.endswith('k.zip') or is_k_zip:
            zipper = zipfile.ZipFile(filename)

            if not 'annotation.xml' in zipper.namelist():
                raise Exception("k.zip file does not contain annotation.xml")

            xml_string = zipper.read('annotation.xml')
            root = cElementTree.fromstring(xml_string)
        else:
            with open(filename) as fh:
                xml_string = fh.read()
            root = cElementTree.fromstring(xml_string)
        try:
            [self.experiment_name] = parse_cET(
                root.find("parameters").find("experiment"),
                [["name", str]])
        except AttributeError:
            self.experiment_name = None

        # Read skeleton time and idle time
        try:
            [self.skeleton_time, skeleton_time_checksum] = parse_cET(
                root.find("parameters").find("time"),
                [["ms", int], ["checksum", str]])
            [self.skeleton_idletime, idletime_checksum] = parse_cET(
                root.find("parameters").find("idleTime"),
                [["ms", int], ["checksum", str]])

        except AttributeError:
            self.skeleton_time = None
            self.skeleton_idletime = None

        if use_file_scaling == True:
            self.scaling = parse_cET(root.find("parameters").find("scale"), [["x", float], ["y", float], ["z", float]])
        else:
            if isinstance(scaling, str):
                self.scaling = [1.,1.,1.]
            else:
                self.scaling = scaling

        try:
            [self.last_saved_version] = parse_cET(root.find("parameters").find("lastsavedin"), [["version", str]])
        except AttributeError:
            self.last_saved_version = '0'
        try:
            [self.created_version] = parse_cET(root.find("parameters").find("createdin"), [["version", str]])
        except AttributeError:
            self.created_version = '0'

        if Version(self.get_version()['saved']) >= Version((3, 4, 2)):
            # Has SHA256 checksums
            if self.skeleton_time:
                if skeleton_time_checksum != integer_checksum(self.skeleton_time):
                    raise Exception('Checksum mismatch!')
            if self.skeleton_idletime:
                if idletime_checksum != integer_checksum(self.skeleton_idletime):
                    raise Exception('Checksum mismatch!')

        base_id = self.get_high_node_id()

        # Construct annotation. Annotations are trees (called things inside the nml files).
        node_ID_to_node = {}
        annotation_elems = root.findall("thing")
        for annotation_elem in annotation_elems:
            annotation = SkeletonAnnotation().fromNmlcTree(
                    annotation_elem,
                    self,
                    base_id=base_id)

            if comment:
                annotation.setComment(comment)
            self.annotations.add(annotation)
            for node in annotation.getNodes():
                node_ID_to_node[node.getUniqueID()] = node

        # Read node comments
        try:
            comment_elems = root.find("comments").findall("comment")
        except:
            print "'NoneType' object has no attribute 'findall'"
            comment_elems = []


        for comment_elem in comment_elems:
            try:
                [nodeID, comment] = parse_cET(comment_elem, [["node", int], ["content", str]])
                node_ID_to_node[nodeID + base_id].setComment(comment)
            except KeyError:
                print('Skipping comment, wrong node ID in comments sections: '
                      '' + str(nodeID))
            except UnicodeEncodeError:
                # this means that a comment contains a non-ascii letter
                [nodeID, comment] = parse_cET(comment_elem, [["node", int], ["content", unicode]])
                ascii_comment = unicodedata.normalize('NFKD', comment).encode('ascii','ignore')
                node_ID_to_node[nodeID + base_id].setComment(ascii_comment)
        # Read branch points
        branch_elems = root.findall("branchpoint")
        for branch_elem in branch_elems:
            [nodeID] = parse_cET(branch_elem, [["id", int]])
            self.branchNodes.append(nodeID)

        # pre init scaled coordinates for later usage
        for node in self.getNodes():
           node.getCoordinate_scaled()

        return self

    def get_high_node_id(self):
        """
        Return highest node ID in any annotation in the skeleton.
        """
        high_ids = [0]
        for cur_anno in self.annotations:
            high_ids.append(cur_anno.nodeBaseID + cur_anno.high_id) # this
            # could be buggy : in my understanding, nodeBaseID should not be
            # added to the high_id; jk 20.01.15 - not changing it to not
            # break stuff
        return max(high_ids)

    def toNml(self, filename, save_empty=True):

        try:
            f = open(filename, "w")
            f.write(self.to_xml_string(save_empty))
            f.close()
        except Exception, e:
            print "Couldn't open file for writing."
            print e
        return

    def to_kzip(self, filename, save_empty=True, force_overwrite=False):
        """
        Similar to self.toNML, but writes NewSkeleton to k_zip.
        :param filename: Path to k.zip
        :param save_empty: like in self.to_xml_string
        :param force_overwrite: overwrite existing .k.zip
        :return:
        """
        if os.path.isfile(filename):
            try:
                if force_overwrite:
                    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr('annotation.xml', self.to_xml_string(save_empty))
                else:
                    remove_from_zip(filename, 'annotation.xml')
                    with zipfile.ZipFile(filename, "a", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr('annotation.xml', self.to_xml_string(save_empty))
            except Exception, e:
                print "Couldn't open file for reading and overwriting.", e
        else:
            try:
                with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr('annotation.xml', self.to_xml_string(save_empty))
            except Exception, e:
                print "Couldn't open file for writing.", e
        return

    def to_xml_string(self, save_empty=True):
        # This is currently only a slight cosmetic duplication of the old Skeleton, therefore far from complete
        doc = minidom.Document()
        annotations_elem = doc.createElement("things")
        doc.appendChild(annotations_elem)

        # Add dummy header
        parameters = doc.createElement("parameters")
        props = doc.createElement("properties")
        annotations_elem.appendChild(parameters)
        expname = doc.createElement("experiment")

        if self.experiment_name is not None:
            build_attributes(expname, [["name", self.experiment_name]])
            parameters.appendChild(expname)

        if self.scaling is not None:
            scale = doc.createElement("scale")
            build_attributes(scale, [["x", self.scaling[0]], ["y", self.scaling[1]], ["z", self.scaling[2]]])
            parameters.appendChild(scale)

        if self.edit_position is not None:
            edit_position = doc.createElement("editPosition")
            build_attributes(
                edit_position,
                [["x", self.edit_position[0]],
                 ["y", self.edit_position[1]],
                 ["z", self.edit_position[2]], ])
            parameters.appendChild(edit_position)

        # find property keys
        orig_keys = ["inVp", "node", "id", "inMag", "radius", "time",
                     "x", "y", "z", "edge", "comment", "content", "target"]
        #  assume properties are set in every node -> only look at a single
        #  node is sufficient
        for n in self.getNodes():
            property_names = []
            for key, val in n.data.iteritems():
                if key not in property_names+orig_keys:
                    property_names.append(key)
                    prop_entry = doc.createElement("property_name")
                    build_attributes(prop_entry,
                                     [["ms", 0], ["checksum", integer_checksum(0)]])
            break

        time = doc.createElement("time")
        build_attributes(time, [["ms", 0], ["checksum", integer_checksum(0)]])
        parameters.appendChild(time)

        time = doc.createElement("idleTime")
        build_attributes(time, [["ms", 0], ["checksum", integer_checksum(0)]])
        parameters.appendChild(time)

        if self.active_node is not None:
            activenode = doc.createElement("activeNode")
            build_attributes(activenode, [["id", self.active_node.getID()]])
            parameters.appendChild(activenode)

        createdInVersion = doc.createElement("createdin")
        build_attributes(createdInVersion, [["version", self.version]])
        parameters.appendChild(createdInVersion)

        createdInVersion = doc.createElement("lastsavedin")
        build_attributes(createdInVersion, [["version", self.version]])
        parameters.appendChild(createdInVersion)

        comments_elem = doc.createElement("comments")
        annotations_elem.appendChild(comments_elem)
        annotation_ID = 0
        for annotation in self.getAnnotations():
            if not save_empty:
                if annotation.isEmpty():
                    continue
            annotation_ID += 1
            annotation.toNml(doc, annotations_elem, comments_elem, annotation_ID)

        branchNodes_elem = doc.createElement("branchpoints")
        annotations_elem.appendChild(branchNodes_elem)
        for branchID in self.branchNodes:
            branchNode_elem = doc.createElement("branchpoint")
            build_attributes(branchNode_elem, [["id", branchID]])
            branchNodes_elem.appendChild(branchNode_elem)

        return doc.toprettyxml()

    def getSkeletonTime(self):
        if self.skeleton_time == None:
            return None
        if Version(self.get_version()['saved']) == Version((3, 4)):
            return self.skeleton_time ^ 1347211871
        else:
            return self.skeleton_time

    def getIdleTime(self):
        if self.skeleton_idletime == None:
            return None
        if Version(self.get_version()['saved']) == Version((3, 4)):
            return self.skeleton_idletime ^ 1347211871
        else:
            return self.skeleton_idletime

    def get_version(self):
        # Check whether the version strings are in the format
        # x.y.z
        try:
            int(self.created_version.translate(None, '.'))
        except ValueError:
            created_version = [0,]
        else:
            created_version = [int(x) for x in self.created_version.split('.')]

        try:
            int(self.last_saved_version.translate(None, '.'))
        except ValueError:
            last_saved_version = [0,]
        else:
            last_saved_version = [int(x) for x in self.last_saved_version.split('.')]

        if self.created_version == '4.0 Beta 2':
            created_version = [3, 99, 2]
        if self.created_version == '4.1 Alpha':
            created_version = [4, 0, 99, 2]
        if self.created_version == '4.1 Pre Alpha':
            created_version = [4, 0, 99, 1]
        if self.last_saved_version == '4.0 Beta 2':
            last_saved_version = [3, 99, 2]
        if self.last_saved_version == '4.1 Alpha':
            last_saved_version = [4, 0, 99, 2]
        if self.last_saved_version == '4.1 Pre Alpha':
            last_saved_version = [4, 0, 99, 1]
        return {'created': created_version, 'saved': last_saved_version}
    #
    def set_scaling(self, scaling):
        if isinstance(scaling, str):
            scaling = None
        self.scaling = scaling
    #
    pass


class SkeletonAnnotation:
    def interpolate_nodes(self, max_node_dist_scaled=50):
        # test scaling available
        # get all edges with length > max_node_distance_scaled
        if not self.scaling:
            raise Exception('Cannot interpolate without scaling.')

        interpolated_edges = 1
        while interpolated_edges:
            interpolated_edges = 0
            for src_node in self.edges.keys():
                for trg_node in self.edges[src_node]:
                    if src_node.distance_scaled(trg_node) > max_node_dist_scaled:
                        interpolated_edges += 1
                        # remove this edge
                        self.removeEdge(src_node, trg_node)

                        # get new node coordinate in between the old nodes
                        a = src_node.getCoordinate()
                        b = trg_node.getCoordinate()
                        c = [int(round((a[0]+b[0])/2., 0)),
                             int(round((a[1]+b[1])/2., 0)),
                             int(round((a[2]+b[2])/2., 0))]

                        # add node from scratch in between
                        new_node = SkeletonNode()
                        new_node.from_scratch(self, c[0], c[1], c[2])
                        self.addNode(new_node)
                        # add edges to new node, and from new node
                        self.addEdge(src_node, new_node)
                        self.addEdge(new_node, trg_node)

        return

    def resetObject(self):
        # Mandatory
        self.root = None
        self.nodes = set()
        self.node_ID_to_node = {}
        self.edges = {}
        self.reverse_edges = {}
        self.nodeBaseID = 1
        self.scaling = None
        self.filename = None
        # Optional
        self.annotation_ID = None
        self.comment = ''
        # Local to annotation, always starts at 0. Highest
        # node ID in the annotation.
        self.high_id = 0
        self.volumes = set()
        return

    def __init__(self):
        self.resetObject()
        return

    def __len__(self):
        return len(self.nodes)

    def __copy__(self):
        """
        Make a complete copy of the object along with its associated nodes.
        """
        new = SkeletonAnnotation()
        new.__dict__.update(self.__dict__)
        new.clearEdges()
        new.clear_nodes()

        old_to_new_nodes = {}
        for cur_n in self.getNodes():
            new_cur_n = copy.copy(cur_n)
            old_to_new_nodes[cur_n] = new_cur_n
            new.addNode(new_cur_n)

        for cur_from, cur_to in self.iter_edges():
            try:
                cur_from_new = old_to_new_nodes[cur_from]
                cur_to_new = old_to_new_nodes[cur_to]
            except KeyError:
                # Can be raised if there is a connection between different
                # annotations
                continue
            new.addEdge(cur_from_new, cur_to_new)

        # This is to allow matching of nodes between copied annotations
        new.old_to_new_nodes = old_to_new_nodes

        return new

    def clear_nodes(self):
        self.nodes = set()
        self.node_ID_to_node = {}
        self.root = None
        self.nodeBaseID = 0
        self.high_id = 0

    def fromNml(self, annotation_elem, skeleton, base_id=0):
        self.resetObject()

        self.setNodeBaseID(base_id)

        [comment] = parse_attributes(annotation_elem, [['comment', str],])
        if comment:
            self.setComment(comment)

        # Read nodes
        node_elems = annotation_elem.getElementsByTagName("node")
        for node_elem in node_elems:
            node = SkeletonNode().fromNml(self, node_elem)
            self.addNode(node)
        #
        # Read edges
        edge_elems = annotation_elem.getElementsByTagName("edge")
        for edge_elem in edge_elems:
            (source_ID, target_ID) = parse_attributes(edge_elem, [["source", int], ["target", int]])
            try:
                source_node = self.node_ID_to_node[source_ID]
                target_node = self.node_ID_to_node[target_ID]
                source_node.addChild(target_node)
            except KeyError:
                print 'Warning: Parsing of edges between different things is not yet supported, skipping edge: ' + str(source_ID) + ' -> ' + str(target_ID)

        self.scaling = skeleton.scaling

        return self

    def fromNmlcTree(self, annotation_elem, skeleton, base_id=0):
        """ Subfunction of fromNmlcTree from NewSkeleton

        Parameters
        ----------
        annotation_elem : XML Element
        skeleton : Skeleton
        base_id : int
        """
        self.resetObject()

        self.setNodeBaseID(base_id)

        self.scaling = skeleton.scaling

        [comment] = parse_cET(annotation_elem, [['comment', str],])
        if comment:
            self.setComment(comment)

        # Read nodes
        node_elems = []

        for i in annotation_elem.findall("nodes"):
            for j in i.findall("node"):
                node_elems.append(j)

        for node_elem in node_elems:
            node = SkeletonNode().fromNmlcTree(self, node_elem)
            self.addNode(node)

        # Read edges
        edge_elems = []

        for i in annotation_elem.findall("edges"):
            for j in i.findall("edge"):
                edge_elems.append(j)

        for edge_elem in edge_elems:
            (source_ID, target_ID) = parse_cET(edge_elem, [["source", int], ["target", int]])
            try:
                source_node = self.node_ID_to_node[source_ID]
                target_node = self.node_ID_to_node[target_ID]
                source_node.addChild(target_node)
            except KeyError:
                print 'Warning: Parsing of edges between different things is not yet supported, skipping edge: ' + str(source_ID) + ' -> ' + str(target_ID)

        # Read patches
        patch_elems = []

        for i in annotation_elem.findall("patches"):
            for j in i.findall("patch"):
                patch_elems.append(j)

        for patch_elem in patch_elems:
            volume = SkeletonVolume().fromNmlcTree(patch_elem)
            self.volumes.add(volume)

        return self

    def toNml(self, doc, annotations_elem, comments_elem, annotation_ID):
        annotation_elem = doc.createElement("thing")
        build_attributes(annotation_elem, [["id", annotation_ID]])

        if self.getComment():
            annotation_elem.setAttribute("comment", self.getComment())

        nodes_elem = doc.createElement("nodes")
        edges_elem = doc.createElement("edges")
        for node in self.getNodes():
            node.toNml(doc, nodes_elem, edges_elem, comments_elem)
        #
        annotation_elem.appendChild(nodes_elem)
        annotation_elem.appendChild(edges_elem)
        annotations_elem.appendChild(annotation_elem)
        return

    def has_node(self, node):
        for cur_node in self.getNodes():
            if node == cur_node.getCoordinate():
                return True
        return False

    def clearNodeEdges(self, node):
        self.edges[node] = set()
        self.reverse_edges[node] = set()
        return

    def getNodes(self):
        return self.nodes

    def iter_edges(self):
        for cur_from_node, to_nodes in self.getEdges().iteritems():
            for cur_to_node in to_nodes:
                yield (cur_from_node, cur_to_node)

    def physical_length(self):
        """
        Return the physical length along all edges in the annotation.
        """

        total_length = 0.

        for from_node, to_node in self.iter_edges():
            total_length += euclidian_distance(
                from_node.getCoordinate_scaled(),
                to_node.getCoordinate_scaled())

        return total_length

    def avg_inter_node_distance(self, outlier_distance = 2000.):
        """
        Calculates the average inter node distance for an annotation. Candidate
        for inclusion into the skeleton annotation object.

        :param anno: annotation object
        :param outlier_filter: float, ignore distances higher than value
        :return: float
        """
        edge_cnt = 0
        total_length = 0.

        for from_node, to_node in self.iter_edges():
            this_dist = euclidian_distance(from_node.getCoordinate_scaled(),
                               to_node.getCoordinate_scaled())
            if this_dist < outlier_distance:
                total_length += this_dist
                edge_cnt += 1

        if edge_cnt:
            avg_dist = total_length / float(edge_cnt)
            return avg_dist
        else:
            print('No edges in current annotation, cannot calculate inter node '
                  'distance.')
            return

    def addNode(self, node):
        this_id = node.getID()
        if this_id == None or self.node_ID_to_node.has_key(this_id):
            this_id = self.high_id + 1
            node.setID(this_id)
        if this_id > self.high_id:
            self.high_id = this_id
        self.nodes.add(node)
        node.annotation = self
        self.node_ID_to_node[this_id] = node
        self.clearNodeEdges(node)
        return id

    def removeNode(self, node):
        if not node.annotation:
            raise Exception('this should not exist')

        node.detachAnnotation()

        reverse_edges = []
        edges = []
        for cur_reverse_edge in self.reverse_edges[node]:
            reverse_edges.append((node, cur_reverse_edge))
        for cur_edge in self.edges[node]:
            edges.append((node, cur_edge))

        for to_n, from_n in reverse_edges:
            self.edges[from_n].remove(to_n)

        for from_n, to_n in edges:
            self.reverse_edges[to_n].remove(from_n)

        self.nodes.remove(node)

        del self.node_ID_to_node[node.getID()]
        del self.edges[node]
        del self.reverse_edges[node]

        return

    def addEdge(self, node, target_node):
        self.edges[node].add(target_node)
        self.reverse_edges[target_node].add(node)
        return

    def removeEdge(self, node, target_node):
        self.edges[node].remove(target_node)
        self.reverse_edges[target_node].remove(node)
        return

    def getNodeEdges(self, node):
        return self.edges[node]

    def getNodeReverseEdges(self, node):
        return self.reverse_edges[node]

    def getEdges(self):
        return self.edges

    def getReverseEdges(self):
        return self.reverse_edges

    def clearEdges(self):
        self.edges = {}
        self.reverse_edges = {}

    def getVolumes(self):
        return self.volumes

    def getRoot(self):
        return self.root

    def setRootInternal(self, root):
        self.root = root
        return

    def setRoot(self, root):
        if self.getRoot() <> None:
            raise RuntimeError("Root already exists!")
        root.setRoot()
        return

    def unRootInternal(self):
        self.root = None
        return

    def unRoot(self):
        if self.root == None:
            raise RuntimeError("No Root!")
        self.root.unRoot()
        return

    def resetRoot(self, new_root):
        if self.getRoot() <> None:
            self.unRoot()
        self.setRoot(new_root)
        return

    def getNodeBaseID(self):
        return self.nodeBaseID

    def setNodeBaseID(self, nodeBaseID):
        self.nodeBaseID = nodeBaseID
        return

    def getNodeByID(self, nodeID):
        return self.node_ID_to_node[nodeID]

    def getNodeByUniqueID(self, uniqueNodeID):
        return self.getNodeByID(uniqueNodeID - self.getNodeBaseID())

    def getComment(self):
        return self.comment

    def setComment(self, comment):
        self.comment = comment

    def appendComment(self, comment):
        cur_comment = self.getComment()
        if cur_comment:
            self.setComment('%s; %s' %  (cur_comment, comment))
        else:
            self.setComment(comment)

    def isEmpty(self):
        if len(self.nodes) == 0:
            return True
        else:
            return False

    def add_path(self, path):
        """
        Add all nodes in path to the annotation, connecting them linearly.

        Parameters
        ----------

        path : List of SkeletonNode objects
            Nodes that will be added to annotation, connected in order.
        """

        self.addNode(path[0])
        for from_node, to_node in zip(path, path[1:]):
            self.addNode(to_node)
            self.addEdge(from_node, to_node)
    #
    pass

node_metadata_begin_token = "__NODE_MD_B__"
node_metadata_end_token = "__NODE_MD_E__"
node_metadata_elem_delim_token = "__NODE_MD_D__"
node_metadata_elem_keyvalue_token = "__NODE_MD_K__"
node_metadata_key_root = "__NODE_ROOT__"


class SkeletonNode:
    def resetObject(self):
        # Using extra data dictionaries is a bad idea and somewhat redundant,
        # see definition of __copy__ below.
        # Uninitialize mandatory
        self.data = {}
        self.ID = None
        # Uninitialized optional members
        self.pure_comment = ""
        self.metadata = {}
        self.annotation = None
        return

    def __init__(self):
        self.resetObject()
        return

    def __repr__(self):
        return "Node at %s" % (self.getCoordinate(),)

    def __copy__(self):
        """
        Allow us to use copy.copy() on SkeletonNode instances without
        the actual content being shared between the copies, as would
        otherwise happen, since the data and metadata attributes would then
        refer to the same dictionaries.
        """

        new = SkeletonNode()
        new.__dict__.update(self.__dict__)
        new.ID = None
        new.data = {}
        new.data.update(self.data)
        new.metadata = {}
        new.metadata.update(self.metadata)

        return new

    def setRoot(self):
        self.addMetaDataElem((node_metadata_key_root, ""))
        return

    def unRoot(self):
        self.removeMetaDataKey(node_metadata_key_root)
        return

    def from_coordinate(coordinate):
        new = SkeletonNode()
        new.setCoordinate(coordinate)

        return new

    def from_scratch(self, annotation, x, y, z, inVp=1, inMag=1, time=0, ID=None, radius=1.0):
        self.resetObject()
        self.annotation = annotation

        self.ID = ID

        self.x = x
        self.y = y
        self.z = z

        self.x_scaled = self.x * self.annotation.scaling[0]
        self.y_scaled = self.y * self.annotation.scaling[1]
        self.z_scaled = self.z * self.annotation.scaling[2]

        self.setDataElem("inVp", inVp)
        self.setDataElem("radius", radius)
        self.setDataElem("inMag", inMag)
        self.setDataElem("time", time)

        return self

    def fromNml(self, annotation, node_elem):
        self.resetObject()
        self.annotation = annotation
        [x, y, z, inVp, inMag, time, ID, radius] = parse_attributes(node_elem, \
            [["x", int], ["y", int], ["z", int], ["inVp", int], ["inMag", int],
             ["time", int], ["id", int], ["radius", float]])
        self.ID = ID
        self.x = x
        self.y = y
        self.z = z
        try:
            self.x_scaled = self.x * self.annotation.scaling[0]
            self.y_scaled = self.y * self.annotation.scaling[1]
            self.z_scaled = self.z * self.annotation.scaling[2]
        except TypeError:
            self.x_scaled = self.x
            self.y_scaled = self.y
            self.z_scaled = self.z

        self.setDataElem("inVp", inVp)
        self.setDataElem("radius", radius)
        self.setDataElem("inMag", inMag)
        self.setDataElem("time", time)
        return self

    def fromNmlcTree(self, annotation, node_elem):
        self.resetObject()
        self.annotation = annotation
        [x, y, z, inVp, inMag, time, ID, radius], additional_attr = parse_cET(node_elem, \
            [["x", int], ["y", int], ["z", int], ["inVp", int], ["inMag", int],
             ["time", int], ["id", int], ["radius", float]], ret_all_attr=True)
        self.ID = ID

        self.x = x
        self.y = y
        self.z = z

        self.x_scaled = self.x * self.annotation.scaling[0]
        self.y_scaled = self.y * self.annotation.scaling[1]
        self.z_scaled = self.z * self.annotation.scaling[2]

        self.setDataElem("inVp", inVp)
        self.setDataElem("radius", radius)
        self.setDataElem("inMag", inMag)
        self.setDataElem("time", time)

        for key, val in additional_attr.iteritems():
            self.setDataElem(key, val)
        return self

    def toNml(self, doc, nodes_elem, edges_elem, comments_elem):
        node_elem = doc.createElement("node")
        build_attributes(node_elem, [("id", self.getUniqueID())])
        build_attributes(node_elem, [("inMag", self.data["inMag"]),
            ("inVp", self.data['inVp']), ("radius", self.data["radius"]),
            ("time", self.data["time"]), ("x", int(self.x)),
            ("y", int(self.y)), ("z", int(self.z))])
        for key, val in self.data.iteritems():
            if key in ["inVp", "node", "id", "inMag", "radius", "time", "x",
                       "y", "z", "edge", "comment", "content", "target"]:
                continue
            build_attributes(node_elem, [(key, val)])
        for child in self.getChildren():
            edge_elem = doc.createElement("edge")
            build_attributes(edge_elem, [["source", self.getUniqueID()], ["target", child.getUniqueID()]])
            edges_elem.appendChild(edge_elem)
        nodes_elem.appendChild(node_elem)
        comment = self.getComment()
        if comment != '' and not comment == None:
            comment_elem = doc.createElement("comment")
            build_attributes(comment_elem, [["node", self.getUniqueID()], ["content", comment]])
            comments_elem.appendChild(comment_elem)
        return

    def getData(self):
        return self.data

    def setData(self, data):
        self.data = data
        return

    def getDataElem(self, elem_name):
        return self.data[elem_name]

    def setDataElem(self, elem_name, elem_value):
        self.data[elem_name] = elem_value
        return

    def getCoordinate(self):
        return [self.x, self.y, self.z]

    def setCoordinate(self, coord):
        if len(coord) is not 3:
            raise Exception('Coordinate dimensionality must be 3.')

        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]

        if not self.annotation:
            raise Exception("No associated annotation for scaling")
        if not self.annotation.scaling:
            raise Exception("No scaling parameter set")

        self.x_scaled = self.x * self.annotation.scaling[0]
        self.y_scaled = self.y * self.annotation.scaling[1]
        self.z_scaled = self.z * self.annotation.scaling[2]

    def getCoordinate_scaled(self):
        return [self.x_scaled, self.y_scaled, self.z_scaled]

    def getID(self):
        # If no corresponding annotation and no ID is set in this
        # instances attributes, then the ID is not defined.
        #
        if self.ID == None:
            self.ID = self.annotation.high_id
            self.annotation.high_id += 1

        return self.ID

    def setID(self, id):
        self.ID = id

    def getUniqueID(self):
        return self.getID() + self.annotation.getNodeBaseID()

    def getPureComment(self):
        return self.pure_comment

    def setPureComment(self, comment):
        self.pure_comment = comment
        return

    def getMetaData(self):
        return self.metadata

    def setMetaData(self, md):
        for md_elem in md.items():
            self.addMetaDataElem(md_elem)
        return

    def removeMetaData(self):
        keys = self.metadata.keys()
        for k in keys:
            self.removeMetaDataKey(k)
        return

    def implyMetadataAddRoot(self):
        self.annotation.setRootInternal(self)
        return

    def addMetaDataElem(self, md_elem):
        add_implications = {node_metadata_key_root:self.implyMetadataAddRoot}
        (k, v) = md_elem
        self.metadata[k] = v
        add_implications[k]()
        return

    def implyMetadataRemoveRoot(self):
        self.annotation.unRootInternal()
        return

    def removeMetaDataKey(self, md_key):
        remove_implications = {node_metadata_key_root:self.implyMetadataRemoveRoot}
        del self.metadata[md_key]
        remove_implications[md_key]()
        return

    def getComment(self):
        def getMetaDataComment(metadata):
            if len(metadata) == 0:
                return ""
            elems_str = node_metadata_elem_delim_token.join([node_metadata_elem_keyvalue_token.join(elem) for elem in metadata.items()])
            return ("%s%s%s" % (node_metadata_begin_token, elems_str, node_metadata_end_token))
        comment = getMetaDataComment(self.metadata) + self.getPureComment()
        return comment

    def delCommentPart(self, comment):
        comment = self.getComment()

        if comment:
            comment = comment.replace('%s; ' % (comment, ), '')
            comment = comment.replace('%s;' % (comment, ), '')
            comment = comment.replace('%s' % (comment, ), '')

            self.setComment(comment)

    def appendComment(self, comment):
        cur_comment = self.getComment()
        if cur_comment:
            self.setComment('%s; %s' %  (cur_comment, comment))
        else:
            self.setComment(comment)

    def setComment(self, comment):
        def parseComment(s):
            pure_comment = ""
            begin_token_split = s.split(node_metadata_begin_token)
            pure_comment += begin_token_split[0]
            chunk_begins = begin_token_split[1:]
            chunks = []
            for x in chunk_begins:
                end_token_split = x.split(node_metadata_end_token)
                pure_comment += end_token_split[1]
                chunks.append(end_token_split[0])
            chunks_elems = []
            for x in chunks:
                chunks_elems += x.split(node_metadata_elem_delim_token)
            chunks_keyvalues = [x.split(node_metadata_elem_keyvalue_token) for x in chunks_elems]
            return (dict(chunks_keyvalues), pure_comment)

        (metadata, pure_comment) = parseComment(comment)
        self.setPureComment(pure_comment)
        self.removeMetaData()
        self.setMetaData(metadata)
        return

    def has_comment_token(self, token):
        comment = self.getComment()
        if comment:
            cur_tokens = comment.split(';')
            cur_tokens = [x.strip() for x in cur_tokens]
            if token in cur_tokens:
                return True
        return False

    def getChildren(self):
        return self.annotation.getNodeEdges(self)

    def addChild(self, child):
        self.annotation.addEdge(self, child)
        return

    def removeChild(self, child):
        self.annotation.removeEdge(self, child)
        return

    def getParents(self):
        return self.annotation.getNodeReverseEdges(self)

    def addParent(self, parent):
        parent.addChild(self)

    def removeParent(self, parent):
        parent.removeChild(self)
        return

    def getSingleParent(self):
        parents = self.getParents()
        if len(parents) <> 1:
            raise RuntimeError("Not a Single Parent!")
        return list(parents)[0]

    def setSingleParent(self, parent):
        if self.getSingleParent() <> None:
            raise RuntimeError("Parent Already Set!")
        self.addParent(parent)
        return

    def removeSingleParent(self):
        parent = self.getSingleParent()
        self.removeParent(parent)
        return

    def getNeighbors(self):
        return set().union(self.getParents(), self.getChildren())

    def detachAnnotation(self):
        self.annotation = None
        return

    def distance_scaled(self, to_node):
        c_1 = self.getCoordinate_scaled()
        c_2 = to_node.getCoordinate_scaled()

        dst = sqrt(pow(c_1[0] - c_2[0], 2) +
                pow(c_1[1] - c_2[1], 2) +
                pow(c_1[2] - c_2[2], 2))

        return dst

    def degree(self):
        return len(self.getNeighbors())
    pass


class SkeletonVolume:
    """ Class for handling patches (=volumes)  """
    def resetObject(self):
        self.loops = set()
        self.scaling = None
        self.bias = [100000, 100000, 100000]
        self.far = [0, 0, 0]
        self.comment = ''
        self.vp = 0
        return

    def __init__(self):
        self.resetObject()
        return

    def __len__(self):
        return len(self.loops)

    def fromNmlcTree(self, patch_elem):
        """ Subfunction of fromNmlcTree from SkeletonAnnotation

        Parameters
        ----------
        patch_elem : XML Element
        """
        self.resetObject()

        [comment] = parse_cET(patch_elem, [['comment', str],])
        if comment:
            self.setComment(comment)

        for loop_elem in patch_elem.findall("loop"):
            loop = SkeletonLoop().fromNmlcTree(loop_elem)
            self.loops.add(loop)
            self.setVP(loop.getVP())
        self.setLimits()
        return self

    def setLimits(self):
        """ Calculates the two outer points (3D) of the box exactly around the volume """
        for loop in self.getLoops():
            bias, far = loop.getLimits()
            for dim in range(3):
                if bias[dim] < self.bias[dim]:
                    self.bias[dim] = bias[dim]
                if far[dim] > self.far[dim]:
                    self.far[dim] = far[dim]

    def getLimits(self):
        return self.bias, self.far

    def getLoops(self):
        return self.loops

    def getComment(self):
        return self.comment

    def setComment(self, comment):
        self.comment = comment

    def getVP(self):
        return self.vp

    def setVP(self, vp):
        self.vp = vp
    pass


class SkeletonLoop:
    """ Class for handling loops of patches """
    def resetObject(self):
        self.bias = [100000, 100000, 100000]
        self.far = [0, 0, 0]
        self.vp = None
        self.points = set()
        self.points_reduced = set()
        self.last = None

        return

    def __init__(self):
        self.resetObject()
        return

    def __len__(self):
        return len(self.points)

    def add(self, point):
        self.points.add(point)
        self.last = point
        self.setLimits(point)

    def fillHole(self, curr, prior):
        """ Adds a point to a hole detected by fromNmlcTree()

        The added point has to be a neighbour of curr (= current point)
        to guarantee a successful use of the while loop.
        """
        point = SkeletonLoopPoint()
        point.x = abs(curr[0]+prior[0])/2
        if abs(point.x-prior[0]) > 1:
            if point.x > prior[0]:
                point.x = prior[0]+1
            if point.x < prior[0]:
                point.x = prior[0]-1
        point.y = abs(curr[1]+prior[1])/2
        if abs(point.y-prior[1]) > 1:
            if point.y > prior[1]:
                point.y = prior[1]+1
            if point.y < prior[1]:
                point.y = prior[1]-1
        point.z = abs(curr[2]+prior[2])/2
        if abs(point.z-prior[2]) > 1:
            if point.z > prior[2]:
                point.z = prior[2]+1
            if point.z < prior[2]:
                point.z = prior[2]-1

        self.add(point)

    def fromNmlcTree(self, loop_elem):
        """Subfunction of fromNmlcTree from SkeletonVolume

        Parameters
        ----------
        loop_elem : XML Element
        """
        self.resetObject()

        [vp] = parse_cET(loop_elem, [['inVP', str],])
        if vp:
            self.setVP(vp)
        for point_elem in loop_elem.findall("point"):
            point = SkeletonLoopPoint().fromNmlcTree(point_elem)
            currentcoord = point.getCoordinates()
            if currentcoord[0] > -1:
                if len(self.points) > 0:
                    while False in \
                            (2 > abs(currentcoord[dim]-self.last.getCoordinates()[dim]) \
                            for dim in range(3)):
                        print "Warning: Loop with hole detected"
                        self.fillHole(currentcoord, self.last.getCoordinates())

                self.add(point)

        return self

    def setLimits(self, point):
        """ Calculates the two outer points (3D) of the box exactly around the loop """
        [x, y, z] = point.getCoordinates()
        if self.bias[0] > x:
            self.bias[0] = x
        if self.bias[1] > y:
            self.bias[1] = y
        if self.bias[2] > z:
            self.bias[2] = z

        if self.far[0] < x:
            self.far[0] = x
        if self.far[1] < y:
            self.far[1] = y
        if self.far[2] < z:
            self.far[2] = z

    def getLimits(self):
        return self.bias, self.far

    def getPoints(self):
        return self.points

    def calculateReducedPoints(self):
        """ Creates new pointset without double appearing points """
        points = self.getPoints()
        for point in points:
            check = 1
            for red_point in self.points_reduced:
                if point.getCoordinates() == red_point.getCoordinates():
                    check = 0
            if check == 1:
                self.points_reduced.add(point)

    def getReducedPoints(self):
        return self.points_reduced

    def getVP(self):
        return self.vp

    def setVP(self, vp):
        self.vp = int(vp)
    pass


class SkeletonLoopPoint:
    """ Class for handling points of loops """
    def resetObject(self):
        self.x = None
        self.y = None
        self.z = None

    def __init__(self):
        self.resetObject()
        return

    def __repr__(self):
        return "Looppoint at %s" % (self.getCoordinates(),)

    def fromNmlcTree(self, point_elem):
        """Subfunction of fromNmlcTree from SkeletonLoop

        Parameters
        ----------
        point_elem : XML Element
        """
        self.resetObject()
        self.setCoordinates(point_elem)
        return self

    def setCoordinates(self, point_elem):
        """ Sets the coordinates for a point and convert them to int """
        [x, y, z] = parse_cET(point_elem, [["x", float], ["y", float], ["z", float]])
        if x > -1:
            [x, y, z] = [int(x), int(y), int(z)]
        [self.x, self.y, self.z] = [x, y, z]

    def getCoordinates(self):
        return [self.x, self.y, self.z]
    pass


def id_lookup_in_one_chunk_thread(args):
    chunk = args[0]
    coordinates = np.array(args[1]) - chunk.coordinates
    name = args[2]
    hdf5_name = args[3]
    if len(args) > 4:
        obj_coords = np.array(args[4]) - chunk.coordinates
        obj_ids = args[5]

        x_obj = []
        y_obj = []
        z_obj = []
        for obj_coord in obj_coords:
            x_obj.append(obj_coord[0])
            y_obj.append(obj_coord[1])
            z_obj.append(obj_coord[2])

    f = h5py.File(chunk.folder + name + ".h5", "r")
    data = f[hdf5_name].value
    f.close()

    x = []
    y = []
    z = []
    for coordinate in coordinates:
        x.append(coordinate[0])
        y.append(coordinate[1])
        z.append(coordinate[2])

    if len(args) > 4:
        return [data[tuple([x, y, z])], data[tuple([x_obj, y_obj, z_obj])], obj_ids]
    else:
        return data[tuple([x, y, z])]


def from_id_lists_to_mergelist(id_lists, coordinates, path, immutable=1):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    f = open(path + "/mergelist.txt", "w")
    for ii, id_list in enumerate(id_lists):
        f.write("%d " % id_list[0])
        f.write("%d 0 " % immutable)
        for id in id_list:
            f.write("%d " % id)
        f.write("\n%d %d %d\n" % (coordinates[ii][0], coordinates[ii][1],
                                   coordinates[ii][2]))
        f.write("skeleton\n \n")
    f.close()

    with zipfile.ZipFile(path + ".k.zip", "w",
                         zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(path):
            for file in files:
                zf.write(os.path.join(root, file), file)
    shutil.rmtree(path)


def from_skeleton_to_mergelist(cset, skeleton_path, name, hdf5_name,
                               obj_coordinates=None, objs=None,
                               mergelist_path=None, nb_processes=1):
    if obj_coordinates is None and objs is None:
        obj_activated = False
    else:
        obj_activated = True
    if type(skeleton_path ) == str:
        skeleton = Skeleton().fromNml(skeleton_path)
    else:
        skeleton = skeleton_path
    nodes = skeleton.getNodes()
    coordinates = []
    for node in nodes:
        coordinates.append(np.array(node.getCoordinate())-np.ones(3))

    chunk_rep = cset.map_coordinates_on_chunks(coordinates)
    if obj_activated:
        chunk_obj_rep = cset.map_coordinates_on_chunks(obj_coordinates)
        chunk_list = np.unique(chunk_rep + chunk_obj_rep)
    else:
        chunk_list = np.unique(chunk_rep)

    chunk_list = chunk_list[chunk_list >= 0]

    coordinates = np.array(coordinates)
    if obj_activated:
        obj_coordinates = np.array(obj_coordinates)
        objs = np.array(objs)
    multi_params = []
    for nb_chunk in chunk_list:
        this_coordinates = coordinates[chunk_rep == nb_chunk]
        if not obj_activated:
            multi_params.append([cset.chunk_dict[nb_chunk], this_coordinates, name, hdf5_name])
        else:
            this_obj_coordinates = obj_coordinates[chunk_obj_rep == nb_chunk]
            this_objs = objs[chunk_obj_rep == nb_chunk]
            multi_params.append([cset.chunk_dict[nb_chunk], this_coordinates,
                                 name, hdf5_name, this_obj_coordinates, this_objs])

    if nb_processes > 1:
        pool = Pool(nb_processes)
        results = pool.map(id_lookup_in_one_chunk_thread, multi_params)
        pool.close()
        pool.join()
    else:
        results = map(id_lookup_in_one_chunk_thread, multi_params)

    id_list = []
    svx_list = []
    obj_id_list = []
    for result in results:
        if obj_activated:
            id_list += result[0].tolist()
            svx_list += result[1].tolist()
            obj_id_list += result[2].tolist()
        else:
            id_list += result.tolist()

    id_list = np.unique(id_list)

    if mergelist_path is None:
        if obj_activated:
            return [(obj_id_list[i], True) if svx_list[i] in id_list else (obj_id_list[i], False)
                    for i in range(len(svx_list))]
        else:
            return id_list
    else:
        from_id_lists_to_mergelist([id_list], [coordinates[0]], mergelist_path)
        if obj_activated:
            return [(obj_id_list[i], True) if svx_list[i] in id_list else (obj_id_list[i], False)
                    for i in range(len(svx_list))]


def remove_from_zip(zipfname, *filenames):
    """
    Removes filenames from zipfile.
    :param zipfname: str Path to zipfile
    :param filenames: list of str Files to delete
    """
    tempdir = tempfile.mkdtemp()
    try:
        tempname = os.path.join(tempdir, 'new.zip')
        with zipfile.ZipFile(zipfname, 'r') as zipread:
            with zipfile.ZipFile(tempname, 'w') as zipwrite:
                for item in zipread.infolist():
                    if item.filename not in filenames:
                        data = zipread.read(item.filename)
                        zipwrite.writestr(item, data)
        shutil.move(tempname, zipfname)
    finally:
        shutil.rmtree(tempdir)


def parse_attributes(xml_elem, parse_input):
    # elem - an XML parsing element containing an "attributes" member
    # parse_input - [["attribute_name", python_type_name], ["52", int], ["1.234", float], ["neurite", str], ...]
    # returns the list of python-typed values - [52, 1.234, "neurite", ...]
    parse_output = []
    attributes = xml_elem.attributes
    for x in parse_input:
        try:
            parse_output.append(x[1](attributes[x[0]].value))
        except KeyError:
            parse_output.append(None)
    return parse_output


def parse_cET(xml_elem, parse_input, ret_all_attr=False):
    """
    Implementation of cElementTree Parser

    Keyword arguments:
    xml_elem -- XML Element
    parse_input -- what the parser should look for ["whatyouarelookingfor", type]

    returns a list of python-typed values
    """
    parse_output = []
    already_parsed_input_keys = {}
    additional_attr = {}
    attributes = xml_elem.attrib
    for x in parse_input:
        already_parsed_input_keys[x[0]] = True
        try:
            parse_output.append(x[1](attributes[x[0]]))
        except KeyError:
                parse_output.append(None)
    if ret_all_attr:
        for key in attributes.keys():
            try:
                _ = already_parsed_input_keys[key]
            except KeyError:
                additional_attr[key] = attributes[key]
        return parse_output, additional_attr
    else:
        return parse_output


def build_attributes(xml_elem, attributes):
    for attr in attributes:
        try:
            xml_elem.setAttribute(attr[0], str(attr[1]))
        except UnicodeEncodeError:
            xml_elem.setAttribute(attr[0], str(attr[1].encode('ascii', 'replace')))
    return


def compare_version(v1, v2):
    """
    Return '>', '<' or '==', depending on whether the version
    represented by the iterable v1 is larger, smaller or equal
    to the version represented by the iterable v2.

    Versions are represented as iterables of integers, e.g.
    (3, 4, 2).
    """

    v1 = list(v1)
    v2 = list(v2)

    if len(v1) > len(v2):
        v2.extend([0] * (len(v1) - len(v2)))
    elif len(v1) < len(v2):
        v1.extend([0] * (len(v2) - len(v1)))

    v1v2 = [(v1[x], v2[x]) for (x, y) in enumerate(v1)]

    for (cur_v1, cur_v2) in v1v2:
        if cur_v1 == cur_v2:
            pass
        elif cur_v1 > cur_v2:
            return '>'
        else:
            return '<'

    return '=='


class Version(list):
    def __lt__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '<'

    def __le__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '<' or \
               compare_version(self, other) == '=='

    def __gt__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '>'

    def __ge__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '>' or \
               compare_version(self, other) == '=='

    def __eq__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '=='

    def __ne__(self, other):
        assert isinstance(other, Version)

        return not compare_version(self, other) == '=='


def integer_checksum(i):
    """Computes the SHA256 hash of an integer."""

    h = hashlib.sha256()
    h.update(struct.pack('i', i))

    return h.hexdigest()

def file_md5_checksum(f):
    """"Computes the MD5 hash for a file"""

    hasher = hashlib.md5()
    with open(f, 'r') as fp:
        r = fp.read()
    hasher.update(r)

    return hasher.hexdigest()