import os
import glob
import numpy as np
from typing import Union
from tqdm import tqdm
import networkx as nx
from .utils import label_search
from syconn.reps.super_segmentation_object import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property
from syconn import global_params
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridmesh import HybridMesh
from morphx.processing.basics import load_pkl
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset


def convert_training_gt(ssv_input: str, label_input: str, out_train: str, out_valid: str):
    ssv_input = load_pkl(os.path.expanduser(ssv_input))
    label_input = load_pkl(os.path.expanduser(label_input))
    out_train = os.path.expanduser(out_train)
    out_valid = os.path.expanduser(out_valid)
    if not os.path.exists(out_train):
        os.makedirs(out_train)
    if not os.path.exists(out_valid):
        os.makedirs(out_valid)

    ssd = SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/", version='axgt')
    # convert training gt
    for item in tqdm(ssv_input['train']):
        label = label_input[item]
        sso = ssd.get_super_segmentation_object(item)
        ce = convert_single(sso, label)
        ce.save2pkl(f'{out_train}/sso_{item}.pkl')

    # convert validation gt
    for item in tqdm(ssv_input['valid']):
        label = label_input[item]
        sso = ssd.get_super_segmentation_object(item)
        ce = convert_single(sso, label)
        ce.save2pkl(f'{out_valid}/sso_{item}.pkl')


def convert_test_gt(gt_path: str, out_path: str):
    gt_path = os.path.expanduser(gt_path)
    out_path = os.path.expanduser(out_path)
    files = glob.glob(gt_path + '*.pkl')
    global_params.wd = "/wholebrain/scratch/areaxfs3/"
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        sso_id = int(file[slashs[-1] + 1:-4])
        sso = SuperSegmentationObject(sso_id)
        gt = load_pkl(file)
        label = np.array(gt['compartment_label'])
        ce = convert_single(sso, label, train=False)
        ce.save2pkl(f'{out_path}/sso_{sso_id}.pkl')


def convert_single(sso: SuperSegmentationObject, label: Union[int, np.ndarray], train: bool = True) -> CloudEnsemble:
    # load cell and cell organelles
    meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh, sso.sj_mesh]
    label_map = [-1, 7, 8, 9]
    hms = []
    for ix, mesh in enumerate(meshes):
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        labels = np.ones((len(vertices), 1)) * label_map[ix]
        indices = indices.reshape((-1, 3))
        hm = HybridMesh(vertices=vertices, faces=indices, labels=labels)
        hms.append(hm)

    # load skeleton
    sso.load_skeleton()
    skel = sso.skeleton
    nodes = skel['nodes'] * sso.scaling
    edges = skel['edges']

    if not train:
        g = nx.Graph()
        g.add_nodes_from([(i, dict(label=label[i])) for i in range(len(nodes))])
        g.add_edges_from([(edges[i][0], edges[i][1]) for i in range(len(edges))])
        for node in g.nodes:
            if g.nodes[node]['label'] == -1:
                ix = label_search(g, node)
                label[node] = label[ix]

    # create cloud ensemble
    encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
    obj_names = ['hc', 'mi', 'vc', 'sy']
    hm = None
    clouds = {}
    for ix, cloud in enumerate(hms):
        if ix == 0:
            vertices = hms[0].vertices
            if train:
                hm = HybridMesh(vertices=vertices, labels=np.ones(len(vertices))*label, faces=hms[0].faces, nodes=nodes,
                                edges=edges, encoding=encoding)
            else:
                hm = HybridMesh(vertices=vertices, node_labels=label, faces=hms[0].faces, nodes=nodes,
                                edges=edges, encoding=encoding)
                hm.nodel2vertl()
        else:
            hms[ix].set_encoding({obj_names[ix]: label_map[ix]})
            clouds[obj_names[ix]] = hms[ix]
    ce = CloudEnsemble(clouds, hm, no_pred=['mi', 'vc', 'sy'])

    # add myelin (see docstring of map_myelin2coords)
    sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    majorityvote_skeleton_property(sso, 'myelin')
    myelinated = sso.skeleton['myelin_avg10000']
    nodes_idcs = np.arange(len(hm.nodes))
    myel_nodes = nodes_idcs[myelinated.astype(bool)]
    myel_vertices = []
    for node in myel_nodes:
        myel_vertices.extend(hm.verts2node[node])
    # myelinated vertices get type 1, not myelinated vertices get type 0
    types = np.zeros(len(hm.vertices))
    types[myel_vertices] = 1
    hm.set_types(types)
    return ce


if __name__ == "__main__":
    # ssv_path = '~/thesis/gt/cmn/annotations/axgt_splitting.pkl'
    # label_path = '~/thesis/gt/cmn/annotations/axgt_labels.pkl'
    # train_path = '~/thesis/gt/cmn/train/'
    # val_path = '~/thesis/gt/cmn/validate/'
    # convert_training_gt(ssv_path, label_path, train_path, val_path)

    test_gt = '~/thesis/gt/cmn/annotations/test/'
    test_out = '~/thesis/gt/cmn/test/raw/'
    convert_test_gt(test_gt, test_out)
