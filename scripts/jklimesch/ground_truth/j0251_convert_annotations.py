import numpy as np
import glob
import os
import re
from tqdm import tqdm
from morphx.classes.cloudensemble import CloudEnsemble
from syconn.proc.meshes import write_mesh2kzip
from morphx.classes.hybridcloud import HybridCloud
from utils import anno_skeleton2np, sso2kzip, nxGraph2kzip, map_myelin
from syconn.reps.super_segmentation import SuperSegmentationDataset


col_lookup = {0: (76, 92, 158, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255), 3: (113, 98, 227, 255),
              4: (255, 255, 125, 255), 5: (125, 255, 255, 255), 6: (255, 125, 255, 255), 7: (168, 0, 20, 255),
              8: (0, 97, 10, 255), 9: (255, 205, 130, 255), 10: (168, 0, 95, 255), 11: (191, 191, 191, 255),
              12: (255, 127, 15, 255)}


def process_file(file: str, o_path: str, ctype: str, ssd: SuperSegmentationDataset, convert_to_morphx: bool = False):
    sso_id = int(max(re.findall('(\d+)', file.replace(a_path, '')), key=len))

    kzip_path = o_path + f'{ctype}_{sso_id}.k.zip'
    if os.path.exists(kzip_path):
        print(f"{file.replace(o_path, '')} already processed.")
        return

    sso = ssd.get_super_segmentation_object(sso_id)
    scaling = sso.scaling
    # if 'DA_' in file or 'HVC_' in file or '10074977' in file:
    #     scaling = np.array([10, 10, 20])
    # if 'HVC_53854647' in file:
    #     scaling = sso.scaling
    a_coords, a_edges, a_labels, a_labels_raw, graph = anno_skeleton2np(file, scaling, verbose=True, convert_to_morphx=convert_to_morphx)

    indices, vertices, normals = sso.mesh
    vertices = vertices.reshape((-1, 3))
    labels = np.ones((len(vertices), 1)) * -1
    indices = indices.reshape((-1, 3))
    cell = HybridCloud(vertices=vertices, labels=labels, nodes=a_coords, edges=a_edges, node_labels=a_labels)
    # map labels from nodes to vertices
    cell.nodel2vertl()

    if not convert_to_morphx:
        # --- generate new colorings and save them to new kzips ---
        sso2kzip(sso_id, ssd, kzip_path, skeleton=False)
        cols = np.array([col_lookup[el] for el in cell.labels.squeeze()], dtype=np.uint8)
        write_mesh2kzip(kzip_path, indices.astype(np.float32), cell.vertices.astype(np.float32), None, cols, f'colored.ply')
        labels = [str(label) for label in a_labels_raw]
        nxGraph2kzip(graph, a_coords, labels, kzip_path)
    else:
        # --- convert annotations into MorphX CloudEnsembles ---
        # see comment2int in utils.py
        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3,
                    'terminal': 4, 'neck': 5, 'head': 6, 'nr': 7,
                    'in': 8, 'p': 9, 'st': 10, 'ignore': 11, 'merger': 12,
                    'pure_dendrite': 13, 'pure_axon': 14}

        # --- replace annotation skeleton with real one ---
        sso.load_skeleton()
        skel = sso.skeleton
        nodes = skel['nodes'] * sso.scaling
        edges = skel['edges']
        cell = HybridCloud(vertices=cell.vertices, labels=cell.labels, nodes=nodes, edges=edges, encoding=encoding)

        # --- prepare cell organelles ---
        organelles = ['mi', 'vc', 'sy']
        meshes = [sso.mi_mesh, sso.vc_mesh]
        meshes.append(sso._load_obj_mesh('syn_ssv'))
        label_map = [20, 21, 22]
        clouds = {}
        for ix, mesh in enumerate(meshes):
            indices, vertices, normals = mesh
            vertices = vertices.reshape((-1, 3))
            labels = np.ones((len(vertices), 1)) * label_map[ix]
            organelle = HybridCloud(vertices=vertices, labels=labels)
            organelle.set_encoding({organelles[ix]: label_map[ix]})
            clouds[organelles[ix]] = organelle

        # --- add myelin to main cell and merge main cell with organelles ---
        # hc = map_myelin(sso, hc)
        ce = CloudEnsemble(clouds, cell, no_pred=organelles)
        ce.save2pkl(f'{o_path}/sso_{sso.id}.pkl')


if __name__ == '__main__':
    a_path = '/wholebrain/u/jklimesch/working_dir/test/annotations/'
    o_path = '/wholebrain/u/jklimesch/working_dir/test/data/'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    ssd = SuperSegmentationDataset(working_dir='/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v2/')
    files = os.listdir(a_path)
    for file in files:
        print(f'Processing: {file}')
        if os.path.isdir(file):
            kzips = glob.glob(a_path + file + '/*k.zip')
            for kzip in tqdm(kzips):
                print(f'Processing: {kzip}')
                process_file(kzip, file, ssd)
        else:
            # set convert_to_morphx = False to only generate new colorings of kzips
            process_file(a_path + file, o_path, file[:3], ssd, convert_to_morphx=True)

