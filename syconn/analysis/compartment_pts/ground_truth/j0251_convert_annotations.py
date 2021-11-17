import numpy as np
import glob
import os
import re
from tqdm import tqdm
from scipy.spatial import cKDTree
from morphx.classes.cloudensemble import CloudEnsemble
from syconn.proc.meshes import write_mesh2kzip
from morphx.classes.hybridcloud import HybridCloud
from utils import anno_skeleton2np, sso2kzip, nxGraph2kzip, map_myelin
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.mp.mp_utils import start_multiprocess_imap
try:
    import open3d as o3d
except ImportError:
    pass  # for sphinx build

col_lookup = {0: (76, 92, 158, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255), 3: (113, 98, 227, 255),
              4: (255, 255, 125, 255), 5: (125, 255, 255, 255), 6: (255, 125, 255, 255), 7: (168, 0, 20, 255),
              8: (0, 97, 10, 255), 9: (255, 205, 130, 255), 10: (168, 0, 95, 255), 11: (191, 191, 191, 255),
              12: (255, 127, 15, 255)}


def voxelize_points(pts, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)


def _process_file(args):
    return process_file(*args)


def process_file(file: str, o_path: str, ctype: str, convert_to_morphx: bool = False):
    print(f'Processing: {file}')

    ssd = SuperSegmentationDataset(working_dir='/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/')
    # Point cloud reduction
    voxel_sizes = dict(sv=80, mi=1200, sy=1200, vc=1200)

    sso_id = int(max(re.findall('(\d+)', file.replace(a_path, '')), key=len))

    kzip_path = o_path + f'{ctype}_{sso_id}.k.zip'
    if os.path.exists(kzip_path):
        print(f"{file.replace(o_path, '')} already processed.")
        return

    sso = ssd.get_super_segmentation_object(sso_id)
    scaling = sso.scaling
    try:
        a_coords, a_edges, a_labels, a_labels_raw, graph, a_node_labels_orig = \
            anno_skeleton2np(file, scaling, verbose=False, convert_to_morphx=convert_to_morphx)
    except Exception as e:
        print(f'Could not load annotation file "{file}" due to error: {str(e)}')
        return
    indices, vertices, _ = sso.mesh
    vertices = vertices.reshape((-1, 3))
    # TODO: voxelize requires adaption of indices in the case of convert_to_morphx=False
    if convert_to_morphx:
        vertices = voxelize_points(vertices, voxel_size=voxel_sizes['sv'])
    labels = np.ones((len(vertices), 1)) * -1
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

        # now set skeleton nodes far away from manual annotations to be ignored during source node selection
        kdt = cKDTree(a_coords[a_node_labels_orig != -1])
        node_labels = np.ones((len(nodes), 1))
        dists, ixs = kdt.query(nodes, distance_upper_bound=2000)
        node_labels[dists == np.inf] = 0
        # set nodes that are ignored or merger to 0
        node_labels[a_node_labels_orig[ixs] == 11] = 0
        node_labels[a_node_labels_orig[ixs] == 12] = 0
        node_labels[a_node_labels_orig[ixs] == 13] = 0
        node_labels[a_node_labels_orig[ixs] == 14] = 0

        cell = HybridCloud(vertices=cell.vertices, labels=cell.labels, nodes=nodes, edges=edges, encoding=encoding,
                           node_labels=node_labels)

        # --- prepare cell organelles ---
        organelles = ['mi', 'vc', 'sy']
        meshes = [sso.mi_mesh, sso.vc_mesh]
        meshes.append(sso._load_obj_mesh('syn_ssv'))
        label_map = [20, 21, 22]
        clouds = {}
        for ix, mesh in enumerate(meshes):
            _, vertices, _ = mesh
            vertices = vertices.reshape((-1, 3))
            vertices = voxelize_points(vertices, voxel_size=voxel_sizes[organelles[ix]])
            labels = np.ones((len(vertices), 1)) * label_map[ix]
            organelle = HybridCloud(vertices=vertices, labels=labels)
            organelle.set_encoding({organelles[ix]: label_map[ix]})
            clouds[organelles[ix]] = organelle
        # --- add myelin to main cell and merge main cell with organelles ---
        cell = map_myelin(sso, cell)
        ce = CloudEnsemble(clouds, cell, no_pred=organelles)
        ce.save2pkl(f'{o_path}/sso_{sso.id}.pkl')


if __name__ == '__main__':
    # set convert_to_morphx = False to only generate new colorings of kzips
    convert_to_morphx = False
    a_path = '/wholebrain/scratch/pschuber/tmp/to_colorize/'
    o_path = '/wholebrain/scratch/pschuber/tmp/colored//'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    files = os.listdir(a_path)
    args = []
    for file in files:
        if os.path.isdir(file):
            kzips = glob.glob(a_path + file + '/*k.zip')
            for kzip in tqdm(kzips):
                print(f'Processing: {kzip}')
                args.append([kzip, o_path, file[:3]])
        else:
            args.append([a_path + file, o_path, file[:3], convert_to_morphx])

    start_multiprocess_imap(_process_file, args, nb_cpus=10, debug=False)

