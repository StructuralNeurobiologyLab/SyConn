import numpy as np
import glob
import os
import re
from tqdm import tqdm
from syconn.proc.meshes import write_mesh2kzip
from morphx.classes.hybridcloud import HybridCloud
from utils import anno_skeleton2np, comment2unique, unique2comment, sso2kzip, nxGraph2kzip
from syconn.reps.super_segmentation import SuperSegmentationDataset


col_lookup = {0: (76, 92, 158, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255), 3: (113, 98, 227, 255),
              4: (255, 255, 125, 255), 5: (125, 255, 255, 255), 6: (255, 125, 255, 255), 7: (168, 0, 20, 255),
              8: (0, 97, 10, 255), 9: (255, 205, 130, 255), 10: (168, 0, 95, 255), 11: (191, 191, 191, 255),
              12: (255, 127, 15, 255)}


def process_file(file: str, ctype: str, ssd: SuperSegmentationDataset):
    sso_id = int(max(re.findall('(\d+)', file.replace(a_path, '')), key=len))
    sso = ssd.get_super_segmentation_object(sso_id)
    scaling = sso.scaling
    if 'DA_' in file or 'HVC_' in file or '6000389' in file or '1358090' in file or '10074977' in file:
        scaling = np.array([10, 10, 20])
    if 'HVC_53854647' in file:
        scaling = sso.scaling
    a_coords, a_edges, a_labels, a_labels_raw, graph = anno_skeleton2np(file, scaling, verbose=True)

    indices, vertices, normals = sso.mesh
    vertices = vertices.reshape((-1, 3))
    labels = np.ones((len(vertices), 1)) * -1
    indices = indices.reshape((-1, 3))
    hc = HybridCloud(vertices=vertices, labels=labels, nodes=a_coords, edges=a_edges, node_labels=a_labels)
    hc.nodel2vertl()

    output_path = o_path + ('DA' if ctype == 'DA_' else ctype) + f'_{sso_id}.k.zip'
    sso2kzip(sso_id, ssd, output_path, skeleton=False)
    cols = np.array([col_lookup[el] for el in hc.labels.squeeze()], dtype=np.uint8)
    write_mesh2kzip(output_path, indices.astype(np.float32), hc.vertices.astype(np.float32), None, cols, f'colored.ply')
    labels = [unique2comment(int(label)) for label in a_labels_raw]
    nxGraph2kzip(graph, a_coords, labels, output_path)
    # hc.save2pkl(o_path + ('DA' if ctype == 'DA_' else ctype) + f'_{sso_id}.pkl')


if __name__ == '__main__':
    a_path = '/wholebrain/u/jklimesch/working_dir/gt/j0251/21_03_13_annotations_refinment_round1/raw/'
    o_path = '/wholebrain/u/jklimesch/working_dir/gt/j0251/21_03_13_annotations_refinment_round1/colored/'
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
            process_file(a_path + file, file[:3], ssd)

