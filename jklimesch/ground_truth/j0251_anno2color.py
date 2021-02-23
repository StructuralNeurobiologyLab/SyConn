import numpy as np
import glob
import os
import re
from tqdm import tqdm
from syconn.proc.meshes import write_mesh2kzip
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud
from utils import anno_skeleton2np
from syconn.reps.super_segmentation import SuperSegmentationDataset


col_lookup = {0: (76, 92, 158, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255), 3: (113, 98, 227, 255),
              4: (255, 255, 125, 255), 5: (125, 255, 255, 255), 6: (255, 125, 255, 255), 7: (168, 0, 20, 255),
              8: (0, 97, 10, 255), 9: (255, 205, 130, 255), 10: (168, 0, 95, 255), 11: (191, 191, 191, 255),
              12: (255, 127, 15, 255)}


def kzip2pkl(sso_id, ssd, kzip):
    ssd = SuperSegmentationDataset(working_dir=ssd)
    sso = ssd.get_super_segmentation_object(sso_id)

    scaling = sso.scaling
    if 'DA_' in kzip or 'HVC_' in kzip or '6000389' in kzip or '1358090' in kzip or '10074977' in kzip:
        scaling = np.array([10, 10, 20])
    if 'HVC_53854647' in kzip:
        scaling = sso.scaling
    a_coords, a_edges, a_labels = anno_skeleton2np(kzip, scaling, verbose=True)

    indices, vertices, normals = sso.mesh
    vertices = vertices.reshape((-1, 3))
    labels = np.ones((len(vertices), 1)) * -1
    indices = indices.reshape((-1, 3))
    hc = HybridCloud(vertices=vertices, labels=labels, nodes=a_coords, edges=a_edges,
                     node_labels=a_labels)
    hc.nodel2vertl()
    return hc, indices


def process_file(file: str, ctype: str):
    sso_id = int(max(re.findall('(\d+)', file.replace(a_path, '')), key=len))
    hc, faces = kzip2pkl(sso_id, ssd, file)
    os.rename(file, o_path + ctype + f'_{sso_id}.k.zip')
    cols = np.array([col_lookup[el] for el in hc.labels.squeeze()], dtype=np.uint8)
    write_mesh2kzip(f'{o_path}{ctype}_{sso_id}.k.zip', faces.astype(np.float32),
                    hc.vertices.astype(np.float32), None, cols, f'colored.ply')
    hc.save2pkl(o_path + ctype + f'_{sso_id}.pkl')


if __name__ == '__main__':
    a_path = '/wholebrain/u/jklimesch/working_dir/gt/j0251/21_02_02_annotations/raw/'
    o_path = '/wholebrain/u/jklimesch/working_dir/gt/j0251/21_02_02_annotations/colored/'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    ssd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v2/'
    files = os.listdir(a_path)
    for file in files:
        print(f'Processing: {file}')
        if os.path.isdir(file):
            kzips = glob.glob(a_path + file + '/*k.zip')
            for kzip in tqdm(kzips):
                print(f'Processing: {kzip}')
                process_file(kzip, file)
        else:
            process_file(a_path + file, file[:3])

