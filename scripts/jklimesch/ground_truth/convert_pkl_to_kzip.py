import numpy as np
import glob
import os
import tqdm
import re
from syconn.proc.meshes import write_mesh2kzip
from syconn.handler.basics import load_pkl2obj


""" Converts CloudEnsembles or HybridClouds saved in .pkl format to KNOSSOS .kzips. """


if __name__ == '__main__':
    source_dir = os.path.expanduser('~/thesis/gt/cmn/dnh/raw/')
    out_dir = os.path.expanduser('~/thesis/gt/cmn/dnh/raw/')
    fnames = glob.glob(source_dir + '/*.pkl')
    col_lookup = {0: (125, 125, 125, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255), 3: (125, 125, 255, 255),
                  4: (255, 255, 125, 255), 5: (125, 255, 255, 255), 6: (255, 125, 255, 255), 7: (0, 0, 0, 255),
                  8: (255, 0, 0, 255)}
    for fp in tqdm.tqdm(fnames, total=len(fnames)):
        curr_hc = load_pkl2obj(fp)['hybrid']
        fname = os.path.split(fp)[1][:-4]
        ssv_id = int(re.findall('sso_(\d+).pkl', fp)[0])
        cols = curr_hc['labels']
        assert len(cols) == len(curr_hc['vertices'])
        cols = np.array([col_lookup[el] for el in cols.squeeze()], dtype=np.uint8)
        write_mesh2kzip(f'{out_dir}/{fname}_mesh.k.zip', curr_hc['faces'].astype(np.float32),
                        curr_hc['vertices'].astype(np.float32), None, cols, f'{ssv_id}.ply')
