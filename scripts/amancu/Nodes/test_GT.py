import shutil
import glob
import os
import tqdm
from typing import Callable
from morphx.classes.hybridmesh import HybridCloud
from syconn.handler.prediction_pts import pts_loader_semseg_train, pts_loader_semseg_train_nodes


files = glob.glob(os.path.expanduser(
    f'/wholebrain/scratch/amancu/mergeError/Nodes/TrainingGT/R3000_downsample300/*.pkl'))
train_limit = int(0.8 * (len(files)))
files=files[train_limit:]
hc = HybridCloud()
for i, file in enumerate(files):
    try:
        sample_feats, sample_pts, out_pts, out_labels = \
            [*pts_loader_semseg_train_nodes([file], 4, 20000,
                ctx_size=20000, use_subcell=False,
                gt_type='merger', regression=True)][0]
    except Exception as e:
        print(f'[Error] {e} for: file {os.path.basename(file)}, moving...')
        # shutil.move(file, f'/wholebrain/scratch/amancu/mergeError/Nodes/FailedGT/' + os.path.basename(file))
    