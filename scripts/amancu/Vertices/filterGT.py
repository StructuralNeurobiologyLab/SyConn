import shutil
import glob
import os
import tqdm
from morphx.classes.hybridmesh import HybridCloud

radii = [100, 500, 5000]

for radius in radii:
    files = glob.glob(os.path.expanduser(
        f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{int(radius)}/Hybridcloud/*.pkl'))
        # f'/wholebrain/scratch/amancu/mergeError/test_dataset/R{int(radius)}/*.pkl'))
    hc = HybridCloud()
    des = f'For {radius}'
    for file in tqdm.tqdm(files, desc=des):
        hc.load_from_pkl(file)
        if len(hc.vertices) < 3200:
            shutil.move(file,
                        f'/wholebrain/scratch/amancu/mergeError/ptclouds/Dump/{radius}/' + os.path.basename(file))