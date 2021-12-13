import shutil
import glob
import os
import tqdm

radii = [100, 500, 1000, 2000, 5000]

for radius in radii:
    # files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/ptclouds/Dump/{radius}/*.pkl')
    # for file in files:
    #     shutil.move(file, f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/' + os.pafor radius in cs_merge_radii:
    # files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl')
    # descr = f'for {radius}'
    # for file in tqdm.tqdm(files, desc=descr):
    #     shutil.move(file, f'/wholebrain/scratch/amancu/mergeError/ptclouds/OneCS/R{radius}/Hybridcloud/'
    files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/test_dataset/R{radius}/*.pkl')
    descr = f'for {radius}'
    for file in tqdm.tqdm(files, desc=descr):
        shutil.move(file, f'/wholebrain/scratch/amancu/mergeError/test_dataset/Old/R{radius}/' + os.path.basename(file))
