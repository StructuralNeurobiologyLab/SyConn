from filterLoader import FilterLoader
from tqdm import tqdm
from morphx.classes.hybridmesh import HybridCloud
import glob
import shutil
import os
from tqdm import tqdm

radii = [100, 500, 2000, 5000]

def func(radius):
    load = FilterLoader(radius=radius)

    for i in tqdm(range(len(load)), desc=f'Radius {radius}'):
        l = load[i]


for radius in radii:
    # print(f'Starting with radius {radius}')
    # func(radius)
    files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl')
    print(len(files))
    hc = HybridCloud()
    count = 0
    for file in tqdm(files,desc='Samples'):
        hc.load_from_pkl(file)
        length = len(hc.vertices)
        if length < 3500:
            print(f'Not enough points {length}. Moving... {file}')
            shutil.move(file, f'/wholebrain/scratch/amancu/mergeError/ptclouds/Dump/{radius}/' + os.path.basename(file))
            count += 1
    print(count)