import shutil
import glob
import os

radii = [100, 500, 1000, 2000, 5000]

for radius in radii:
    files = glob.glob(f'/wholebrain/scratch/amancu/mergeError/ptclouds/Dump/{radius}/*.pkl')
    for file in files:
        shutil.move(file, f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/' + os.path.basename(file))