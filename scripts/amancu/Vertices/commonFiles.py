'''
Because of non-deterministic (as a result of multiprocessing) sample generations, we have to delete files that are not found in all cs_merge_radii folders.
Thus, Training and Test Data will be the same for all trainings.

Max Planck Institute of Neurobiology, Munich, Germany
Author: Andrei Mancu
'''

import os
import glob
import shutil
from tqdm import tqdm

radii = [100, 500, 1000, 2000, 5000]


def dir_path(radius):
    return f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/'


def findCommonDeep(path1, path2):
    '''
    Returns: Intersection of the files in 2 folders
    '''
    return set.intersection(
        *(set(os.path.relpath(os.path.join(root, file), path) for root, _, files in os.walk(path) for file in files) for
          path in (path1, path2)))


if __name__ == '__main__':
    files100_500 = findCommonDeep(dir_path(100), dir_path(500))
    files1000_2000 = findCommonDeep(dir_path(1000), dir_path(2000))
    files5000 = set([os.path.basename(x) for x in
                     glob.glob('/wholebrain/scratch/amancu/mergeError/ptclouds/R5000/Hybridcloud/*.pkl')])

    common_files = files100_500.intersection(files1000_2000)
    common_files = common_files.intersection(files5000)
    print(len(common_files))

    # filter files
    for radius in radii:
        keep_inside = 0
        pathname = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl'
        all_files = glob.glob(pathname)
        print(f'Found {len(all_files)} files in radius {radius}')

        for file in tqdm(all_files):
            if os.path.basename(file) not in common_files:
                shutil.move(file,
                            f'/wholebrain/scratch/amancu/mergeError/ptclouds/noCommon/{radius}/' + os.path.basename(
                                file))
            else:
                keep_inside += 1

        print(f'For Radius {radius} {keep_inside} files were kept')
