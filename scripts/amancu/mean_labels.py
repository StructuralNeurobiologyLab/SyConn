import glob
import numpy as np
from morphx.classes.hybridcloud import HybridCloud
import multiprocessing as mp

# cs_merge_radii = [100,500,1000,2000,5000]
radii = [1000, 2000]

def func(radius):
    hclouds_path = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl'

    # get all Hybridcloud files
    fnames = glob.glob(hclouds_path)

    print(f'Using radius {radius} and {len(fnames)} cell samples ')

    means = []

    #calculate mean of ptcloud
    hc = HybridCloud()
    for fname in fnames:
        hc.load_from_pkl(fname)
        # print(f'shape of labels {hc.labels.shape}')
        zeros = len(np.where(hc.labels==0)[0])
        ones = len(np.where(hc.labels==1)[0])
        ratio = zeros/ones
        means.append(ratio)


    avg = np.mean(means)
    print(f'For radius {radius} the mean of the labels is {avg}')


if __name__ == '__main__':
    running_tasks = [mp.Process(target=func, args=(radius,)) for radius in radii]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()