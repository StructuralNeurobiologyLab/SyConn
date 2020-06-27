import time
from syconn import global_params
from syconn.exec import exec_inference

if __name__ == '__main__':
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"

    global_params.config['use_point_models'] = True
    global_params.config['ncores_per_node'] = 10
    global_params.config['ngpus_per_node'] = 1
    global_params.config['nnodes_total'] = 1
    global_params.config['log_level'] = 'DEBUG'
    global_params.config['batch_proc_system'] = None

    start = time.time()
    exec_inference.run_semsegaxoness_prediction()
    print(time.time()-start)
