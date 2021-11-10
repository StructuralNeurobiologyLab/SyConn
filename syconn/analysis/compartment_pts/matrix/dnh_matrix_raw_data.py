import pickle as pkl
import numpy as np


if __name__ == '__main__':
    with open(f'/wholebrain/scratch/jklimesch/paper/dnh_matrix_update_cmn_ads/dnh_matrix_syn.pkl', 'rb') as f:
        data = pkl.load(f)

    contexts = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    points = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    matrix = [[True, True, True, True, True, True, True, True, True, True],
              [True, True, True, True, True, True, True, True, True, True],
              [False, True, True, True, True, True, True, True, True, True],
              [False, False, True, True, True, True, True, True, True, True],
              [False, False, True, True, True, True, True, True, True, True],
              [False, False, False, False, True, True, True, True, True, True],
              [False, False, False, False, False, False, False, True, True, True]]

    scores = {}
    for points_ix in range(len(points)):
        for contexts_ix in range(len(contexts)):
            if matrix[points_ix][contexts_ix]:
                scores[str(contexts[contexts_ix]) + '_' + str(points[points_ix])] = np.array(data[f'{contexts[contexts_ix] * 1000}_{points[points_ix]}'])

    with open(f'/wholebrain/scratch/jklimesch/paper/dnh_matrix_update_cmn_ads/dnh_matrix_raw.txt', 'w') as f:
        for key in list(scores.keys()):
            f.write(key + ': \t' + str(scores[key]) + '\n')
