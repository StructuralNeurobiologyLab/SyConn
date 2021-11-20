import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle as pkl


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == '__main__':
    base_dir = '/wholebrain/scratch/pschuber/syconn_v2_paper/supplementals/' \
               'compartment_pts/dnh_matrix_update_cmn_ads/'
    pred_key = 'do_cmn_large'

    with open(f'{base_dir}/dnh_matrix_syn_e_final_{pred_key}.pkl', 'rb') as f:
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

    for points_ix in range(len(points)):
        for contexts_ix in range(len(contexts)):
            if matrix[points_ix][contexts_ix]:
                matrix[points_ix][contexts_ix] = np.array(data[f'{contexts[contexts_ix] * 1000}_{points[points_ix]}']).mean()
            else:
                matrix[points_ix][contexts_ix] = -1

    matrix = np.array(matrix)

    fig, ax = plt.subplots()
    im, cbar = heatmap(matrix, points, contexts, ax=ax, cmap="YlGn", cbarlabel="f1-score", vmin=0.83, vmax=0.98)


    def func(x, pos):
        return "{:.2f}".format(x).replace("-1.00", "")

    texts = annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
    fig.tight_layout()

    plt.savefig(f'{base_dir}/dnh_matrix_e_final_{pred_key}_YlGn_083_098.eps')

    fig, ax = plt.subplots()
    im, cbar = heatmap(matrix, points, contexts, ax=ax, cmap="YlGn", cbarlabel="f1-score", vmin=0.83, vmax=0.98)

    fig.tight_layout()

    plt.savefig(f'{base_dir}/dnh_matrix_e_final_{pred_key}_nolabels_YlGn_083_098.eps')