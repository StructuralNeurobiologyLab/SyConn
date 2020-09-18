from syconn.proc.stats import FileTimer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob

if __name__ == '__main__':
    base_dir = '/mnt/example_runs/'
    wds = glob.glob(base_dir + '/j0251_*')
    res_dc = {'time': [], 'time_rel': [], 'step': [], 'datasize[um3]': [], 'datasize[GVx]': [],
              'speed[um3]': [], 'speed[GVx]': []}
    for wd in wds:
        if wd == '/mnt/example_runs/j0251_off9463_9579_5699_size8192_8192_4096_24nodes':
            continue
        ft = FileTimer(wd)
        dt_tot = np.sum(list(ft.timings.values()))
        dt_views = np.sum([ft.timings[k] for k in ft.timings if ('multiv-view' in k) or ('multi-view' in k)])
        dt_points = np.sum([ft.timings[k] for k in ft.timings if 'points' in k])
        dt_database = np.sum([ft.timings[k] for k in ['SD generation', 'Glia splitting', 'SSD generation', 'Preparation']])
        dt_syns = np.sum([ft.timings[k] for k in ['Synapse detection', 'Spine head calculation', 'Matrix export']])
        dt_skels = np.sum([ft.timings[k] for k in ['Skeleton generation']])
        assert np.isclose(dt_tot, dt_views + dt_points + dt_database + dt_syns + dt_skels)
        for name, dt in [('total', dt_tot), ('views', dt_views), ('points', dt_points),
                         ('database', dt_database), ('synapses', dt_syns), ('skeletons', dt_skels)]:
            dt = dt / 3600
            res_dc['time'].append(dt)
            res_dc['time_rel'].append(dt / dt_tot * 100)
            res_dc['step'].append(name)
            res_dc['datasize[um3]'].append(ft.dataset_mm3)
            res_dc['datasize[GVx]'].append(ft.dataset_nvoxels)
            # TODO: use actually processed volums (e.g. all for synapses, glia-free rag for cell type inference)
            res_dc['speed[um3]'].append(ft.dataset_mm3 / dt)
            res_dc['speed[GVx]'].append(ft.dataset_nvoxels / dt)
    df = pd.DataFrame(data=res_dc)
    plt.figure()
    axes = sns.barplot(data=df, x="datasize[GVx]", y="speed[GVx]", hue="step")
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('speed [GVx / h]')
    # Adjust
    plt.subplots_adjust(right=0.75)
    plt.savefig(base_dir + '/speed_barplot.png')
    plt.close()

    plt.figure()
    axes = sns.barplot(data=df, x="datasize[GVx]", y="time", hue="step")
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('speed [GVx / h]')
    # Adjust
    plt.subplots_adjust(right=0.75)
    plt.savefig(base_dir + '/time_barplot.png')
    plt.close()

    plt.figure()
    axes = sns.scatterplot(data=df, x="datasize[GVx]", y="time", hue="step")
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    # Adjust
    plt.subplots_adjust(right=0.75)
    plt.savefig(base_dir + '/time_pointplot.png')
    plt.close()
