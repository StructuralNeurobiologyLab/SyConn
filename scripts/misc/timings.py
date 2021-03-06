from syconn.proc.stats import FileTimer
from syconn.handler.config import initialize_logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import glob
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
import statsmodels.api as sm


palette_ident = 'colorblind'


def get_speed_plots():
    sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
    wds = glob.glob('/mnt/example_runs/j0251_*')
    base_dir = '/mnt/example_runs/timings/'
    log = initialize_logging(f'speed_plots', log_dir=base_dir)
    os.makedirs(base_dir, exist_ok=True)
    res_dc = {'time': [], 'step': [], 'datasize[mm3]': [], 'datasize[GVx]': [],
              'speed[mm3]': [], 'speed[GVx]': []}

    for wd in sorted(wds, key=lambda x: FileTimer(x).dataset_mm3):
        ft = FileTimer(wd, add_detail_vols=True)
        log.info(f'\n-----------------------------------\nLoading time data of "{ft.working_dir}"')
        log.info(f'{ft.prepare_report()}')
        # no reasonable volume information for these steps:
        for name in ['Preparation', 'Matrix export', 'Spine head calculation', 'Glia splitting']:
            del ft.timings[name]
        for name, dt in ft.timings.items():
            dt = dt / 3600
            res_dc['time'].append(dt)
            res_dc['step'].append(name)
            res_dc['datasize[mm3]'].append(ft.dataset_mm3['cube'])
            res_dc['datasize[GVx]'].append(ft.dataset_nvoxels['cube'])
            # use actually processed volums (e.g. all for synapses, glia-free rag for cell type inference)
            if 'glia' in name.lower():
                vol_mm3 = ft.dataset_mm3['neuron'] + ft.dataset_mm3['glia']
                vol_nvox = ft.dataset_nvoxels['neuron'] + ft.dataset_nvoxels['glia']
            elif name in ['SD generation', 'Synapse detection', 'Skeleton generation']:
                vol_mm3 = ft.dataset_mm3['cube']
                vol_nvox = ft.dataset_nvoxels['cube']
            else:
                vol_mm3 = ft.dataset_mm3['neuron']
                vol_nvox = ft.dataset_nvoxels['neuron']
            res_dc['speed[mm3]'].append(vol_mm3 / dt)
            res_dc['speed[GVx]'].append(vol_nvox / dt)
    assert len(wds) > 0
    palette = sns.color_palette(n_colors=len(np.unique(res_dc['step'])), palette=palette_ident)
    palette = {k: v for k, v in zip(np.unique(res_dc['step']), palette)}
    df = pd.DataFrame(data=res_dc)
    df.to_csv(f'{base_dir}/speed_data.csv')
    fmt = '{:0.2f}'
    # Speed bar plot
    plt.figure()
    axes = sns.barplot(data=df, x="datasize[GVx]", y="speed[GVx]", hue="step", palette=palette)
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('speed [GVx / h]')
    axes.set_xlabel('size [GVx]')
    xticklabels = []
    for item in axes.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    axes.set_xticklabels(xticklabels)
    plt.subplots_adjust(right=0.5)
    plt.savefig(base_dir + '/speed_barplot.png')
    plt.close()

    # Speed scatter plot regression
    log_reg = initialize_logging(f'speed_pointplot_reg', log_dir=base_dir)
    plt.figure()
    axes = sns.scatterplot(data=df, x="datasize[GVx]", y="speed[GVx]", hue="step", palette=palette)
    for ii, step in enumerate(np.unique(res_dc['step'])):
        x = np.array([df['datasize[GVx]'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == step])
        y = [df['speed[GVx]'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == step]

        # mod = sm.OLS(np.log(y), sm.add_constant(x), weights=np.sqrt(y))  # weight large y s.t. they are equally
        # # weighted to small values. discrepancy due to log() transform.
        # res = mod.fit()
        # log.info(res.summary())
        # x_fit = np.linspace(np.min(x), np.max(x), 1000)
        # y_fit = np.exp(res.params[1] * x_fit + res.params[0])

        mod = sm.OLS(y, sm.add_constant(x))
        res = mod.fit()
        log_reg.info(f'Fit summary for step "{step}"')
        log_reg.info(res.summary())
        x_fit = np.linspace(np.min(x), np.max(x), 1000)
        y_fit = res.params[1] * x_fit + res.params[0]
        plt.plot(x_fit, y_fit, color=palette[step])
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('speed [GVx / h]')
    axes.set_xlabel('size [GVx]')
    plt.subplots_adjust(right=0.5)
    plt.savefig(base_dir + '/speed_pointplot_reg.png')
    plt.close()

    # Speed scatter plot
    plt.figure()
    axes = sns.scatterplot(data=df, x="datasize[GVx]", y="speed[GVx]", hue="step", palette=palette)
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('speed [GVx / h]')
    axes.set_xlabel('size [GVx]')
    plt.subplots_adjust(right=0.5)
    plt.savefig(base_dir + '/speed_pointplot.png')
    plt.close()


def get_timing_plots():
    sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
    wds = glob.glob('/mnt/example_runs/j0251_*')
    base_dir = '/mnt/example_runs/timings/'
    log = initialize_logging(f'time_plots', log_dir=base_dir)
    os.makedirs(base_dir, exist_ok=True)
    res_dc = {'time': [], 'time_rel': [], 'step': [], 'datasize[mm3]': [], 'datasize[GVx]': []}
    high_level_res_dc = defaultdict(list)
    # /mnt/example_runs/j0251_off9463_9579_5699_size8192_8192_4096_24nodes_run2 is currently the only WD with
    # probably reasonably outcome; PS 23Sep2020

    for wd in sorted(wds, key=lambda x: FileTimer(x).dataset_mm3):
        ft = FileTimer(wd, add_detail_vols=False)
        log.info(f'\n-----------------------------------\nLoading time data of "{ft.working_dir}"')
        log.info(f'{ft.prepare_report()}')
        dt_tot = np.sum([ft.timings[k] for k in ft.timings if not ('multiv-view' in k) and not ('multi-view' in k)])
        dt_views = np.sum([ft.timings[k] for k in ft.timings if ('multiv-view' in k) or ('multi-view' in k) or
                           (k == 'Glia splitting')])
        dt_points = np.sum([ft.timings[k] for k in ft.timings if ('points' in k) or (k == 'Glia splitting')])
        dt_database = np.sum(
            [ft.timings[k] for k in ['SD generation', 'SSD generation', 'Preparation', 'Skeleton generation']])
        dt_syns = np.sum([ft.timings[k] for k in ['Synapse detection']])
        dt_syn_enrich = np.sum([ft.timings[k] for k in ['Spine head calculation', 'Matrix export']])
        assert np.isclose(dt_tot, dt_points + dt_database + dt_syns + dt_syn_enrich)
        # gigavoxels per h; excluding views
        high_level_res_dc['speed_total_nvox[h/GVx]'].append(ft.dataset_nvoxels / dt_tot * 3600)
        high_level_res_dc['datasize [GVx]'].append(ft.dataset_nvoxels)  # in giga voxels
        high_level_res_dc['datasize [mm3]'].append(ft.dataset_mm3)
        high_level_res_dc['total_time [h]'].append(dt_tot / 3600)  # excluding views
        high_level_res_dc['dt_views [h]'].append(dt_views / 3600)  # in h
        high_level_res_dc['dt_points [h]'].append(dt_points / 3600)  # in h
        high_level_res_dc['dt_points_over_views'].append(dt_points / dt_views)
        for name, dt in [('total', dt_tot), ('views', dt_views), ('points', dt_points),
                         ('data structure', dt_database), ('synapses', dt_syns), ('synapse enrichment', dt_syn_enrich)]:
            dt = dt / 3600
            res_dc['time'].append(dt)
            res_dc['time_rel'].append(dt / dt_tot * 100)
            res_dc['step'].append(name)
            res_dc['datasize[mm3]'].append(ft.dataset_mm3)
            res_dc['datasize[GVx]'].append(ft.dataset_nvoxels)
    assert len(wds) > 0
    palette = sns.color_palette(n_colors=len(np.unique(res_dc['step'])), palette=palette_ident)
    palette = {k: v for k, v in zip(np.unique(res_dc['step']), palette)}
    df_highlevel = pd.DataFrame(data=high_level_res_dc)
    df_highlevel.to_csv(f'{base_dir}/data_timings_highlevel.csv')
    df = pd.DataFrame(data=res_dc)
    df.to_csv(f'{base_dir}/time_data.csv')
    fmt = '{:0.2f}'

    # Time bar plot
    plt.figure()
    axes = sns.barplot(data=df, x="datasize[GVx]", y="time", hue="step", palette=palette)
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('Time [h]')
    axes.set_xlabel('size [GVx]')
    xticklabels = []
    for item in axes.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    axes.set_xticklabels(xticklabels)
    plt.subplots_adjust(right=0.75)
    plt.savefig(base_dir + '/time_barplot.png')
    plt.close()

    # Time scatter plot
    plt.figure()
    axes = sns.scatterplot(data=df, x="datasize[GVx]", y="time", hue="step", palette=palette)
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('time [h]')
    axes.set_xlabel('size [GVx]')
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    plt.subplots_adjust(right=0.75)
    plt.savefig(base_dir + '/time_pointplot.png')
    plt.close()

    # # Total time regression plot
    # x = [df['datasize[GVx]'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == 'total']
    # y = [df['time'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == 'total']
    # mod = sm.OLS(y, sm.add_constant(x))
    # res = mod.fit()
    # log.info(res.summary())
    # x_fit = np.linspace(np.min(x), np.max(x), 1000)
    # y_fit = res.params[1] * x_fit + res.params[0]
    # plt.figure()
    # axes = sns.scatterplot(x=x, y=y, palette=palette)
    # axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
    #             loc='upper left', borderaxespad=0.)
    # plt.plot(x_fit, y_fit)
    # axes.set_ylabel('time [h]')
    # axes.set_xlabel('size [GVx]')
    # plt.subplots_adjust(right=0.75)
    # plt.savefig(base_dir + '/totaltime_regplot.png')
    # plt.close()

    # Time reg plot
    # # https://seaborn.pydata.org/generated/seaborn.regplot.html
    # regplot does not return the parameter values after fitting and the "Guete" of the fit.
    # plt.figure()
    # g = sns.FacetGrid(df, hue='step', palette=palette_ident, size=5)
    # g.map(sns.regplot, "datasize[GVx]", "time", ci=None, robust=1)
    # g.map(plt.scatter, "datasize[GVx]", "time", s=40, alpha=.7, linewidth=.5, edgecolor=None)
    # g.add_legend()
    # g.set_ylabels('time [h]')
    # g.set_xlabels('size [GVx]')
    # g.set(ylim=ylim)
    # g.set(xlim=xlim)
    # plt.savefig(base_dir + '/time_regplot.png')
    # plt.close()

    # All steps time regression plot
    log_reg = initialize_logging(f'time_allsteps_regplot', log_dir=base_dir)
    plt.figure()
    axes = sns.scatterplot(data=df, x="datasize[GVx]", y="time", hue="step", palette=palette)
    for ii, step in enumerate(np.unique(res_dc['step'])):
        x = np.array([df['datasize[GVx]'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == step])
        y = [df['time'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == step]
        mod = sm.OLS(y, sm.add_constant(x))
        res = mod.fit()
        log_reg.info(f'Fit summary for step "{step}"')
        log_reg.info(f'\n{res.summary()}\n\n')
        x_fit = np.linspace(np.min(x), np.max(x), 1000)
        y_fit = res.params[1] * x_fit + res.params[0]
        plt.plot(x_fit, y_fit, color=palette[step])
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('time [h]')
    axes.set_xlabel('size [GVx]')
    plt.subplots_adjust(right=0.75)
    plt.savefig(base_dir + '/timing_allsteps_regplot.png')
    plt.close()

    # stacked bar plot
    steps = ['points', 'data structure', 'synapses', 'synapse enrichment']  # ['views']
    f, ax = plt.subplots()
    bar_plts = []
    x = [df['datasize[GVx]'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == 'total']
    ind = np.arange(len(x))
    width = 0.35
    cumulated_bar_vals = np.zeros((len(x)))
    for ii, step in enumerate(steps):
        y = np.array([df['time'][ii] for ii in range(len(df['datasize[GVx]'])) if df['step'][ii] == step])
        p = plt.bar(ind, y, width, bottom=cumulated_bar_vals, color=palette[step], linewidth=0)  # yerr=None
        cumulated_bar_vals += y
        bar_plts.append(p[0])
    plt.legend(bar_plts, steps)
    plt.xticks(ind, [fmt.format(el) for el in x])
    ax.set_ylabel('Time [h]')
    ax.set_xlabel('size [GVx]')
    plt.subplots_adjust(right=0.75)
    plt.savefig(base_dir + '/time_stackedbarplot.png')
    plt.close()


if __name__ == '__main__':
    get_timing_plots()
    get_speed_plots()

