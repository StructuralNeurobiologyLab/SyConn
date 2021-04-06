from syconn.proc.stats import FileTimer
from syconn.handler.config import initialize_logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import glob
import re
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
import statsmodels.api as sm


palette_ident = 'colorblind'
scatter_size = None


def adapt_ax_params(ax, ls=6):
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10, rotation=45)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def get_speed_plots(base_dir):
    # sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
    wds = glob.glob(f'{base_dir}/j0251_*')
    assert len(wds) > 0
    base_dir = base_dir + '/timings/'
    log = initialize_logging(f'speed_plots', log_dir=base_dir)
    log.info(f'Creating speed plots in base directory "{base_dir}".')
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
    adapt_ax_params(axes)
    plt.subplots_adjust(right=0.5)
    plt.savefig(base_dir + '/speed_barplot.png', dpi=600)
    plt.close()

    # Speed bar plot - only biggest data set
    plt.figure()
    df_biggest = df.loc[lambda df: df['datasize[GVx]'] == df['datasize[GVx]'].max(), :]
    axes = sns.barplot(data=df_biggest, x="step", y="speed[GVx]", palette=palette)
    axes.set_ylabel('speed [GVx / h]')
    axes.set_xlabel('step')
    adapt_ax_params(axes)
    plt.tight_layout()
    plt.savefig(base_dir + '/speed_barplot_biggest_only.png', dpi=600)
    plt.close()

    # Speed scatter plot regression
    log_reg = initialize_logging(f'speed_pointplot_reg', log_dir=base_dir)
    plt.figure()
    axes = sns.scatterplot(data=df, x="datasize[GVx]", y="speed[GVx]", hue="step", palette=palette,
                           size=scatter_size)
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
    adapt_ax_params(axes)
    plt.subplots_adjust(right=0.5)
    plt.savefig(base_dir + '/speed_pointplot_reg.png', dpi=600)
    plt.close()

    # Speed scatter plot
    plt.figure()
    axes = sns.scatterplot(data=df, x="datasize[GVx]", y="speed[GVx]", hue="step", palette=palette,
                           size=scatter_size)
    axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
    axes.set_ylabel('speed [GVx / h]')
    axes.set_xlabel('size [GVx]')
    adapt_ax_params(axes)
    plt.subplots_adjust(right=0.5)
    plt.savefig(base_dir + '/speed_pointplot.png', dpi=600)
    plt.close()


def get_timing_plots(base_dir):
    sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
    wds = glob.glob(base_dir + '/j0251_*')
    base_dir = base_dir + '/timings/'
    log = initialize_logging(f'time_plots', log_dir=base_dir)
    log.info(f'Creating timing plots in base directory "{base_dir}".')
    os.makedirs(base_dir, exist_ok=True)
    res_dc = {'time': [], 'time_rel': [], 'step': [], 'datasize[mm3]': [], 'datasize[GVx]': [], 'n_compute_nodes': []}
    high_level_res_dc = defaultdict(list)
    # /mnt/example_runs/j0251_off9463_9579_5699_size8192_8192_4096_24nodes_run2 is currently the only WD with
    # probably reasonably outcome; PS 23Sep2020
    n_cores_per_node = 32
    n_gpus_per_node = 2
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
        n_compute_nodes = int(re.findall(r'_(\d+)nodes', wd)[0])
        # gigavoxels per h; excluding views
        high_level_res_dc['speed_total_nvox[GVx/h]'].append(ft.dataset_nvoxels / dt_tot * 3600)
        high_level_res_dc['speed_total_nvox[h/GVx]'].append(1/high_level_res_dc['speed_total_nvox[GVx/h]'][-1])
        high_level_res_dc['datasize [GVx]'].append(ft.dataset_nvoxels)  # in giga voxels
        high_level_res_dc['datasize [mm3]'].append(ft.dataset_mm3)
        high_level_res_dc['total_time [h]'].append(dt_tot / 3600)  # excluding views
        high_level_res_dc['dt_views [h]'].append(dt_views / 3600)  # in h
        high_level_res_dc['dt_points [h]'].append(dt_points / 3600)  # in h
        high_level_res_dc['dt_points_over_views'].append(dt_points / dt_views)
        high_level_res_dc['n_compute_nodes'].append(n_compute_nodes)
        for name, dt in [('total', dt_tot), ('views', dt_views), ('points', dt_points),
                         ('data structure', dt_database), ('synapses', dt_syns), ('synapse enrichment', dt_syn_enrich)]:
            dt = dt / 3600
            res_dc['time'].append(dt)
            res_dc['time_rel'].append(dt / dt_tot * 100)
            res_dc['step'].append(name)
            res_dc['datasize[mm3]'].append(ft.dataset_mm3)
            res_dc['datasize[GVx]'].append(ft.dataset_nvoxels)
            res_dc['n_compute_nodes'].append(n_compute_nodes)
    assert len(wds) > 0
    palette = sns.color_palette(n_colors=len(np.unique(res_dc['step'])), palette=palette_ident)
    palette = {k: v for k, v in zip(np.unique(res_dc['step']), palette)}
    df_highlevel = pd.DataFrame(data=high_level_res_dc)
    df_highlevel.to_csv(f'{base_dir}/data_timings_highlevel.csv')
    df = pd.DataFrame(data=res_dc)
    df.to_csv(f'{base_dir}/time_data.csv')
    fmt = '{:0.2f}'
    if len(np.unique(res_dc['n_compute_nodes'])) != 1:
        # All steps time regression plot
        log_reg = initialize_logging(f'time_allsteps_regplot_diff_nodes', log_dir=base_dir)
        plt.figure()
        axes = sns.scatterplot(data=df, x="n_compute_nodes", y="time", hue="step", palette=palette,
                               size=scatter_size)
        for ii, step in enumerate(np.unique(res_dc['step'])):
            x = np.array([df['n_compute_nodes'][ii] for ii in range(len(df['n_compute_nodes'])) if df['step'][ii] == step])
            y = [df['time'][ii] for ii in range(len(df['n_compute_nodes'])) if df['step'][ii] == step]
            mod = sm.OLS(y, sm.add_constant(x))
            res = mod.fit()
            log_reg.info(f'Fit summary for step "{step}"')
            log_reg.info(f'\n{res.summary()}\n\n')
            x_fit = np.linspace(np.min(x), np.max(x), 1000)
            y_fit = res.params[1] * x_fit + res.params[0]
            # plt.plot(x_fit, y_fit, color=palette[step])
        # plt.yscale('log')
        plt.xticks(np.arange(8, 28, step=4))
        axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                    loc='upper left', borderaxespad=0.)
        axes.set_ylabel('time [h]')
        axes.set_xlabel('no. compute nodes [1]')
        adapt_ax_params(axes)
        plt.subplots_adjust(right=0.75)
        plt.savefig(base_dir + '/timing_allsteps_regplot_diff_nodes.png', dpi=600)
        plt.close()

        # All steps time regression plot without views
        log_reg = initialize_logging(f'time_allsteps_regplot_diff_nodes_wo_views', log_dir=base_dir)
        plt.figure()
        to_be_removed = []
        for ii in range(len(res_dc['step'])):
            if res_dc['step'][ii] == 'views':
                to_be_removed.append(ii)
        for ii in np.sort(to_be_removed)[::-1]:
            res_dc['time'].pop(ii)
            res_dc['time_rel'].pop(ii)
            res_dc['step'].pop(ii)
            res_dc['datasize[mm3]'].pop(ii)
            res_dc['datasize[GVx]'].pop(ii)
            res_dc['n_compute_nodes'].pop(ii)
        df = pd.DataFrame(data=res_dc)
        axes = sns.scatterplot(data=df, x="n_compute_nodes", y="time", hue="step", palette=palette,
                               size=scatter_size)
        for ii, step in enumerate(np.unique(res_dc['step'])):
            x = np.array([df['n_compute_nodes'][ii] for ii in range(len(df['n_compute_nodes'])) if df['step'][ii] == step])
            y = [df['time'][ii] for ii in range(len(df['n_compute_nodes'])) if df['step'][ii] == step]
            mod = sm.OLS(y, sm.add_constant(x))
            res = mod.fit()
            log_reg.info(f'Fit summary for step "{step}"')
            log_reg.info(f'\n{res.summary()}\n\n')
            x_fit = np.linspace(np.min(x), np.max(x), 1000)
            y_fit = res.params[1] * x_fit + res.params[0]
            # plt.plot(x_fit, y_fit, color=palette[step])
        # plt.yscale('log')
        plt.xticks(np.arange(8, 28, step=4))
        axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                    loc='upper left', borderaxespad=0.)
        axes.set_ylabel('time [h]')
        axes.set_xlabel('no. compute nodes [1]')
        adapt_ax_params(axes)
        plt.subplots_adjust(right=0.75)
        plt.savefig(base_dir + '/time_allsteps_regplot_diff_nodes_wo_views.png', dpi=600)
        plt.close()

    else:
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
        adapt_ax_params(axes)
        plt.subplots_adjust(right=0.75)
        plt.savefig(base_dir + '/time_barplot.png', dpi=600)
        plt.close()

        # Time scatter plot
        plt.figure()
        axes = sns.scatterplot(data=df, x="datasize[GVx]", y="time", hue="step", palette=palette,
                               size=scatter_size)
        axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                    loc='upper left', borderaxespad=0.)
        # plt.xlim(0, plt.xlim()[1])
        plt.ylim(0, plt.ylim()[1])
        axes.set_ylabel('time [h]')
        axes.set_xlabel('size [GVx]')
        adapt_ax_params(axes)
        plt.subplots_adjust(right=0.75)
        plt.savefig(base_dir + '/time_pointplot.png', dpi=600)
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
        # plt.savefig(base_dir + '/totaltime_regplot.png', dpi=600)
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
        # plt.savefig(base_dir + '/time_regplot.png', dpi=600)
        # plt.close()

        # All steps time regression plot
        log_reg = initialize_logging(f'time_allsteps_regplot', log_dir=base_dir)
        plt.figure()
        axes = sns.scatterplot(data=df, x="datasize[GVx]", y="time", hue="step", palette=palette,
                               size=scatter_size)
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
        # plt.yscale('log')

        axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                    loc='upper left', borderaxespad=0.)
        axes.set_ylabel('time [h]')
        axes.set_xlabel('size [GVx]')
        plt.xlim(0, plt.xlim()[1])
        # plt.ylim(0, plt.ylim()[1])
        adapt_ax_params(axes)
        plt.subplots_adjust(right=0.75)
        plt.savefig(base_dir + '/timing_allsteps_regplot.png', dpi=600)
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
        plt.xlim(0, plt.xlim()[1])
        # plt.ylim(0, plt.ylim()[1])
        adapt_ax_params(axes)
        plt.subplots_adjust(right=0.75)
        plt.savefig(base_dir + '/time_stackedbarplot.png', dpi=600)
        plt.close()

        # All steps time regression plot without views
        log_reg = initialize_logging(f'time_allsteps_regplot_wo_views', log_dir=base_dir)
        plt.figure()
        to_be_removed = []
        for ii in range(len(res_dc['step'])):
            if res_dc['step'][ii] == 'views':
                to_be_removed.append(ii)
        for ii in np.sort(to_be_removed)[::-1]:
            res_dc['time'].pop(ii)
            res_dc['time_rel'].pop(ii)
            res_dc['step'].pop(ii)
            res_dc['datasize[mm3]'].pop(ii)
            res_dc['datasize[GVx]'].pop(ii)
            res_dc['n_compute_nodes'].pop(ii)
        df = pd.DataFrame(data=res_dc)
        axes = sns.scatterplot(data=df, x="datasize[GVx]", y="time", hue="step", palette=palette,
                               size=scatter_size)
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
        # plt.yscale('log')
        axes.legend(*axes.get_legend_handles_labels(), bbox_to_anchor=(1.05, 1),
                    loc='upper left', borderaxespad=0.)
        axes.set_ylabel('time [h]')
        axes.set_xlabel('size [GVx]')
        plt.xlim(0, plt.xlim()[1])
        # plt.ylim(0, plt.ylim()[1])
        adapt_ax_params(axes)
        plt.subplots_adjust(right=0.75)
        plt.savefig(base_dir + '/timing_allsteps_regplot_wo_views.png', dpi=600)
        plt.close()


if __name__ == '__main__':
    get_timing_plots('/mnt/example_runs/nodes_vs_time/')
    get_timing_plots('/mnt/example_runs/vol_vs_time/')
    get_speed_plots('/mnt/example_runs/vol_vs_time/')
