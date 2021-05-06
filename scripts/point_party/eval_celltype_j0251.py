import collections
import glob
import pandas
import seaborn as sns
from syconn.handler import basics, config
from syconn.cnn.TrainData import CellCloudDataJ0251
from syconn.handler.prediction import certainty_estimate, str2int_converter, int2str_converter
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset

from syconn.handler.prediction_pts import predict_pts_plain, pts_loader_scalar_infer,\
    pts_loader_scalar, pts_pred_scalar_nopostproc, get_celltype_model_pts, get_pt_kwargs
import os

palette_ident = 'colorblind'


def predict_celltype_gt(ssd_kwargs, **kwargs):
    """
    Perform cell type predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.

    Args:
        ssd_kwargs:

    Returns:

    """
    out_dc = predict_pts_plain(ssd_kwargs, get_celltype_model_pts, pts_loader_scalar_infer, pts_pred_scalar_nopostproc,
                               **kwargs)
    for ssv_id in out_dc:
        logit = np.concatenate(out_dc[ssv_id])
        cls = np.argmax(logit, axis=1).squeeze()
        if np.ndim(cls) == 0:
            cls = cls[None]
        cls_maj = collections.Counter(cls).most_common(1)[0][0]
        out_dc[ssv_id] = (cls_maj, certainty_estimate(logit, is_logit=True))
    return out_dc


def create_catplot(dest_p, qs, ls=6, r=(0, 1.0), add_boxplot=False, legend=False, **kwargs):
    """
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
     The box extends from the lower to upper quartile values of the data, with
      a line at the median. The whiskers extend from the box to show the range
       of the data (1.5* interquartile range (Q3-Q1). Flier points are those past the end of the whiskers.

    https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

    https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot


    Parameters
    ----------
    dest_p :
    qs :
    r :
    add_boxplot:
    legend :
    ls :

    Returns
    -------

    """
    fig = plt.figure()
    c = '0.25'
    size = 10
    if 'size' in kwargs:
        size = kwargs['size']
        del kwargs['size']
    if add_boxplot:
        ax = sns.boxplot(data=qs, palette="Greys", showfliers=False, **kwargs)
    ax = sns.swarmplot(data=qs, clip_on=False, color=c, size=size, **kwargs)
    if not legend:
        plt.gca().legend().set_visible(False)
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10, rotation=45)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(r)
    plt.tight_layout()
    fig.savefig(dest_p, dpi=400)
    qs.to_excel(dest_p[:-4] + ".xls")
    plt.close()


def create_lineplot(dest_p, df, ls=6, r=(0, 1.0), legend=True, **kwargs):
    """
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
     The box extends from the lower to upper quartile values of the data, with
      a line at the median. The whiskers extend from the box to show the range
       of the data (1.5* interquartile range (Q3-Q1). Flier points are those past the end of the whiskers.

    https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

    https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot


    Parameters
    ----------
    dest_p :
    r :
    legend :
    ls :

    Returns
    -------

    """
    fig = plt.figure()
    size = 10
    if 'size' in kwargs:
        size = kwargs['size']
        del kwargs['size']
    ax = sns.lineplot(data=df, **kwargs)
    if not legend:
        plt.gca().legend().set_visible(False)
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10, rotation=45)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(r)
    plt.tight_layout()
    fig.savefig(dest_p, dpi=400)
    df.to_excel(dest_p[:-4] + ".xls")
    plt.close()


def create_pointplot(dest_p, df, ls=6, r=(0, 1.0), legend=True, **kwargs):
    """
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
     The box extends from the lower to upper quartile values of the data, with
      a line at the median. The whiskers extend from the box to show the range
       of the data (1.5* interquartile range (Q3-Q1). Flier points are those past the end of the whiskers.

    https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

    https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot


    Parameters
    ----------
    dest_p :
    r :
    legend :
    ls :

    Returns
    -------

    """
    fig = plt.figure()
    size = 8
    if 'size' in kwargs:
        size = kwargs['size']
        del kwargs['size']
    ax = sns.pointplot(data=df, size=size, **kwargs)
    if not legend:
        plt.gca().legend().set_visible(False)
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10, rotation=45)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(r)
    plt.tight_layout()
    fig.savefig(dest_p, dpi=400)
    df.to_excel(dest_p[:-4] + ".xls")
    plt.close()


def plot_performance_summary(bd, include_special_inputs=False):
    res_dc_pths = glob.glob(bd + '*/redun*_prediction_results.pkl', recursive=True)
    fscores = []
    labels = []
    redundancies = []
    ctx = []
    npts = []
    for fp in res_dc_pths:
        dc = basics.load_pkl2obj(fp)
        if not include_special_inputs and (not dc['use_syntype'] or dc['cellshape_only']):
            continue
        res = list(dc[f'fscore_macro'])
        fscores.extend(res)
        labels.extend([dc['model_tag']] * len(res))
        redundancies.extend([dc['redundancy']] * len(res))
        npts.extend([dc['npts']] * len(res))
        ctx.extend([dc['ctx']] * len(res))
    index = pandas.MultiIndex.from_arrays([labels, redundancies, npts, ctx], names=('labels', 'redundancy', 'npts', 'ctx'))
    df = pandas.DataFrame(fscores, index=index, columns=['fscore'])
    df = df.sort_values(by=['npts', 'ctx', 'redundancy'], ascending=True)
    create_pointplot(f"{bd}/performance_summary_allRedundancies_pointplot{'_special' if include_special_inputs else ''}.png", df.reset_index(), ci='sd',
                     x='labels', y='fscore', hue='redundancy', dodge=True, r=(0 if include_special_inputs else 0.4, 1), palette=palette_ident,
                     capsize=.1, scale=0.75, errwidth=1)
    create_lineplot(f"{bd}/performance_summary_allRedundancies{'_special' if include_special_inputs else ''}.png", df.reset_index(), ci='sd', err_style='band',
                     x='labels', y='fscore', hue='redundancy', r=(0 if include_special_inputs else 0.4, 1), palette=palette_ident)


if __name__ == '__main__':
    ncv_min = 0
    n_cv = 10
    nclasses = 11
    int2str_label = {ii: int2str_converter(ii, 'ctgt_j0251_v2') for ii in range(nclasses)}
    str2int_label = {int2str_converter(ii, 'ctgt_j0251_v2'): ii for ii in range(nclasses)}
    overwrite = False
    n_runs = 3

    state_dict_fname = 'state_dict.pth'
    wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/"
    bbase_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint_celltypes_j0251_rerunFeb21/'
    all_res_paths = set()
    for ctx, npts, use_syntype, cellshape_only in [(20000, 50000, False, False), (20000, 50000, True, False),
                                                   (20000, 50000, True, True),
                                                   (20000, 25000, True, False), (20000, 75000, True, False),
                                                   (20000, 5000, True, False), (4000, 25000, True, False)]:
        scale = ctx // 10
        skip_model = False
        base_dir = f'{bbase_dir}//celltype_pts{npts}_ctx{ctx}'
        if cellshape_only:
            base_dir += '_cellshape_only'
        if not use_syntype:  # ignore if cell shape only
            base_dir += '_no_syntype'
        mfold = base_dir + '/celltype_CV{}/celltype_pts_j0251v2_scale{}_nb{}_ctx{}_relu{}{}_gn_CV{}_eval{}/'
        for run in range(n_runs):
            for CV in range(ncv_min, n_cv):
                mfold_complete = mfold.format(CV, scale, npts, ctx, "" if use_syntype else "_noSyntype",
                                              "_cellshapeOnly" if cellshape_only else "_myelin", CV, run)
                mpath = f'{mfold_complete}/{state_dict_fname}'
                if not os.path.isfile(mpath):
                    msg = f"'{mpath}' not found. Skipping entire eval run for {base_dir}."
                    raise ValueError(msg)
        if skip_model:
            continue
        # prepare GT
        check_train_ids = set()
        check_valid_ids = []
        for CV in range(ncv_min, n_cv):
            ccd = CellCloudDataJ0251(cv_val=CV)
            check_train_ids.update(set(ccd.splitting_dict['train']))
            check_valid_ids.extend(list(ccd.splitting_dict['valid']))
        assert len(check_train_ids) == len(check_valid_ids)
        assert np.max(np.unique(check_valid_ids, return_counts=True)[1]) == 1
        target_names = [int2str_label[kk] for kk in range(nclasses)]
        csv_p = ccd.csv_p
        df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
        ssv_ids = df[:, 0].astype(np.uint64)
        if len(np.unique(ssv_ids)) != len(ssv_ids):
            raise ValueError('Multi-usage of IDs!')
        str_labels = df[:, 1]
        ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
        ssd_kwargs = dict(working_dir=wd)
        ssd = SuperSegmentationDataset(**ssd_kwargs)
        for redundancy in [1, 10, 20, 50]:
            perf_res_dc = collections.defaultdict(list)  # collect for each run
            for run in range(n_runs):
                log = config.initialize_logging(f'log_eval{run}_sp{npts}k_redun{redundancy}', base_dir)
                log.info(f'\nStarting evaluation of model with npoints={npts}, eval. run={run}.\n'
                         f'GT data at wd={wd}\n')
                for CV in range(ncv_min, n_cv):
                    mfold_complete = mfold.format(CV, scale, npts, ctx, "" if use_syntype else "_noSyntype",
                                                  "_cellshapeOnly" if cellshape_only else "_myelin", CV, run)
                    mpath = f'{mfold_complete}/{state_dict_fname}'
                    assert os.path.isfile(mpath)
                    mkwargs, loader_kwargs = get_pt_kwargs(mpath)
                    assert loader_kwargs['npoints'] == npts
                    log.info(f'model_kwargs={mkwargs}')
                    ccd = CellCloudDataJ0251(cv_val=CV)
                    split_dc = ccd.splitting_dict
                    if 'myelin' in mpath:
                        map_myelin = True
                    else:
                        map_myelin = False
                    if '_noSyntype' in mpath:
                        assert not use_syntype
                    if '_cellshapeOnly' in mpath:
                        assert cellshape_only
                    mkwargs['mpath'] = mpath
                    log.info(f'Using model "{mpath}" for cross-validation split {CV}.')
                    fname_pred = f'{os.path.split(mpath)[0]}/ctgt_v4_splitting_cv{CV}_redun{redundancy}_{run}_10fold_PRED.pkl'
                    assert fname_pred not in all_res_paths
                    all_res_paths.add(fname_pred)
                    # check pred if available
                    incorrect_pred = False
                    if os.path.isfile(fname_pred) and not overwrite:
                        os.path.isfile(fname_pred)
                        res_dc = basics.load_pkl2obj(fname_pred)
                        res_dc = dict(res_dc)  # convert to standard dict
                        incorrect_pred = (len(res_dc) != len(split_dc['valid'])) or (not np.all([k in split_dc['valid'] for k in res_dc]))
                        if incorrect_pred:
                            print(f'Wrong prediction stored at: {fname_pred}. Recomputing now.')
                    if overwrite or not os.path.isfile(fname_pred) or incorrect_pred:
                        res_dc = predict_celltype_gt(ssd_kwargs, mpath=mpath, bs=10,
                                                     nloader=10, device='cuda', seeded=True, ssv_ids=split_dc['valid'],
                                                     npredictor=4, use_test_aug=False,
                                                     loader_kwargs={'redundancy': redundancy, 'map_myelin': map_myelin,
                                                                    'use_syntype': use_syntype,
                                                                    'cellshape_only': cellshape_only},
                                                     **loader_kwargs)
                        incorrect_pred = (len(res_dc) != len(split_dc['valid'])) or (not np.all([k in split_dc['valid'] for k in res_dc]))
                        if incorrect_pred:
                            raise ValueError('Incorrect prediction.')
                        basics.write_obj2pkl(fname_pred, res_dc)
                valid_ids, valid_ls, valid_preds, valid_certainty = [], [], [], []

                for CV in range(ncv_min, n_cv):
                    ccd = CellCloudDataJ0251(cv_val=CV)
                    split_dc = ccd.splitting_dict
                    mfold_complete = mfold.format(CV, scale, npts, ctx, "" if use_syntype else "_noSyntype",
                                                  "_cellshapeOnly" if cellshape_only else "_myelin", CV, run)
                    mpath = f'{mfold_complete}/{state_dict_fname}'
                    assert os.path.isfile(mpath)
                    fname_pred = f'{os.path.split(mpath)[0]}/ctgt_v4_splitting_cv{CV}_redun{redundancy}_{run}_10fold_PRED.pkl'
                    res_dc = basics.load_pkl2obj(fname_pred)
                    res_dc = dict(res_dc)  # convert to standard dict
                    assert len(res_dc) == len(split_dc['valid'])
                    assert np.all([k in split_dc['valid'] for k in res_dc])
                    valid_ids_local, valid_ls_local, valid_preds_local = [], [], []
                    for ix, curr_id in enumerate(ssv_ids):
                        if curr_id not in split_dc['valid']:
                            continue
                        curr_l = ssv_labels[ix]
                        valid_ls.append(curr_l)
                        curr_pred, curr_cert = res_dc[curr_id]
                        valid_preds.append(curr_pred)
                        valid_certainty.append(curr_cert)
                        valid_ids.append(curr_id)
                valid_preds = np.array(valid_preds)
                valid_certainty = np.array(valid_certainty)
                valid_ls = np.array(valid_ls)
                valid_ids = np.array(valid_ids)
                log.info(f'Final prediction result for run {run} with {loader_kwargs} and {mkwargs}.')
                class_rep = classification_report(valid_ls, valid_preds, labels=np.arange(nclasses), target_names=target_names,
                                                  output_dict=True)
                for ii, k in enumerate(target_names):
                    perf_res_dc[f'fscore_class_{ii}'].append(class_rep[k]['f1-score'])
                perf_res_dc['fscore_macro'].append(f1_score(valid_ls, valid_preds, average='macro'))
                perf_res_dc['accuracy'].append(accuracy_score(valid_ls, valid_preds))
                perf_res_dc['cert_correct'].append(valid_certainty[valid_preds == valid_ls])
                perf_res_dc['cert_incorrect'].append(valid_certainty[valid_preds != valid_ls])
                log.info(classification_report(valid_ls, valid_preds, labels=np.arange(nclasses), target_names=target_names))
                log.info(confusion_matrix(valid_ls, valid_preds, labels=np.arange(nclasses)))
                log.info(f'Mean certainty correct:\t{np.mean(valid_certainty[valid_preds == valid_ls])}\n'
                         f'Mean certainty incorrect:\t{np.mean(valid_certainty[valid_preds != valid_ls])}')
                log.info(f'Incorrectly predicted IDs (ID, label, prediction): '
                         f'{[(ix, int2str_label[label], int2str_label[pred]) for ix, label, pred in zip(valid_ids[valid_preds != valid_ls], valid_ls[valid_preds != valid_ls], valid_preds[valid_preds != valid_ls])]}')
            # plot everything
            perf_res_dc = dict(perf_res_dc)
            model_tag = f'ctx{loader_kwargs["ctx_size"]}_nb{npts}'
            if cellshape_only:
                model_tag += 'cellshapeOnly'
            if not use_syntype:
                model_tag += 'noSyntype'
            perf_res_dc['model_tag'] = model_tag
            perf_res_dc['ctx'] = ctx
            perf_res_dc['redundancy'] = redundancy
            perf_res_dc['npts'] = npts
            perf_res_dc['cellshape_only'] = cellshape_only
            perf_res_dc['use_syntype'] = use_syntype

            basics.write_obj2pkl(f"{base_dir}/redun{redundancy}_prediction_results.pkl", perf_res_dc)
            fscores = np.concatenate([perf_res_dc[f'fscore_class_{ii}'] for ii in range(nclasses)] +
                                     [perf_res_dc[f'fscore_macro'], perf_res_dc['accuracy']]).squeeze()
            labels = np.concatenate([np.concatenate([[int2str_label[ii]] * n_runs for ii in range(nclasses)]),
                                     np.array(['f1_score_macro'] * n_runs + ['accuracy'] * n_runs)])

            df = pandas.DataFrame(data={'quantity': labels, 'f1score': fscores})
            create_catplot(f"{base_dir}/redun{redundancy}_performances.png", qs=df, x='quantity', y='f1score',
                           size=10)

            cert_correct = np.concatenate(perf_res_dc['cert_correct'])
            cert_incorrect = np.concatenate(perf_res_dc['cert_incorrect'])
            df = pandas.DataFrame(data={'quantity': ['correct'] * len(cert_correct) + ['incorrect'] * len(cert_incorrect),
                                        'certainty': np.concatenate([cert_correct, cert_incorrect]).squeeze()})
            create_catplot(f"{base_dir}/redun{redundancy}_certainty.png", qs=df, x='quantity', y='certainty',
                           add_boxplot=True, size=4)
    plot_performance_summary(bbase_dir)
    plot_performance_summary(bbase_dir, include_special_inputs=True)
