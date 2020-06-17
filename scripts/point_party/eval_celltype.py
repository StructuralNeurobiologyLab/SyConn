import collections
import pandas
from syconn.handler import basics, config
from syconn.handler.prediction import certainty_estimate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.classification import classification_report
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset

from syconn.handler.prediction_pts import predict_pts_plain,\
    pts_loader_scalar, pts_pred_scalar_nopostproc, get_celltype_model_pts, get_pt_kwargs
import os


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
    out_dc = predict_pts_plain(ssd_kwargs, get_celltype_model_pts, pts_loader_scalar, pts_pred_scalar_nopostproc,
                               **kwargs)
    for ssv_id in out_dc:
        logit = np.concatenate(out_dc[ssv_id])
        if da_equals_tan:
            # accumulate evidence for DA and TAN
            logit[:, 1] += logit[:, 6]
            # remove TAN in proba array
            logit = np.delete(logit, [6], axis=1)
            # INT is now at index 6 -> label 6 is INT
        cls = np.argmax(logit, axis=1).squeeze()
        if np.ndim(cls) == 0:
            cls = cls[None]
        cls_maj = collections.Counter(cls).most_common(1)[0][0]
        out_dc[ssv_id] = (cls_maj, certainty_estimate(logit, is_logit=True))
    return out_dc


def create_catplot(dest_p, qs, ls=20, r=(0, 1.0), add_boxplot=False, legend=False, **kwargs):
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
    import seaborn as sns
    fig = plt.figure()
    c = '0.25'
    if 'size' in kwargs:
        size = kwargs['size']
        del kwargs['size']
    if add_boxplot:
        ax = sns.boxplot(data=qs, palette="Greys", showfliers=False, **kwargs)
    ax = sns.swarmplot(data=qs, clip_on=False, color=c, size=size, **kwargs)
    if not legend:
        plt.gca().legend().set_visible(False)
    # set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
    ax.tick_params(axis='x', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.tick_params(axis='y', which='both', labelsize=ls, direction='out',
                   length=4, width=3, right=False, top=False, pad=10)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(r)
    fig.savefig(dest_p, dpi=300)
    qs.to_excel(dest_p[:-4] + ".xlsx")
    plt.close()


if __name__ == '__main__':
    ncv_min = 0
    n_cv = 10
    nclasses = 8
    da_equals_tan = True
    overwrite = True
    n_runs = 2
    state_dict_fname = 'state_dict.pth'
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    gt_version = "ctgt_v4"
    # base_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/celltype_pts25000_ctx10000/'
    # mfold = base_dir + '/celltype_eval{}_sp25k/celltype_pts_scale1000_nb25000_ctx10000_swish_gn_CV{}_eval{}/'
    # base_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/celltype_pts2500_ctx10000/'
    # mfold = base_dir + '/celltype_eval{}_sp2k/celltype_pts_scale1000_nb2500_ctx10000_swish_gn_CV{}_eval{}/'
    base_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint_celltypes/celltype_pts50000_ctx20000_eval{}/'
    mfold = base_dir + '/celltype_pts50000_ctx20000/celltype_pts_scale2000_nb50000_ctx20000_swish_gn_CV{}_eval{}/'
    for run in range(n_runs):
        for CV in range(ncv_min, n_cv):
            mpath = f'{mfold.format(run, CV, run)}/{state_dict_fname}'
            assert os.path.isfile(mpath), f"'{mpath}' not found."

    # prepare GT
    str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, GPe=5, INT=7, FS=8, GLIA=9)
    del str2int_label['GLIA']
    del str2int_label['FS']
    str2int_label["GP "] = 5  # typo
    int2str_label = {v: k for k, v in str2int_label.items()}
    target_names = [int2str_label[kk] for kk in range(nclasses)]
    if da_equals_tan:
        target_names[1] = 'Modulatory'
        target_names.remove('TAN')
        nclasses = 7
    csv_p = '/wholebrain/songbird/j0126/GT/celltype_gt/j0126_cell_type_gt_areax_fs6_v3.csv'
    df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
    ssv_ids = df[:, 0].astype(np.uint)
    if len(np.unique(ssv_ids)) != len(ssv_ids):
        raise ValueError('Multi-usage of IDs!')
    str_labels = df[:, 1]
    ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)

    ssd_kwargs = dict(working_dir=wd, version=gt_version)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    mkwargs, loader_kwargs = get_pt_kwargs(mfold)
    for redundancy in [1, 10, 20, 50]:
        perf_res_dc = collections.defaultdict(list)  # collect for each run
        for run in range(n_runs):
            npoints = loader_kwargs['npoints']
            log = config.initialize_logging(f'log_eval{run}_sp{npoints}k_redun{redundancy}', base_dir.format(run))
            log.info(f'\nStarting evaluation of model with npoints={npoints}, eval. run={run}, '
                     f'model_kwargs={mkwargs} and da_equals_tan={da_equals_tan}.\n'
                     f'GT: version={gt_version} at wd={wd}\n')
            for CV in range(ncv_min, n_cv):
                split_dc = basics.load_pkl2obj(f'/wholebrain/songbird/j0126/areaxfs_v6/ssv_ctgt_v4'
                                               f'/ctgt_v4_splitting_cv{CV}_10fold.pkl')
                mpath = f'{mfold.format(run, CV, run)}/{state_dict_fname}'
                assert os.path.isfile(mpath)
                mkwargs['mpath'] = mpath
                log.info(f'Using model "{mpath}" for cross-validation split {CV}.')
                fname_pred = f'{os.path.split(mpath)[0]}/../ctgt_v4_splitting_cv{CV}_10fold_PRED.pkl'
                if overwrite or not os.path.isfile(fname_pred):
                    res_dc = predict_celltype_gt(ssd_kwargs, mpath=mpath, redundancy=(redundancy, redundancy), bs=10,
                                                 nloader=10, device='cuda', seeded=True, ssv_ids=split_dc['valid'],
                                                 npredictor=5, use_test_aug=False, **loader_kwargs)
                    basics.write_obj2pkl(fname_pred, res_dc)
            valid_ids, valid_ls, valid_preds, valid_certainty = [], [], [], []

            for CV in range(ncv_min, n_cv):
                mpath = f'{mfold.format(run, CV, run)}/{state_dict_fname}'
                res_dc = basics.load_pkl2obj(f'{os.path.split(mpath)[0]}/../ctgt_v4_splitting_cv{CV}_10fold_PRED.pkl')
                split_dc = basics.load_pkl2obj(f'/wholebrain/songbird/j0126/areaxfs_v6/ssv_ctgt_v4'
                                               f'/ctgt_v4_splitting_cv{CV}_10fold.pkl')
                valid_ids_local, valid_ls_local, valid_preds_local = [], [], []
                for ix, curr_id in enumerate(ssv_ids):
                    if curr_id not in split_dc['valid']:
                        continue
                    curr_l = ssv_labels[ix]
                    if da_equals_tan:
                        # adapt GT labels
                        if curr_l == 6: curr_l = 1  # TAN and DA are the same now
                        if curr_l == 7: curr_l = 6  # INT now has label 6
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
                perf_res_dc[f'fscore_class_{ii}'].append([class_rep[k]['f1-score']])
            perf_res_dc['fscore_macro'].append(f1_score(valid_ls, valid_preds, average='macro'))
            perf_res_dc['accuracy'].append(accuracy_score(valid_ls, valid_preds))
            perf_res_dc['cert_correct'].append(valid_certainty[valid_preds == valid_ls])
            perf_res_dc['cert_incorrect'].append(valid_certainty[valid_preds != valid_ls])
            basics.write_obj2pkl(f"{base_dir.format(run)}/prediction_results_redun{redundancy}.pkl", perf_res_dc)
            log.info(classification_report(valid_ls, valid_preds, labels=np.arange(nclasses), target_names=target_names))
            log.info(confusion_matrix(valid_ls, valid_preds, labels=np.arange(nclasses)))
            log.info(f'Mean certainty correct:\t{np.mean(valid_certainty[valid_preds == valid_ls])}\n'
                     f'Mean certainty incorrect:\t{np.mean(valid_certainty[valid_preds != valid_ls])}')
            for ix in range(len(valid_ls)):
                curr_l = valid_ls[ix]
                curr_pred = valid_preds[ix]
                curr_id = valid_ids[ix]
                curr_cert = valid_certainty[ix]
                # if curr_pred != curr_l:
                #     log.info(f'id: {curr_id} target: {curr_l} prediction: {curr_pred} certainty: {curr_cert:.2f}')
                    # ssv = ssd.get_super_segmentation_object(curr_id)
                    # print(syn_sign_ratio_celltype(ssv, comp_types=[0, ]), syn_sign_ratio_celltype(ssv, comp_types=[0, ]))
                    # ssv.meshes2kzip(f'/wholebrain/scratch/pschuber/tmp/{ssv.id}_p{curr_pred}_t{curr_l}_cert{curr_cert>0.75}.k.zip', synssv_instead_sj=True)
            # log.info('-------------------------------')

        # plot everything
        fscores = np.concatenate([np.array(perf_res_dc[f'fscore_class_{ii}']).squeeze() for ii in range(nclasses)] +
                                 [perf_res_dc[f'fscore_macro'], perf_res_dc['accuracy']]).squeeze()
        df = pandas.DataFrame(data={'quantity': np.concatenate([[int2str_label[ii]] * n_runs for ii in range(nclasses)] +
                                                               [['f1_score_macro'] * n_runs + ['accuracy'] * n_runs]),
                                    'f1score': fscores})
        create_catplot(f"{base_dir.format(run)}/performances_redun{redundancy}.png", qs=df, x='quantity', y='f1score', size=10)

        cert_correct = np.concatenate(perf_res_dc['cert_correct'])
        cert_incorrect = np.concatenate(perf_res_dc['cert_incorrect'])
        df = pandas.DataFrame(data={'quantity': ['correct'] * len(cert_correct) + ['incorrect'] * len(cert_incorrect),
                                    'certainty': np.concatenate([cert_correct, cert_incorrect]).squeeze()})
        create_catplot(f"{base_dir.format(run)}/certainty_redun{redundancy}.png", qs=df, x='quantity', y='certainty', add_boxplot=True,
                       size=4)
