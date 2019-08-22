# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
from syconn.handler.prediction import NeuralNetworkInterface
from syconn.handler.basics import chunkify, get_filepaths_from_dir
from syconn.proc.stats import projection_pca
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.proc.rendering import render_sso_ortho_views
from syconn import global_params
import numpy as np
import tqdm
import os
from syconn.cnn.TrainData import SSVCelltype


def latent_data_loader(args):
    ssd, ssv_ids, diagonal_size = args
    latent = []
    for ix in ssv_ids:
        ssv = ssd.get_super_segmentation_object(ix)
        ssv.load_attr_dict()
        if diagonal_size is not None:
            bb = np.array(ssv.bounding_box) * global_params.config.entries['Dataset']['scaling']
            diagonal = np.linalg.norm(bb[1] - bb[0])
            if not diagonal > diagonal_size:
                continue
        if "latent_ortho" in ssv.attr_dict:
            latent.append(ssv.attr_dict["latent_ortho"])
    return np.array(latent, dtype=np.float32)


def load_latent_data(ssd, ssv_ids=None, diagonal_size=None):
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    nb_params = np.min([1000, len(ssv_ids)])
    params = [(ssd, ixs, diagonal_size) for ixs in chunkify(ssv_ids, nb_params)]
    res = start_multiprocess_imap(latent_data_loader, params, nb_cpus=20)
    latent = np.concatenate(res).astype(np.float32)

    # use all three views
    # initial shape: (N, 10, 3); number of SSVs, latent space dim. Z, number of views
    latent = latent.swapaxes(2, 1)  # (N, 3, Z)
    latent = latent.reshape(-1, 10) # (N * 3, Z)
    print("Collected %d views." % len(latent))
    return latent


def predict_latent_ssd(ssd, m, ssv_ids=None):
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    # shuffle SV IDs
    np.random.seed(0)
    np.random.shuffle(ssv_ids)
    pbar = tqdm.tqdm(total=len(ssv_ids))
    for i in xrange(0, len(ssv_ids), 66):
        ixs = ssv_ids[i:i+66]
        views_chunk = np.zeros((len(ixs), 4, 3, 512, 512))
        for ii, ix in enumerate(ixs):
            ssv = ssd.get_super_segmentation_object(ix)
            ssv.load_attr_dict()
            if ssv.size > 1e5 and not "latent_ortho" in ssv.attr_dict:
                try:
                    views_chunk[ii] = ssv.load_views("ortho").swapaxes(1, 0)[None, ..., ::2, ::2]
                except (KeyError, ValueError) as e:
                    print("Error loading 'ortho' views of SSV %d. Re-rendering.\n %s"
                          % (ix, e))
                    views = render_sso_ortho_views(ssv)
                    ssv.save_views(views, view_key='ortho')
                    views_chunk[ii] = views.swapaxes(1, 0)[None, ..., ::2, ::2]
        latent = m.predict_proba(views_chunk)
        for ii, ix in enumerate(ixs):
            ssv = ssd.get_super_segmentation_object(ix)
            ssv.load_attr_dict()
            if ssv.size > 1e5:
                ssv.save_attributes(["latent_ortho"], [latent[ii]])
        pbar.update(len(ixs))


def load_celltype_ctgt(m):
    ct = SSVCelltype(None, None)
    ssv_ids = list(ct.train_d.squeeze()) + list(ct.valid_d.squeeze())
    ssv_labels = list(ct.train_l) + list(ct.valid_l)
    ssv_labels = np.concatenate([[l] * 3 for l in ssv_labels])
    ssd = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/",
                                   version="6")
    predict_latent_ssd(ssd, m, ssv_ids)
    latent = load_latent_data(ssd, ssv_ids)
    return latent, ssv_labels


if __name__ == "__main__":
    fold = "/wholebrain/scratch/pschuber/"

    # # all SSV
    # for ds in [50e3, 100e3]:
    #     ssd = SuperSegmentationDataset(working_dir=wd)
    #     predict_latent_ssd(ssd)
    #     latent = load_latent_data(ssd, diagonal_size=ds)
    #     labels = np.zeros((len(latent))).astype(np.int8)
    #     _ = projection_pca(latent, labels, fold + "/%s_diagonal%d_kde_pca.png" %
    #                        ("all", ds), pca=None, colors=None, target_names=["None"])
    #     tsne_kwargs = {"n_components": 3, "random_state": 0,
    #                    "perplexity": 20, "n_iter": 10000}
    #     projection_tSNE(latent, labels,fold + "/%s_diagonal%d_kde_tsne.png" % ("all", ds),
    #                     target_names=["none"], **tsne_kwargs)

    # celltype gt only
    m_dir = "/wholebrain/scratch/pschuber/CNN_Training/SyConn/tripletnet_SSV/"
    m_ps = get_filepaths_from_dir(m_dir, recursively=True, ending='LAST.mdl')
    for m_p in m_ps:
        try:
            m = NeuralNetworkInterface(m_p,
                imposed_batch_size=6,
                nb_labels=10, arch="triplet")
            _ = m.predict_proba(np.zeros((1, 4, 3, 512, 512)))
            ssd = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/",
                                           version="6")
            # m = get_tripletnet_model_ortho()
            latent, labels = load_celltype_ctgt(m)

            _ = projection_pca(latent, labels, fold + "/ctgt_{}_kde_pca.tif".format(os.path.split(m_p)[1][:-4]),
                               pca=None, colors=None,target_names=["EA", "MSN", "GP", "INT"])
        except ValueError:
            print("Skipped", m_p)
        # tsne_kwargs = {"n_components": 3, "random_state": 0,
        #                "perplexity": 20, "n_iter": 10000}
        # projection_tSNE(latent, labels,fold + "/ctgt_%d_kde_tsne.png" % ii,
        #                 target_names=["EA", "MSN", "GP", "INT"], **tsne_kwargs)