# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import os
from syconn.mp import qsub_utils as qu
from syconn.mp.shared_mem import start_multiprocess
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.basics import chunkify
from syconn.proc.mapping import map_glia_fraction
import numpy as np
import itertools

# outdated analysis script


def get_glia_frac(cs):
    cs.load_attr_dict()
    if not "glia_vol_frac" in cs.attr_dict.keys() and cs.attr_dict["synaptivity_proba"] <= 0.5:
        return -1, -1, -1, -1, cs.size
    vc_size = np.sum([np.sum(v) for v in cs.attr_dict["mapped_vc_sizes"].values()])
    return cs.attr_dict["glia_vol_frac"], vc_size, cs.attr_dict["glia_cov_frac"], cs.attr_dict["glia_cov"], cs.size#, cs.rep_coord


def get_vc_sizes(cs):
    cs.load_attr_dict()
    if cs.attr_dict["synaptivity_proba"] > 0.5:
        # iterates over the two partner vc's
        return np.sum([np.sum(v) for v in cs.attr_dict["mapped_vc_sizes"].values()])
    return -1


def get_glia_cov_frac(cs):
    cs.load_attr_dict()
    if cs.attr_dict["synaptivity_proba"] > 0.5:
        # iterates over the two partner vc's
        if not "glia_cov_frac" in cs.attr_dict.keys():
            return -1
        return cs.attr_dict["glia_cov_frac"]
    return -1


if __name__ == "__main__":
    script_folder = os.path.abspath(os.path.dirname(__file__) + "/../qsub_scripts/")
    sds = SegmentationDataset("cs", working_dir="/wholebrain/scratch/areaxfs/", version="33")
    fold = sds.so_storage_path
    f1 = np.arange(0, 100)
    f2 = np.arange(0, 100)
    f3 = np.arange(0, 10)
    all_poss_attr_dicts = list(itertools.product(f1, f2, f3))
    assert len(all_poss_attr_dicts) == 100*100*10
    multi_params = ["%s/%d/%d/%d/" % (fold, par[0], par[1], par[2]) for par in all_poss_attr_dicts]
    multi_params = chunkify(multi_params, 3000)
    multi_params = [[par] for par in multi_params]
    path_to_out = qu.QSUB_script(multi_params, "map_glia",
                                 n_max_co_processes=160, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="_v3")
    #
    all_cs = list(sds.sos)
    # # collect / reduce results
    # res = start_multiprocess(get_glia_frac, all_cs, nb_cpus=20)
    # tmp_dc = {}
    # for i in range(len(res)):
    #     tmp_dc[all_cs[i].id] = res[i]
    # ordered_fracs = np.zeros(len(res))
    # for ii, sj in enumerate(sds.ids):
    #     ordered_fracs[ii] = tmp_dc[sj]
    # np.save(sds.path + "/glia_vol_fracs.npy", ordered_fracs)
    #
    # res = start_multiprocess(get_vc_sizes, all_cs, nb_cpus=20)
    # tmp_dc = {}
    # for i in range(len(res)):
    #     tmp_dc[all_cs[i].id] = res[i]
    # ordered_fracs = np.zeros(len(res))
    # for ii, sj in enumerate(sds.ids):
    #     ordered_fracs[ii] = tmp_dc[sj]
    # np.save(sds.path + "/mapped_vc_sizes.npy", ordered_fracs)

    res = start_multiprocess(get_glia_frac, all_cs, nb_cpus=20)
    tmp_dc_size = {}
    tmp_dc_vc = {}
    tmp_dc_glia_cov_frac = {}
    tmp_dc_glia_cov = {}
    tmp_dc_glia_vol_frac = {}
    tmp_dc_rep_coord = {}
    for i in range(len(res)):
        tmp_dc_size[all_cs[i].id] = res[i][4]
        tmp_dc_vc[all_cs[i].id] = res[i][1]
        tmp_dc_glia_cov_frac[all_cs[i].id] = res[i][2]
        tmp_dc_glia_cov[all_cs[i].id] = res[i][3]
        tmp_dc_glia_vol_frac[all_cs[i].id] = res[i][0]
        # tmp_dc_rep_coord[all_cs[i].id] = res[i][-1]
    ordered_fracs_size = np.zeros(len(res))
    ordered_fracs_vc = np.zeros(len(res))
    ordered_fracs_glia_cov = np.zeros(len(res))
    ordered_fracs_glia_cov_frac = np.zeros(len(res))
    ordered_fracs_glia_vol_frac = np.zeros(len(res))
    # ordered_rep_coord = np.zeros(len(res))
    for ii, sj in enumerate(sds.ids):
        ordered_fracs_size[ii] = tmp_dc_size[sj]
        ordered_fracs_vc[ii] = tmp_dc_vc[sj]
        ordered_fracs_glia_cov[ii] = tmp_dc_glia_cov[sj]
        ordered_fracs_glia_cov_frac[ii] = tmp_dc_glia_cov_frac[sj]
        ordered_fracs_glia_vol_frac[ii] = tmp_dc_glia_vol_frac[sj]
        # ordered_rep_coord[ii] = tmp_dc_rep_coord[sj]
    np.save(sds.path + "/glia_cov_fracs.npy", ordered_fracs_glia_cov_frac)
    np.save(sds.path + "/glia_covs.npy", ordered_fracs_glia_cov)
    np.save(sds.path + "/glia_vol_fracs.npy", ordered_fracs_glia_vol_frac)
    np.save(sds.path + "/mapped_vc_sizes.npy", ordered_fracs_vc)
    np.save(sds.path + "/sizes.npy", ordered_fracs_size)
    # np.save(sds.path + "/rep_coords.npy", ordered_fracs_size)