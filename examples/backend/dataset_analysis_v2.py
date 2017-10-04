# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import time
import numpy as np
from syconnfs.handler.compression import AttributeDict
from syconnfs.handler.basics import write_obj2pkl, load_pkl2obj, chunkify
import itertools
import os
import fadvise
from syconnmp.shared_mem import start_multiprocess


def loader(ixs):
    ix1, ix2, ix3 = ixs
    curr_p = base_p + "/%d/%d/%d/attr_dict.pkl" % (ix1, ix2, ix3)
    try:
        return load_pkl2obj(curr_p)
    except IOError:
        return {}


def extract_arr(glob_ad, base_p):
    nb_samples = len(glob_ad)
    ids = np.zeros(nb_samples, dtype=np.uint)
    rep_coords = np.zeros((nb_samples, 3))
    axs = np.zeros((nb_samples, 2), dtype=np.uint8)
    cts = np.zeros((nb_samples, 2), dtype=np.uint8)
    proba = np.zeros(nb_samples)
    overlap_area = np.zeros(nb_samples)
    partners = np.zeros((nb_samples, 2), dtype=np.uint)
    ix_array = np.ones(nb_samples, dtype=np.bool)
    cnt = 0
    for k, v in glob_ad.iteritems():
        ids[cnt] = k
        try:
            rep_coords[cnt] = v["rep_coord"]
            axs[cnt] = v["neuron_partner_ax"]
            cts[cnt] = v["neuron_partner_ct"]
            proba[cnt] = v["synaptivity_proba"]
            overlap_area[cnt] = v["overlap_area"]
            partners[cnt] = v["neuron_partners"]
        except KeyError:
            ix_array[cnt] = 0
        cnt += 1
    loss_ids = ids[~ix_array]
    print "%d losses." % len(loss_ids)
    ids = ids[ix_array]
    rep_coords = rep_coords[ix_array]
    axs = axs[ix_array]
    cts = cts[ix_array]
    proba = proba[ix_array]
    partners = partners[ix_array]
    overlap_area = overlap_area[ix_array]
    np.save(base_p + "idss.npy", ids)
    np.save(base_p + "rep_coords.npy", rep_coords)
    np.save(base_p + "neuron_partner_axs.npy", axs)
    np.save(base_p + "neuron_partner_cts.npy", cts)
    np.save(base_p + "synaptivity_probas.npy", proba)
    np.save(base_p + "overlap_areas.npy", overlap_area)
    np.save(base_p + "neuron_partnerss.npy", partners)
    np.save(base_p + "loss_ids.npy", loss_ids)
    raise()


if __name__ == "__main__":
    base_p = "/wholebrain/scratch/areaxfs/cs_33/so_storage/"
    start = time.time()
    cnt = 0
    glob_ad = {}
    if 0:
        print "Gathering attribute dicts multiprocessed."
        all_ixs = list(itertools.product(np.arange(0, 100), np.arange(0, 100), np.arange(10)))
        ch_ixs = chunkify(all_ixs, 100)
        for ch in ch_ixs:
            curr_dcs = start_multiprocess(loader, ch, nb_cpus=20)
            for dc in curr_dcs:
                glob_ad.update(dc)
            print "Number keys in glob dc:", len(glob_ad)
        write_obj2pkl(base_p + "global_attr_dict_v2.pkl", glob_ad)

    if 0:
        print "Gathering attribute dicts."
        for ix1, ix2, ix3 in itertools.product(np.arange(0, 100), np.arange(0, 100), np.arange(10)):
            cnt += 1
            curr_p = base_p + "/%d/%d/%d/attr_dict.pkl" % (ix1, ix2, ix3)
            if not os.path.isfile(curr_p):
                print curr_p
                continue
            ad = load_pkl2obj(curr_p)
            for k, v in ad.iteritems():
                glob_ad[k] = v
            if cnt == 1000:
                print "1000 look ups took:", time.time() - start
                print "%d entries in curr dictionary" % len(ad)
                start = time.time()
                cnt = 0
        print "%d entries in global dictionary" % len(glob_ad)

    if 1:
        print "Writing .npy arrays."
        ad = load_pkl2obj(base_p + "global_attr_dict.pkl")
        extract_arr(ad, base_p)
        # write_obj2pkl(base_p + "global_attr_dict.pkl", glob_ad)
