# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import numpy as np
from scipy import spatial
from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)
from knossos_utils.knossosdataset import KnossosDataset
from skimage.segmentation import find_boundaries

from ..reps.super_segmentation_helper import get_sso_axoness_from_coord
from ..reps.segmentation import SegmentationDataset, SegmentationObject
from ..reps.super_segmentation import SuperSegmentationObject
from . import log_proc


def map_glia_fraction(so, box_size=None, min_frag_size=10, overwrite=True):
    """
    Map glia properties within subvolume to SegmentationObject (cs). Requires
    attribute 'neuron_partners'.

    Parameters
    ----------
    so : SegmentationObject
    box_size : np.array
        size in voxels (XYZ), default: (500, 500, 250)
    min_frag_size : int
    overwrite : bool
    """
    if not overwrite:
        so.load_attr_dict()
        if "glia_vol_frac" in so.attr_dict.keys():
            return
    if box_size is None:
        box_size = np.array([300, 300, 150])
    kd = KnossosDataset()
    # TODO: Hack
    kd.initialize_from_knossos_path(
        so.working_dir + "knossosdatasets/j0126_realigned_v4b_cbs_ext0_fix/")
    bndry = np.array(kd.boundary)
    if np.any(so.rep_coord >= bndry) or np.any(so.rep_coord < np.zeros_like(bndry)):
        log_proc.warning(so.id, so.rep_coord)
        so.save_attributes(["glia_vol_frac", "glia_sv_ids", "glia_cov_frac",
                            "glia_cov"], [-1, -1, -1, -1])
        return
    c = so.rep_coord - (box_size // 2)
    c, box_size = crop_box_to_bndry(c, box_size, bndry)
    seg = kd.from_overlaycubes_to_matrix(box_size, c, show_progress=False)
    ids, cnts = np.unique(seg, return_counts=True)
    sv_ds = SegmentationDataset("sv", working_dir=so.working_dir)
    # remove small fragments, but include background label 0 in
    # cnts for proper volume estimation
    ids = ids[cnts >= min_frag_size]
    cnts = cnts[cnts >= min_frag_size]
    glia_vx = 0
    glia_sv_ids = []
    for ix, cnt in zip(ids, cnts):
        if ix == 0:  # ignore ECS
            continue
        sv = sv_ds.get_segmentation_object(ix)
        if sv.glia_pred():
            glia_vx += cnt
            glia_sv_ids.append(ix)
    nb_box_vx = np.sum(cnts)
    glia_vol_frac = glia_vx / float(nb_box_vx)

    # get glia coverage
    neuron_ids = so.attr_dict["neuron_partners"]
    sso = SuperSegmentationObject(neuron_ids[0], working_dir=so.working_dir,
                                  create=False)
    sso.load_attr_dict()
    neuron_sv_ids = list(sso.sv_ids)
    sso = SuperSegmentationObject(neuron_ids[1], working_dir=so.working_dir,
                                  create=False)
    sso.load_attr_dict()
    neuron_sv_ids += list(sso.sv_ids)
    sv_ids_in_seg = np.array([ix in ids for ix in neuron_sv_ids], dtype=bool)
    assert np.sum(sv_ids_in_seg) >= 2
    nb_cov_vx, frac_cov_vx = get_glia_coverage(seg, neuron_sv_ids, glia_sv_ids,
                                               300, kd.scale)

    so.save_attributes(["glia_vol_frac", "glia_sv_ids",
                        "glia_cov_frac", "glia_cov"],
                       [glia_vol_frac, glia_sv_ids, frac_cov_vx,
                        nb_cov_vx])


def get_glia_coverage(seg, neuron_ids, glia_ids, max_dist, scale):
    """
    Computes the glia coverage of neurons in a segmentation volume. Neurons
    and glia are treated as two classes and coverage is defined as neuron
    boundary voxels close (within max_dist) to the glia boundary.

    Parameters
    ----------
    seg : np.array
    neuron_ids : list
    glia_ids : list
    max_dist : int/float
    scale : np.array

    Returns
    -------
    int, float
        Number and fraction of neuron boundary voxels close to glia boundary
    """
    seg = np.array(seg, np.int)
    for ix in neuron_ids:
        seg[seg == ix] = -1
    for ix in glia_ids:
        seg[seg == ix] = -2
    neuron_bndry = find_boundaries(seg == -1, mode='inner', background=0)
    glia_bndry = find_boundaries(seg == -2, mode='inner', background=0)
    neuron_bndry = np.argwhere(neuron_bndry) * scale
    glia_bndry = np.argwhere(glia_bndry) * scale
    kd_t = spatial.cKDTree(neuron_bndry)
    dists, close_neuron_vx = kd_t.query(glia_bndry, distance_upper_bound=max_dist)
    close_neuron_vx = close_neuron_vx[dists <= max_dist]
    close_neuron_vx = np.unique(close_neuron_vx)
    return len(close_neuron_vx), float(len(close_neuron_vx)) / len(neuron_bndry)


def crop_box_to_bndry(offset, box_size, bndry):
    """
    Restricts box_size and offset to valid values, i.e. within an upper
    limit (bndry) and a lower limit (0, 0, 0).

    Parameters
    ----------
    offset : np.array
    box_size : np.array / list
    bndry : np.array

    Returns
    -------
    np.array
        Valid box size and offset
    """
    diff = offset.copy() + box_size.copy() - bndry
    if np.any(diff > 0):
        # print offset, box_size, bndry
        diff[diff < 0] = 0
        box_size -= diff
        # print offset, box_size, bndry
    if np.any(offset < 0):
        # print offset, box_size, bndry
        diff = offset.copy()
        diff[diff > 0] = 0
        box_size += diff
        offset[offset < 0] = 0
        # print offset, box_size, bndry
    return offset, box_size


# TODO: probably out-dated
def map_cs_properties(cs):
    cs.load_attr_dict()
    if "neuron_partner_ct" in cs.attr_dict:
        return
    partners = cs.attr_dict["neuron_partners"]
    partner_ax = np.zeros(2, dtype=np.uint8)
    partner_ct = np.zeros(2, dtype=np.uint8)
    for kk, ix in enumerate(partners):
        sso = SuperSegmentationObject(ix, working_dir="/wholebrain"
                                                         "/scratch/areaxfs/")
        sso.load_attr_dict()
        ax = get_sso_axoness_from_coord(sso, cs.rep_coord)
        partner_ax[kk] = np.uint(ax)
        partner_ct[kk] = np.uint(sso.attr_dict["celltype_cnn"])
    cs.attr_dict["neuron_partner_ax"] = partner_ax
    cs.attr_dict["neuron_partner_ct"] = partner_ct
    cs.save_attr_dict()
