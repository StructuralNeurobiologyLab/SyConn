from syconnfs.representations.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconnfs.representations.segmentation import SegmentationDataset, SegmentationObject
from syconnfs.representations.utils import subfold_from_ix
from syconnfs.handler.basics import chunkify
from syconnmp.shared_mem import start_multiprocess
import shutil
import sys
import os
orig_mesh_dir = "/u/pschuber/areaxfs/sv_0/so_storage/"


def copy_mesh_for_so(so):
    so_id = so.id
    if not os.path.isdir(so.segobj_dir):
        print so.segobj_dir
        return
        os.makedirs(so.segobj_dir)
    try:
        shutil.copy(orig_mesh_dir + subfold_from_ix(so_id)+ "mesh.pkl", so.segobj_dir)
    except:
        pass

def copy_ssds_meshs(ssd):
    for sso in ssd.ssvs:
        # sso.load_attr_dict()
        svs = list(sso.svs)
        if len(svs) > 1e4:
            print "Skipped %d with %d SV's." % (sso.id, len(svs))
            continue
        start_multiprocess(copy_mesh_for_so, svs, nb_cpus=20)
        sys.stdout.write("\r%d" % sso.id)
        sys.stdout.flush()


def mesh_creator(so, force=True):
    if not force:
        _ = so.mesh
    else:
        indices, vertices = so._mesh_from_scratch()
        try:
            so._save_mesh(indices, vertices)
        except Exception, e:
            print "\n-----------------------\n" \
                  "Mesh of %s %d could not be saved:\n%s\n" \
                  "-------------------------\n" % (so.type, so.id, e)

def mesh_creator_sso(sso_ix):
    sso = SuperSegmentationObject(sso_ix, working_dir="/wholebrain/scratch/areaxfs/", nb_cpus=1, version="6")
    sso.load_attr_dict()
    print sso.id, len(sso.sv_ids)
    _ = sso.mesh
    _ = sso.mi_mesh
    _ = sso.sj_mesh
    _ = sso.vc_mesh


def mesh_creator_mi(ixs):
    for ix in ixs:
        so = SegmentationObject(ix, "mi", working_dir="/wholebrain/scratch/areaxfs/", version="7")
        mesh_creator(so)
        del so


def mesh_creator_sj(ixs):
    for ix in ixs:
        so = SegmentationObject(ix, "sj", working_dir="/wholebrain/scratch/areaxfs/", version="8")
        mesh_creator(so, force=False)
        del so


def mesh_creator_vc(ixs):
    for ix in ixs:
        so = SegmentationObject(ix, "vc", working_dir="/wholebrain/scratch/areaxfs/", version="4")
        mesh_creator(so)
        del so


def preproc_meshs(ssd):
    for sso in ssd.ssvs:
        start_multiprocess(mesh_creator, sso.svs, nb_cpus=20, debug=False)
        sys.stdout.write("\r%d" % sso.id)
        sys.stdout.flush()


def write_meshs_helper(sso):
    sso.meshs2kzip()
    sso.mergelist2kzip()


if __name__ == "__main__":
    ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/",
                                    version="6")
    start_multiprocess(mesh_creator_sso, ssds.ssv_ids, nb_cpus=20)
    # start_multiprocess(write_meshs_helper, list(ssds.ssvs), nb_cpus=20)
    # start_multiprocess(write_meshs_helper, list(ssds.ssvs), nb_cpus=20)

    # copy_ssds_meshs(ssds)
    # sds = SegmentationDataset("sj", working_dir="/wholebrain/scratch/areaxfs/")
    # usable_ids = sds.ids[sds.sizes > 498]
    # print "Processing %d sj's." % (len(usable_ids))
    # chs = chunkify(usable_ids, 1000)
    # start_multiprocess(mesh_creator_sj, chs, nb_cpus=20, debug=False)
    # sds = SegmentationDataset("vc", working_dir="/wholebrain/scratch/areaxfs/")
    # usable_ids = sds.ids[sds.sizes > 1584]
    # print "Processing %d vc's." % (len(usable_ids))
    # chs = chunkify(usable_ids, 1000)
    # start_multiprocess(mesh_creator_vc, chs, nb_cpus=20, debug=False)
    # sds = SegmentationDataset("mi", working_dir="/wholebrain/scratch/areaxfs/")
    # usable_ids = sds.ids[sds.sizes > 2786]
    # print "Processing %d mi's." % (len(usable_ids))
    # chs = chunkify(usable_ids, 1000)
    # start_multiprocess(mesh_creator_mi, chs, nb_cpus=20, debug=False)
    # copy_ssds_meshs(ssds)
    # preproc_meshs(ssds)
    # ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/",
    #                                 version="ctgt")
    # copy_ssds_meshs(ssds)
    # preproc_meshs(ssds)