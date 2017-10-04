import os
from syconnmp.shared_mem import start_multiprocess
from syconnfs.representations.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject


def collect_so_dirs(sso_id):
    # dirs = []
    # for sso_id in ssd.ssv_ids:
        sso = SuperSegmentationObject(sso_id, create=False, version="3", scaling=(10, 10, 20), working_dir="/wholebrain/scratch/areaxfs/")
        sso.load_attr_dict()
        for sv in sso.svs:
            new_dir = sv.working_dir + "/../../u/pschuber/areaxfs/"
            sv._version = 0
            sv._working_dir = new_dir
            # mesh_dc = load_pkl2obj(self.mesh_path)
            change_perm([sv.segobj_dir, "pschuber"])
            # dirs.append(sv.segobj_dir)
        # for mi in sso.mis:
        #     dirs.append(mi.segobj_dir)
        # for sj in sso.sjs:
        #     dirs.append(sj.segobj_dir)
        # for vc in sso.vcs:
        #     dirs.append(vc.segobj_dir)
    # return dirs


def change_perm(args):
    dir, username = args
    os.system("setfacl -m user:%s:rwx %s/" % (username, dir))


if __name__ == "__main__":
    ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/", version="axgt")
    sso_ids = ssds.ssv_ids
    # dirs = collect_so_dirs(SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/", version="3"))

    # multi_params = zip(dirs, ["sdorkenw"]*len(dirs))
    start_multiprocess(collect_so_dirs, sso_ids, nb_cpus=20, debug=False)
    # print multi_params