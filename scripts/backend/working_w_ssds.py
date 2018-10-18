from syconn.reps import super_segmentation as ss
from syconn.mp.mp_utils import start_multiprocess

# most methods can be run via qsub or shared_mem multiprocessing
# set your favorite qsub_pe / qsub_queue as parameter
# you should also set a sufficient stride: ~1000 jobs as result is good

# my_mergelist = "/wholebrain/u/pschuber/NeuroPatch/datasets/rag_recon_pruned_glia_ml_v3.k/mergelist.txt"
#
# # INITIALIZATION
# # new dataset with mergelist
# ssd = ss.SuperSegmentationDataset("/wholebrain/scratch/areaxfs/", version="new",
#                                   sv_mapping=my_mergelist)
# ssd.save_dataset_shallow()
#
# # # without mergelist
# # ssd = ss.SuperSegmentationDataset("working/dir", version="my_version")
#
# # Fundamental Setup
# ssd.save_dataset_deep(qsub_pe="openmp")
#
# # Map cell objects - only works when cell objects were extracted and mapped to
# # segmentation objects
#
# # raise("Wait for SJ threshold. Should come June 3rd.")
# # ssd.aggregate_segmentation_object_mappings(["sj", "mi", "vc"])
# # ssd.apply_mapping_decisions(["sj", "mi", "vc"])

def mesh_creator_sso(ssv):
    # try:
    ssv.load_attr_dict()
    ssv._map_cellobjects()
    _ = ssv._load_obj_mesh(obj_type="mi", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sj", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="vc", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sv", rewrite=False)
    ssv.calculate_skeleton()
    ssv.clear_cache()


if __name__ == "__main__":
    ssd = ss.SuperSegmentationDataset(version="new", ssd_type="ssv",
                                      sv_mapping="/mnt/j0126/areaxfs_v10/RAGs/v4b_20180214_nocb_merges_reconnected_knossos_mergelist.txt")
    ssd.save_dataset_shallow()
    # generell dann
    # ssd.save_dataset_deep(qsub_pe="openmp", n_max_co_processes=100)
    # da das aber overkill ist (und der stride bei default auch zu gross ist fuer das mini dataset), reicht
    ssd.save_dataset_deep(nb_cpus=20, stride=5)

    # start_multiprocess(mesh_creator_sso, list(ssd.ssvs), nb_cpus=20, debug=False)