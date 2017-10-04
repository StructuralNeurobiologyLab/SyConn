from syconnfs.representations import super_segmentation as ss
from syconnfs.representations.utils import parse_cc_dict_from_kzip, read_txt_from_zip
# most methods can be run via qsub or shared_mem multiprocessing
# set your favorite qsub_pe / qsub_queue as parameter
# you should also set a sufficient stride: ~1000 jobs as result is good

my_mergelist = "/wholebrain/u/pschuber/NeuroPatch/datasets/rag_recon_pruned_glia_ml_v3.k/mergelist.txt"

# INITIALIZATION
# with mergelist
ssd = ss.SuperSegmentationDataset("/wholebrain/scratch/areaxfs/", version="6",
                                  sv_mapping=my_mergelist)
ssd.save_dataset_shallow()

# # without mergelist
# ssd = ss.SuperSegmentationDataset("working/dir", version="my_version")

# Fundamental Setup
ssd.save_dataset_deep(qsub_pe="openmp")

# Map cell objects - only works when cell objects were extracted and mapped to
# segmentation objects

# raise("Wait for SJ threshold. Should come June 3rd.")
# ssd.aggregate_segmentation_object_mappings(["sj", "mi", "vc"])
# ssd.apply_mapping_decisions(["sj", "mi", "vc"])

