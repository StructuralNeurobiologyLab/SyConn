#import sys, os
#sys.path.insert(0, os.path.abspath('..'))

from knossos_utils import knossosdataset
from knossos_utils import chunky
import time
from syconn.config import global_params

from syconn.proc import sd_proc
from syconn.proc import ssd_proc

from syconn.extraction import cs_extraction_steps as ces
from syconn.extraction import cs_processing_steps as cps
from syconn.extraction import object_extraction_wrapper as oew
from syconn.reps import segmentation as seg

kd_seg_path = "/mnt/j0126_cubed/"
wd = "/mnt/j0126/areaxfs_v10/"
cd_dir = wd + "chunkdatasets/"

# wd = "/wholebrain/songbird/j0126/areaxfs_v5/extract_combine_test"   # TODO: del
# cd_dir = wd + "/chunkdatasets"  # TODO: del

#### initializing and loading and chunk dataset from knossosdataset
kd = knossosdataset.KnossosDataset()    # Sets initial values of object
kd.initialize_from_knossos_path(kd_seg_path)     # Initializes the dataset by parsing the knossos.conf in path + "mag1"

cd = chunky.ChunkDataset()      # Class that contains a dict of chunks (with coordinates) after initializing it
cd.initialize(kd, kd.boundary, [512, 512, 512], cd_dir,
                        box_coords=[0, 0, 0], fit_box_size=True)

chunky.save_dataset(cd)
cd = chunky.load_dataset(cd_dir)
# chunks = [i for i in range(1000)]   # TODO: del
#########
# extract_combine_test
# oew.from_ids_to_objects(cd, None, hdf5names=["sv"],
#                    n_folders_fs=10000,
#                    overlaydataset_path=kd_seg_path,
#                    chunk_list=chunks, suffix="", n_chunk_jobs=1000,
#                    qsub_pe="openmp", qsub_queue=None,
#                    n_max_co_processes=100)
#########

# Object extraction
#oew.from_ids_to_objects(cd, None, overlaydataset_path=kd_seg_path, n_chunk_jobs=5000,
#                        hdf5names=["sv"], n_max_co_processes=5000, qsub_pe='default', qsub_queue='all.q', qsub_slots=1,
#                        n_folders_fs=10000)
# Object Processing
sd = seg.SegmentationDataset("sv", working_dir=wd)
sd_proc.dataset_analysis(sd, qsub_pe="default", qsub_queue='all.q', stride=10, n_max_co_processes=5000)

#sd_proc.map_objects_to_sv(sd, "sj", kd_seg_path, nb_cpus=1,
#                          n_max_co_processes=100, stride=100)   # TODO: qsub_pe="openmp",

############################################################################################
# ##### Cell object extraction #####
# from_probabilities_to_objects(cd, filename, hdf5names)
#
#
# ##### Object Processing #####
# sj_sd = SegmentationDataset("sj", working_dir="path/to/wd")
# sd_proc.dataset_analysis(sj_sd)
#
# # ??The segmentation needs to be written to a KnossosDataset before running this
# sd_proc.map_objects_to_sv(sj_sd, obj_type, kd_path,
#                           qsub_pe=my_qsub_pe, nb_cpus=1,
#                           n_max_co_processes=100)
#
# ##### SSD Assembly #####
# # ??create SSD and mergelist (knossos)
# # mergelist can be supplied to ssd as parameter during initialization and will be applied automatically
# ssd_proc.aggregate_segmentation_object_mappings(ssd, obj_types, qsub_pe=my_qsub_pe)
# ssd_proc.apply_mapping_decisions(ssd, obj_types, qsub_pe=my_qsub_pe)
#
#
# ##### CS Extraction #####
# ces.find_contact_sites(cset, knossos_path, filename, n_max_co_processes=200,
#                        qsub_pe=my_qsub_pe)
# ces.extract_agg_contact_sites(cset, filename, hdf5name, working_dir,
#                               n_folders_fs=10000, suffix="",
#                               n_max_co_processes=200, qsub_pe=my_qsub_pe)
#
# ##### CS Processing #####
# cps.combine_and_split_cs_agg(working_dir, cs_gap_nm=300,
#                              stride=100, qsub_pemyqsub_pe,
#                              n_max_co_processes=200)
#
#
# ## CS Classification ##
# sd_proc.export_sd_to_knossosdataset(cs_sd, cs_kd, block_edge_length=512,
#                                     qsub_pe=my_qsub_pe, n_max_co_processes=100)
# cps.overlap_mapping_sj_to_cs_via_kd(cs_sd, sj_sd, cs_kd, qsub_pe=my_qsub_pe, n_max_co_processes=100, n_folders_fs=10000)
#
#
# ## Synaptic Classification ##
#
