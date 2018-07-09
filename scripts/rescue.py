import syconnfs.representations.segmentation as segmentation
import syconnfs.representations.super_segmentation as ss
import syconnfs.representations.connectivity

working_dir = "/wholebrain/scratch/areaxfs/"

# ssd = ss.SuperSegmentationDataset(working_dir=working_dir)
#
# ssd.associate_objs_with_skel_nodes(("mi", "sj", "vc"), qsub_pe="openmp", stride=100)
#
# ssd.predict_axoness(qsub_pe="openmp", stride=100)
# ssd.predict_cell_types(qsub_pe="openmp", stride=100)

cm = syconnfs.representations.connectivity.ConnectivityMatrix(
    working_dir=working_dir, version=2, sj_version=4, ssd_version=4, create=True)

cm.extract_connectivity(qsub_pe="openmp")
cm.get_sso_specific_info(qsub_pe="openmp")