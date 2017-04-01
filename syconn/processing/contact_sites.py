import cPickle as pkl
import glob

from ..processing import contact_sites_helper as csh
from ..processing import predictor_cnn as pc
from ..multi_proc import multi_proc_main as mpm
from ..utils import datahandler#, segmentationdataset

from syconnfs.representations import segmentation


def find_contact_sites(cset, knossos_path, filename, qsub_pe=None,
                       qsub_queue=None):
    multi_params = []
    for chunk in cset.chunk_dict.values():
        multi_params.append([chunk, knossos_path, filename])

    if qsub_pe is None and qsub_queue is None:
        results = mpm.start_multiprocess(csh.contact_site_detection_thread,
                                         multi_params, debug=False)
    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "contact_site_detection",
                                      pe=qsub_pe, queue=qsub_queue)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file) as f:
                results.append(pkl.load(f))
    else:
        raise Exception("QSUB not available")


def extract_contact_sites(cset, filename, working_dir, stride=10, qsub_pe=None,
                          qsub_queue=None):
    segdataset = segmentation.SegmentationDataset("cs_pre",
                                                  version="new",
                                                  working_dir=working_dir,
                                                  create=True)

    multi_params = []
    chunks = cset.chunk_dict.values()
    for chunk_block in [chunks[i: i + stride]
                        for i in xrange(0, len(chunks), stride)]:
        multi_params.append([chunk_block, working_dir, filename,
                             segdataset.version])

    if qsub_pe is None and qsub_queue is None:
        results = mpm.start_multiprocess(csh.extract_pre_cs_thread,
                                         multi_params, debug=False)
    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "extract_pre_cs",
                                      pe=qsub_pe, queue=qsub_queue)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file) as f:
                results.append(pkl.load(f))
    else:
        raise Exception("QSUB not available")