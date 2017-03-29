import cPickle as pkl
import glob

from ..processing import contact_sites_helper as csh
from ..processing import predictor_cnn as pc
from ..multi_proc import multi_proc_main as mpm
from ..utils import datahandler#, segmentationdataset


def extract_ids(cset, knossos_path, filename, qsub_pe=None, qsub_queue=None):
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
