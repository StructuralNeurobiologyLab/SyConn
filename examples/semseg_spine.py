from syconn import global_params
from syconn.reps.super_segmentation import *
from syconn.reps.super_segmentation_helper import semseg_of_sso_nocache
from syconn.proc.ssd_assembly import init_sso_from_kzip
from syconn.handler.prediction import get_semseg_spiness_model
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SyConn example run')
    parser.add_argument('--working_dir', type=str,
                        default=os.path.expanduser("~/SyConn/example_cube1/"),
                        help='Working directory of SyConn')
    parser.add_argument('--kzip', type=str, default='',
                        help='path to kzip file which contains a cell reconstruction (see '
                             'SuperSegmentationObject().export2kzip())')
    args = parser.parse_args()

    # path to working directory of example cube - required to load pretrained models
    path_to_workingdir = os.path.expanduser(args.working_dir)

    # path to cell reconstruction k.zip
    cell_kzip_fn = os.path.abspath(os.path.expanduser(args.kzip))
    if not os.path.isfile(cell_kzip_fn):
        raise FileNotFoundError
    # set working directory to obtain models
    global_params.wd = path_to_workingdir

    # get model for spine detection
    m = get_semseg_spiness_model()

    # load SSO instance from k.zip file
    sso = init_sso_from_kzip(cell_kzip_fn, sso_id=1)

    # run prediction and store result in new kzip
    cell_kzip_fn_spines = cell_kzip_fn[:-6] + '_spines.k.zip'
    semseg_of_sso_nocache(sso, dest_path=cell_kzip_fn_spines, verbose=True,
                          semseg_key="spinesstest", k=0, ws=(256, 128),
                          model=m,  nb_views=2, comp_window=8e3)
    node_preds = sso.semseg_for_coords(
        sso.skeleton['nodes'], "spinesstest",
        **global_params.semseg2coords_spines)
    sso.skeleton["spinesstest"] = node_preds
    sso.save_skeleton_to_kzip(dest_path=cell_kzip_fn_spines,
                              additional_keys=["spinesstest"])

