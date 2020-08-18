from syconn import global_params
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SyConn morphology embedding with multi-views.', )
    parser.add_argument('--working_dir', type=str,
                        default=os.path.expanduser("~/SyConn/example_cube1/"),
                        help='Working directory of SyConn')
    parser.add_argument('--kzip', type=str, default=None,
                        help='path to kzip file which contains a cell reconstruction (see '
                             'SuperSegmentationObject().export2kzip())')
    parser.add_argument('--modelpath', type=str, default=None,
                        help='path to the model.pt file of the trained model.')
    args = parser.parse_args()

    # path to working directory of example cube - required to load pretrained models
    path_to_workingdir = os.path.abspath(os.path.expanduser(args.working_dir))

    # path to cell reconstruction k.zip
    if args.kzip is None:
        args.kzip = '../data/2_axonexample.k.zip'
    cell_kzip_fn = os.path.abspath(os.path.expanduser(args.kzip))
    if not os.path.isfile(cell_kzip_fn):
        raise FileNotFoundError('Could not find cell reconstruction file at the'
                                f' specified location {cell_kzip_fn}.')

    # set working directory to obtain models
    global_params.wd = path_to_workingdir

    # import rendering.py after setting working directory so that OpenGL settings specified in
    # the config are applied properly
    from syconn.reps.super_segmentation import *
    from syconn.reps.super_segmentation_helper import view_embedding_of_sso_nocache
    from syconn.proc.ssd_assembly import init_sso_from_kzip
    from syconn.handler.prediction import get_tripletnet_model_e3

    model_p = args.modelpath

    # get model for spine prediction
    if model_p is None:
        if not os.path.isdir(path_to_workingdir):
            msg = f'Could not find the specified working directory at ' \
                  f'{path_to_workingdir}. Please make sure to point to a ' \
                  f'proper directory which contains the model file at ' \
                  f'{global_params.config.mpath_spiness} or specify a model via' \
                  ' --modelpath.'
            raise FileNotFoundError(msg)
        m = get_tripletnet_model_e3()
    else:
        if not os.path.isfile(cell_kzip_fn):
            raise FileNotFoundError('Could not find a model at the specified '
                                    f'location: {model_p}.')
        try:
            from elektronn3.models.base import InferenceModel
        except ImportError as e:
            msg = "elektronn3 could not be imported ({}). Please see 'https://github." \
                  "com/ELEKTRONN/elektronn3' for more information.".format(e)
            raise ImportError(msg)
        m = InferenceModel(model_p)
        m._path = model_p

    # load SSO instance from k.zip file
    sso = init_sso_from_kzip(cell_kzip_fn, sso_id=1)

    # run inference and store result in new kzip
    cell_kzip_fn_embed = cell_kzip_fn[:-6] + '_embed.k.zip'
    # generate skeleton attribute 'latent_morph'
    view_embedding_of_sso_nocache(sso, dest_path=cell_kzip_fn_embed, verbose=True, ws=(256, 128),
                                  model=m,  nb_views=2, comp_window=8e3, add_cellobjects=('mi', 'vc', 'sj'))
    # store RGB projection of the embedding in k.zip file
    sso.morphembed2mesh(dest_path=cell_kzip_fn_embed)
    # also save skeleton as Knossos tree with latent vector
    sso.save_skeleton_to_kzip(dest_path=cell_kzip_fn_embed, additional_keys=["latent_morph"])
