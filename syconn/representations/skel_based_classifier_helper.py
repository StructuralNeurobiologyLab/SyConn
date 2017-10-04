import numpy as np

import super_segmentation as ss
import skel_based_classifier as sbc


def generate_clf_data_thread(args):
    this_id = args[0]
    version = args[1]
    working_dir = args[2]
    feature_contexts_nm = args[3]
    save_path = args[4]

    ssd = ss.SuperSegmentationDataset(working_dir, version)
    sso = ssd.get_super_segmentation_object(this_id)

    for feature_context_nm in feature_contexts_nm:
        print "---", this_id, feature_context_nm
        feats = sso.extract_ax_features(feature_context_nm=feature_context_nm)
        np.save(save_path % (feature_context_nm, this_id), feats)


def classifier_production_thread(args):
    working_dir = args[0]
    ssd_version = args[1]
    clf_name = args[2]
    n_estimators = args[3]
    feature_context_nm = args[4]

    sc = sbc.SkelClassifier(working_dir=working_dir,
                            ssd_version=ssd_version,
                            create=False)

    sc.train_clf(name=clf_name, n_estimators=n_estimators,
                 feature_context_nm=feature_context_nm, production=True,
                 save=True)
