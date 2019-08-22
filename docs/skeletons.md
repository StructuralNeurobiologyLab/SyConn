# Skeletons
Skeletons are a sparse representation of the segmentation underlying a `SegmentationObjects` (usually cellular SV) or `SuperSegmentationObjects` (usually cellular SSVs).
Currently base-skeleton are stored on `SegmentationObject`-level and then stitched and pruned when they are combined for a
 `SuperSegmentationObject` (see section 'Skeletonization'). Skeletons can serve as basis for
collecting skeleton-node based statistics/features to classifiy spines, cell
compartments and cell types (see section 'Skeleton-based classification')

## Skeletonization of SSVs
### Prerequisites
* Initialized SuperSegmentationDataset ([SSD](super_segmentation_datasets.md) of SVs)
* SV skeletons (TODO: add reskeletonization workflow)

[SSVs](super_segmentation_objects.md) can be initialized by calling `SuperSegmentationObjects` with the corresponding working directory
(PATH_TO_WORKING_DIR) and their unique ID (SSV_ID)

Based on these and given the skeletons for every SV part of it,
one can reskeletonize this SSV by calling:
```
from syconn.reps.super_segmentation_helper import create_sso_skeleton
from syconn.reps.super_segmentation_object import SuperSegmentationObject
sso = SuperSegmentationObject(SSV_ID, working_dir=PATH_TO_WORKING_DIR)
sso.calculate_skeleton()
```
The object `sso` then has a attribute `sso.skeleton` which stores nodes, edges
and possible properties in a dictionary.

In order to precompute the skeletons of all SSV, the QSUB script `QSUB_export_skeletons_new`
in the `QSUB_scripts` folder inside `syconn` is used by calling the
script `generation.py` in `scripts/skeletons/`.

## Skeleton-based classification
**TODO: add runnable scripts for dataset predictions; elaborate on GT generation**

### Prerequisites
* SSV skeletons
* Mapped cellular organelles, see [here](object_mapping.md)
* Task-specific ground truth SuperSegmentationDataset

Skeleton-based features can be extracted by calling `sso.skel_features(ctx)` of
a SuperSegmentationObject instance. Note that this is already precomputed
for two context windiws when running the 'QSUB_export_skeletons_new' script.

Based on these features, supervised models like random forest classifiers (RFCs)
can be trained to predict cell type, compartments and spines. The class
`SkelClassifier` was implemented to handle data generation, training and inference.
It requires label dicts in the working directory (e.g. PATH_TO_WORKING_DIR + "/axgt_labels.pkl")
which stores the labels of SSVs in the corresponding ground truth SuperSegmentationDataset
(e.g. ssd = SuperSegmentationDataset(working_dir, version="axgt"). If node-wise labels
have to be provided, store them as skeleton.k.zip in the folder of the corresponding SSV
and give them node comments as defined in `soco.proc.skel_based_classifier`, i.e.:
```
comment_converter = {"axgt": {"soma": 2, "axon": 1, "dendrite": 0},
                     "spgt": {"shaft": 0, "head": 1, "neck": 2},
                     "ctgt": {}}
```
After the ground truth SuperSegmentationDataset has been set up, one can
instantiate `sbc = SkelClassifier(target_type="axgt", working=PATH_TO_WORKING_DIR, create=True)` and
generate ground truth data by calling `sbc.generate_data()`.

To evaluate the models and/or save them for later inference use:
```
sbc.classifier_production(production=False)  # for evaluating the current model
sbc.classifier_production(production=True)  # for inference models, which are trained on all the data available
```
Evaluation results are stored at`sbc.working_dir + "/skel_clf_%s_%s" % (sbc.target_type, sbc.ssd_version)`.