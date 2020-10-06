**TODO: Adapt to generic job-scheduling system and add more tutorials**

In order to run a QSUB script on the cluster one has to provide the path
 to the executed script and the corresponding script name.

 In the following example, the skeletons of all SSVs are precomputed
 (given the SV skeletons exist).
For precomputing skeletons of all SSV use the python script `batchjob_export_skeletons_fallback`,
which can be found in the `batchjob_scripts` folder inside `syconn`.
(identify the folder of `batchjob_export_skeletons_fallback`: PATH_TO_SCRIPT_DIR):
```
import os
from syconn.mp import batchjob_utils as bu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.config import global_params
import numpy as np

if __name__ == "__main__":
    ssds = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    multi_params = ssds.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 4000)
    # note that the prefix 'batchjob_' is not passed
    path_to_out = bu.batchjob_script(multi_params, "export_skeletons_fallback",
                                     script_folder=PATH_TO_SCRIPT_DIR)
```