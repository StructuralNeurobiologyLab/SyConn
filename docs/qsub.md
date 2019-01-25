**TODO: Adapt to generic job-scheduling system and add more tutorials**

In order to run a QSUB script on the cluster one has to provide the path
 to the executed script and the corresponding script name.

 In the following example, the skeletons of all SSVs are precomputed
 (given the SV skeletons exist).
For precomputing skeletons of all SSV use the QSUB script `QSUB_export_skeletons_new`,
which can be found in the `QSUB_scripts` folder inside `syconn`.
(identify the folder of `QSUB_export_skeletons_new`: FOLDER_OF_QSUB_SCRIPT):
```
import os
from syconn.mp import qsub_utils as qu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.config import global_params
import numpy as np

if __name__ == "__main__":
    ssds = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    multi_params = ssds.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 4000)
    path_to_out = qu.QSUB_script(multi_params, "export_skeletons_new",
                                 n_max_co_processes=200, pe="openmp", queue=None,
                                 script_folder=PATH_TO_QSUB_SCRIPT, suffix="")
```