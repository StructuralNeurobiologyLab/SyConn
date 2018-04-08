# Synapse Type

Predicting the synapse type is accomplished in two steps. First, a CNN predicts asymmetric and symmetric synapses in the volume in the same way any other object is predicted. Then, each synapse `SegmentationObjects` gathers the ratio of this prediction within its voxels.
The CNN prediction follows the standard steps of predicting any class in a volume. These predictions need to be transformed into `knossosdataset`. Finally, the synapse type for each object is extracted with

```
from syconn.proc import sd_proc
sd_proc.extract_synapse_type(sj_sd, kd_asym_path, kd_sym_path, qsub_pe=my_qsub_pe, 
                             n_max_co_processes=200)
```
