# Config

Each working directory has a single `config` file which stores some parameters for convenience. The `config` parser is implemented in `syconn.config.parser`. The `config` itself is stored as `config.ini` in the working directory.
It is mostly used for setting the versions of the latest datasets for each object type. All parameters are defaults for various functions and class initialization.

An example `config` using all defined categories:

```
[Versions]
sv = 0
vc = 0
sj = 0
syn = 0
syn_ssv = 0
mi = 0
ssv = 0
cs_agg = 0
ax_gt = 0

[Paths]
kd_seg_path = path_seg
kd_sym_path = path_sym
kd_asym_path = path_asym
kd_sj = path_sj
kd_vc = path_vc
kd_mi = path_mi

[LowerMappingRatios]
mi = 0.5
sj = 0.1
vc = 0.5

[UpperMappingRatios]
mi = 1.
sj = 0.9
vc = 1.

[Sizethresholds]
mi = 2786
sj = 498
vc = 1584
```

Additionally, `syconn.config.global_params.py` stores `SyConn`-wide(!) parameters such as currently active working directory and meshing parameters.
This will be refactored at some point into a single configuration location.

# Working directory
The working directory stores SegmentationDatasets and SuperSegmentationDatasets of the initial, the glia split RAG, the
 cell organelles and contact sites / synapses and is specified in `syconn.config.global_params.py`. The above config has to be place within the working directory.