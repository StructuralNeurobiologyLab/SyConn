# Config
Each working directory has a single `config` file which stores dataset specific parameter. The `Config` class is implemented in `syconn.handler.config`.
The `config` itself is stored as `config.ini` in the working directory. It is mostly used for setting the versions of the latest datasets for each object type.
All parameters are defaults for various functions and class initialization.

An example `config.ini` using all defined categories:

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
kd_seg = path_seg
kd_sym = path_sym
kd_asym = path_asym
kd_sj = path_sj
kd_vc = path_vc
kd_mi = path_mi
init_rag = path_initrag
py36path =

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


The data types for the `config.ini` entries are defined in `configspec.ini` which also has to exist in the working directory:
```
[Versions]
__many__ = string

[Paths]
__many__ = string()

[Dataset]
scaling = float_list(min=3, max=3)

[LowerMappingRatios]
__many__ = float

[UpperMappingRatios]
__many__ = float

[Sizethresholds]
__many__ = integer

```
Additionally, `syconn.global_params.py` stores `SyConn`-wide(!) parameters such as currently active working directory and meshing parameters.
This will be refactored at some point into a single configuration location.

# Working directory
The working directory stores SegmentationDatasets and SuperSegmentationDatasets of the initial, the glia split RAG, the
 cell organelles and contact sites / synapses and is specified in `syconn.global_params.py`. The above config has to be place within the working directory.