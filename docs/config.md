# Config

Each working directory has a single `config` file which stores some parameters for convenience. The `config` parser is implemented in `syconn.config.parser`. The `config` itself is stored as `config.ini` in the working directory.
It is mostly used for setting the versions of the latest datasets for each object type. All parameters are defaults for various functions and class initialization.

An example `config` using all defined categories:

```
[Versions]
sv = 0
vc = 0
sj = 5
mi = 0
ssv = 0
cs = 29
cs_agg = 26
conn = 12
ax_gt = 0

[Dataset]
scaling = 10., 10., 20.
seg_path = /path/to/knossosdataset/with/seg/
super_seg_path = /path/to/knossosdataset/with/superseg/

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