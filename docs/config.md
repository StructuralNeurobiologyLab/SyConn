# Config
Each working directory has a single `config.yml` file which stores dataset 
specific parameters. The `Config` class is implemented in `syconn.handler.config`.
The `config` itself is stored as `config.ini` in the working directory. 
It is mostly used for setting the versions of the latest datasets for 
each object type. All parameters are defaults for various functions and 
class initialization. 

A default config is created inside `generate_default_conf()` function 
implemented in `syconn.handler.config`. All default parameters are defined there.

An example `config.ini` contains all defined categories:

```
[Versions]
sv = 0
vc = 0
sj = 0
syn = 0
syn_ssv = 0
mi = 0
ssv = 0
ax_gt = 0
cs = 0

[Paths]
kd_seg =
kd_sym =
kd_asym =
kd_sj =
kd_vc =
kd_mi =
init_rag =
use_new_subfold =

[Dataset]
scaling = 1, 1, 1
syntype_avail = False

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

[Probathresholds]
mi = 0.428571429
sj = 0.19047619
vc = 0.285714286

[Mesh]
allow_mesh_gen_cells = True
use_new_meshing = True

[MeshDownsampling]
sv = 4, 4, 2
sj = 2, 2, 1
vc = 4, 4, 2
mi = 8, 8, 4
cs = 2, 2, 1
conn = 2, 2, 1
syn_ssv = 2, 2, 1

[MeshClosing]
sv = 0
s = 0
vc = 0
mi = 0
cs = 0
conn = 4
syn_ssv = 0

[Skeleton]
allow_skel_gen = True

[Views]
use_large_fov_views_ct = False
use_new_renderings_locs = True

[Glia]
prior_glia_removal = True
```


The data types for the `config.ini` entries are defined in `configspec.ini` 
which also has to exist in the working directory:
```
[Versions]
__many__ = string

[Paths]
__many__ = string
use_new_subfold = boolean

[Dataset]
scaling = float_list(min=3, max=3)
syntype_avail = boolean

[LowerMappingRatios]
__many__ = float

[UpperMappingRatios]
__many__ = float

[Sizethresholds]
__many__ = integer

[Probathresholds]
__many__ = float

[Mesh]
allow_mesh_gen_cells = boolean
use_new_meshing = boolean

[MeshDownsampling]
__many__ = int_list(min=3, max=3)

[MeshClosing]
__many__ = integer

[Skeleton]
allow_skel_gen = boolean

[Views]
use_large_fov_views_ct = boolean
use_new_renderings_locs = boolean

[Glia]
prior_glia_removal = boolean
```

# Additional parameters for a config file
Additional parameters can be added while creating the config file for a new run.
In the script that starts the whole pipeline, e.g. `example_run.py` or `full_run.py`,
one has to extend the `key_val_pairs_conf` list by a tuple (key, value).  


# Working directory
The working directory stores SegmentationDatasets and SuperSegmentationDatasets 
of the initial, the glia split RAG, the cell organelles and contact sites / 
synapses and is specified in the starting script (e.g. 'example_run.py' or 'full_run.py'),
or can be given as a command line argument. 
The above config has to be place within the working directory.