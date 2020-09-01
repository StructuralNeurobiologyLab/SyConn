# Config
Each working directory has a single `config.yml` file which stores dataset
specific parameters. The `Config` class is implemented in `syconn.handler.config`.
The `config` itself is stored as `config.yml` in the working directory.
It is used for setting the versions and paths of the different dataset, e.g.
KnossosDataset of source data like cell segmentation or predictions of
sub-cellular structures (synaptic junctions, mitochondria, ..).

A default config is created inside `generate_default_conf()` function
implemented in `syconn.handler.config`. The default values can be found at
`syconn.handler.config.yml` and are always used as fallback if a value cannot
be found in the config file of the current working directory.


# Modifying parameters in a config file
Parameters can be adapted while creating the config file for a new run.
In the script that starts the whole pipeline, e.g. `example_run.py` or `full_run.py`,
one has to extend the `key_val_pairs_conf` list by a tuple (key, value).


# Working directory
The working directory stores SegmentationDatasets and SuperSegmentationDatasets
of the initial, the glia split RAG, the cell organelles and contact sites /
synapses and is specified in `syconn.global_params.py`. The above config has to
be placed within the working directory.


