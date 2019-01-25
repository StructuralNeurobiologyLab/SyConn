# Documentation

## Installation
We recommend installing the latest Anaconda release. Then set up the python environment:
```
conda create -n py36 python=3.6 anaconda
source activate py36
```
Install all prerequisites and finally git clone and install syconn:
```
conda install vigra -c conda-forge
conda install mesa -c menpo
conda install osmesa -c menpo
conda install freeglut
conda install pyopengl
git clone https://github.com/StructuralNeurobiologyLab/SyConn.git
cd SyConn
pip install -r requirements.txt
pip install .
```
Or alternatively with the developer flag:
```
pip install -e .
```
In order to use elektronn3 models, python>=3.6 is required (when installing elektronn3 checkout
 branch `phil`; this will be updated soon to work with the `master` branch).

In the case that there are problems with snappy/python-snappy remove previous installations and
install them via conda:
```
conda uninstall snappy
conda install snappy
conda install python-snappy
```

## Example run
Place the example data in syconn/scripts/example_run/data/, `cd` to SyConn/scripts/example_run/ and then run
```
python start.py --working_dir=~/SyConn/example_cube/
```


## Analysis steps
After initialization of the SDs (SVs and cellular organelles) and the SSD
containing the agglomerated SVs, several analysis steps can be applied:

* [Optional] [Glia removal](glia_removal.md)

* [Neuronal morphology analysis and classification](neuron_analysis.md) to identify cellular compartments (e.g. axons and spines) and to perform morphology based cell type classification.

* [Contact site extraction](contact_site_extraction.md)

* [Identification of synapses and extraction of a wiring diagram](contact_site_classification.md)


## Package structure and data classes
The basic data structures and initialization procedures are explained in the following sections:

* SyConn operates with a pre-defined [working directory and config files](config.md)

* Super voxels (and cellular organelles) are stored in the SegmentationObject data class ([SO](segmentation_datasets.md)), which are
organized in [SegmentationDatasets](segmentation_datasets.md).

* SyConn principally supports different [backends](backend.md) for data storage, the current default is a simple shared filesystem
(such as lustre, Google Cloud Filestore or AWS Elastic File System).

* Agglomerated super voxels (SVs) are implemented as SuperSegmentationObjects ([SSO](super_segmentation_objects.md)). The collection
 of super-SVs are usually defined in a region adjacency graph (RAG) which is used to initialize the SuperSegmentationDataset
  ([SSD](super_segmentation_datasets.md)).

* [Skeletons](skeletons.md) of (super) super voxels, usually computed from variants of the TEASAR algorithm (https://ieeexplore.ieee.org/document/883951).

* [Mesh](meshes.md) generation and representation of SOs

* Multi-view representation of SSOs (see docs for [glia](glia_removal.md) and [neuron](neuron_analysis.md) analysis; [preprint](https://www.biorxiv.org/content/early/2018/07/06/364034) on biorXiv)


## Flowchart of SyConn

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSY7p2boPxb9OICxNhSrHQlvuHTBRbSMeIOgQ4_NV6pflxc0FKJvPBtskYMAgJsX_OP-6CNmb08tLC5/pub?w=1920&amp;h=1024">
