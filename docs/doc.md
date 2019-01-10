# Documentation

## Installation
We recommend installing the latest Anaconda release. Then set up an environment:
```
conda create -n py35 python=3.5 anaconda
source activate py35
```
Then install all prerequisites and finally git clone and install syconn:
```
conda install vigra -c ukoethe
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
 branch `phil`; this will be updated soon to work with the `master` branch):
```
conda create -n py36 python=3.6 anaconda
source activate py36
```
Specify the path to this python executable at `syconn/config/global_params.py36path`.
For the SyConn installation in the py36 environment `vigra` can be ignored.

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

## Analysis steps
After initialization of the SDs (SVs and cellular organelles) and the SSD
containing the agglomerated SVs, several analysis steps can be applied:

* [Optional] [Glia removal](glia_removal.md)

* [Neuronal morphology analysis and classification](neuron_analysis.md) to identify cellular compartments (e.g. axons and spines) and to perform morphology based cell type classification.

* [Contact site extraction](contact_site_extraction.md)

* [Identification of synapses and extraction of a wiring diagram](contact_site_classification.md)


## Flowchart of SyConn

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSY7p2boPxb9OICxNhSrHQlvuHTBRbSMeIOgQ4_NV6pflxc0FKJvPBtskYMAgJsX_OP-6CNmb08tLC5/pub?w=1920&amp;h=1024">
