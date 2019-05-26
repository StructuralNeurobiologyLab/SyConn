# Documentation

## Installation
* Python 3.6
* The whole pipeline was designed and tested on Linux systems
* SyConn is based on the packages [elektronn3](https://github.com/ELEKTRONN/elektronn3) and [knossos-utils](https://github.com/knossos-project/knossos_utils)
* [KNOSSOS](http://knossostool.org/) is used for visualization and annotation of 3D EM data sets.


1. We recommend installing Anaconda and setting up a python environment:
```
conda create -n pysy python=3.6 anaconda
conda activate pysy
```
2. a) Either git clone and install syconn with all prerequisites:
```
git clone https://github.com/StructuralNeurobiologyLab/SyConn.git
cd SyConn
sh install.sh
```

2. b) Or alternatively run these commands to install them manually:
```
conda install cmake
conda install vigra -c conda-forge
conda install mesa -c menpo
conda install osmesa -c menpo
conda install freeglut
conda install pyopengl
conda install snappy
conda install python-snappy
conda install tensorboard tensorflow
conda install -c conda-forge sip=4.18.1

git clone https://github.com/StructuralNeurobiologyLab/SyConn.git
cd SyConn
pip install -r requirements.txt
pip install -e .
```


## Example run
Place the example and model data (provided upon request) in `SyConn/scripts/example_run/`,
cd to `SyConn/scripts/example_run/` and run
```
python start.py
```

The example script analyzes the EM data based on KnossosDatasets (see `knossos_utils`) of the cell segmentation, probability maps of cell organelles
(mitochondria, vesicle clouds and synaptic junctions) and synapse type (inhibitory, excitatory).

On a machine with 20 CPUs (Intel(R) Xeon(R) @ 2.60GHz) and 2 GPUs (GeForce GTX 980 Ti) SyConn
finished the following analysis steps for an example cube with shape \[400 400 600] after 00h:09min:32s.

\[0/8] Preparation          00h:00min:08s       1%

\[1/8] SD generation        00h:04min:35s       48%

\[2/8] SSD generation       00h:00min:13s       2%

\[3/8] Neuron rendering     00h:00min:40s       7%

\[4/8] Synapse detection    00h:01min:26s       15%

\[5/8] Axon prediction      00h:01min:07s       11%

\[6/8] Spine prediction     00h:00min:54s       9%

\[7/8] Celltype analysis    00h:00min:22s       3%

\[8/8] Matrix export        00h:00min:02s       0%


## SyConn KNOSSOS viewer
The following packages have to be available for the system's python2 interpreter
(will differ from the conda environment):

- numpy

- lz4

- requests

In order to inspect the resulting data via the SyConnViewer KNOSSOS-plugin follow these steps:

- Wait until `start.py` finished. For restarting the server run `SyConn/scripts/kplugin/server.py --working_dir=<path>`
pointing to your working directory (`<path>`). The server address and port will be printed here.

- Download and run the nightly build of KNOSSOS (https://github.com/knossos-project/knossos/releases/tag/nightly)

- In KNOSSOS -> File -> Choose Dataset -> browse to your working directory and open
`knossosdatasets/seg/mag1/knossos.conf` with enabled 'load_segmentation_overlay' (at the bottom of the dialog).

- Then go to Scripting (top row) -> Run file -> browse to `SyConn/scripts/kplugin/syconn_knossos_viewer.py`, open it and enter
the port and address of the syconn server.

- After the SyConnViewer window has opened, the selection of segmentation fragments in the slice-viewports (exploration mode) or in the
list of cell IDs followed by pressing 'show neurite' will trigger the rendering of the corresponding cell reconstruction mesh in the 3D viewport.
 The plugin will display additional information about the selected cell and a list of detected synapses (shown as tuples of cell IDs;
 clicking the entry will trigger a jump to the synapse location) and their respective
 properties. In case the window does not pop-up check Scripting->Interpreter for errors.


## Analysis steps
After initialization of the SDs (SVs and cellular organelles) and the SSD
containing the agglomerated SVs, several analysis steps can be applied:

* [Optional] [Glia removal](glia_removal.md)

* [Neuronal morphology analysis and classification](neuron_analysis.md) to identify cellular
compartments (e.g. axons and spines) and to perform morphology based cell type classification.

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
