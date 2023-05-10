.. _Instuctions:

Instructions
============
This chapter provides some basic knowledge in order to use 
SyConn appropiatly. You will find the main points on this page 
divided into the subchapters


.. contents::
   :local:

More details are linked in the respective chapters but unfortunately 
you have to use the back button in your browser to return to this page
(documentation in progress).


Installation
------------

- Python 3.7
- The whole pipeline was designed and tested on Linux systems

Before you can set up SyConn, ensure that the
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__
package manager is installed on your system. Then you can install SyConn
and all of its dependencies into a new conda
`environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`__
named “syconn2” by running:

::

   git clone https://github.com/StructuralNeurobiologyLab/SyConn
   cd SyConn
   conda env create -f environment.yml -n syconn2 python=3.7
   conda activate syconn2
   pip install -e .

The last command will install SyConn in
`editable <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__
mode, which is useful for development on SyConn itself. If you want to
install it as a regular read-only package instead, replace the last
command with:
::

   pip install .

To update the environment, e.g. if the environment file changed, use:

::

   conda env update --name syco --file environment.yml --prune

If you encounter

::

    /lib64/libm.so.6: version `GLIBC_2.27' not found

with open3d, you need to upgrade your system or downgrade open3d to
``<=0.9``.

ANM: When creating the environment, make sure to have a available GPU in
order to install pytorch on a GPU instead a CPU.  
.. this has (what ???) computational reasons. 
To test for this Run: nvidia-smi if you dont see an row according to at 
least one GPU you dont have one in use. To change this, run::

    srun –time=2-0 –gres=gpu:1 –mem=100000 –tasks 1 –cpus-per-task 8 –pty bash

To check if it worked properly see if pytorch is build with a **cuda** driver::

   conda activate syconn2
   conda list pytorch



Example run
-----------

Place the example data and models (provided upon request) in
``~/SyConnData/``, cd to ``SyConn/examples/`` and run::

   python start.py [--working_dir=..]

The example script analyzes the EM data together with the cell
segmentation, probability maps of sub-cellular structures (mitochondria,
vesicle clouds and synaptic junctions) and synapse type (inhibitory,
excitatory). For adding further cell organelles to this pipeline see
:doc:`here <cellorganelle_integration>`.

The data format for raw image and segmentation data is based on
``KnossosDataset`` (see
`knossos_utils <https://github.com/knossos-project/knossos_utils>`__).

On a machine with 20 CPUs (Intel Xeon @ 2.60GHz) and 2 GPUs (NVidia
Quadro RTX 5000) SyConn finished the following analysis steps for an
example cube of shape [1100 1100 600] (1.452e-06 mm^3; 0.726 GVx) after
00h:31min:46s.

[1/11] Preparation 0d:0h:1min:20s 4.2%

[2/11] Dense predictions 0d:0h:1min:2s 3.3%

[3/11] SD generation 0d:0h:3min:55s 12.3%

[4/11] SSD generation 0d:0h:0min:33s 1.7%

[5/11] Skeleton generation 0d:0h:8min:35s 27.0%

[6/11] Synapse detection 0d:0h:5min:35s 17.6%

[7/11] Contact detection 0d:0h:0min:0s 0.0%

[8/11] Compartment predictions 0d:0h:6min:4s 19.1%

[9/11] Morphology extraction 0d:0h:2min:7s 6.7%

[10/11] Celltype analysis 0d:0h:2min:23s 7.5%

[11/11] Matrix export 0d:0h:0min:7s 0.4%

Example scripts and API usage
-----------------------------

An introduction on how to use the example scripts can be found
:doc:`here <examples>` and API code examples :doc:`here <api>`.

Flowchart of SyConn
-------------------

.. image:: https://docs.google.com/drawings/d/e/2PACX-1vSY7p2boPxb9OICxNhSrHQlvuHTBRbSMeIOgQ4_NV6pflxc0FKJvPBtskYMAgJsX_OP-6CNmb08tLC5/pub?w=2880&h=1200


Package structure and data classes
----------------------------------

The basic data structures and initialization procedures are explained in
the following sections:

-  SyConn operates with a pre-defined :doc:`working directory and config files <config>`

-  Supervoxels (and cellular organelles) are organized as
   ``SegmentationObject`` which are handled by the
   ``SegmentationDatasets``. For a more detailed description see
   :doc:`here <segmentation_datasets>`.

-  SyConn principally supports different :doc:`backends <backend>` for
   data storage. The current default is a simple shared filesystem (such
   as lustre, Google Cloud Filestore or AWS Elastic File System).

-  Agglomerated supervoxels (SVs) are implemented as
   SuperSegmentationObjects (:doc:`SSO <super_segmentation_objects>`).
   The collection of super-SVs are usually defined in a region
   supervoxel graph which is used to initialize the
   SuperSegmentationDataset (:doc:`SSD <super_segmentation_datasets>`).

-  :doc:`Skeletons <skeletons>` of (super-) supervoxels, usually
   computed from variants of the TEASAR algorithm
   (https://ieeexplore.ieee.org/document/883951) - currently a fall-back
   to a sampling procedure is in use.

-   :doc:`Mesh <meshes>` generation and representation of supervoxels

-  Multi-view representation of neuron reconstructions for
   :doc:`glia <glia_removal>` and :doc:`neuron <neuron_analysis>`
   analysis (published in `Nature
   Communications <https://www.nature.com/articles/s41467-019-10836-3>`__)

Analysis steps
--------------

After initialization of the SDs (cell and sub-cellular structures, step
1 in the example run) and the SSD containing the agglomerated cell SVs
(step 3), several analysis steps can be applied:

-  [Optional] :doc:`Glia removal <glia_removal>`

-  :doc:`Neuronal morphology analysis and classification <neuron_analysis>` 
   to identify cellular compartments (e.g. axons and spines) and to perform 
   morphology based cell type classification (steps 3-7).

-  :doc:`Contact site extraction <contact_site_extraction>` (step 4)

-  :doc:`Identification of synapses and extraction of a wiring diagram <contact_site_classification>`
   (steps 4 and 8)

SyConn KNOSSOS viewer
---------------------

The following packages have to be available in the system’s python2
interpreter (will differ from the conda environment):

-  numpy
-  lz4
-  requests

In order to inspect the resulting data via the SyConnViewer
KNOSSOS-plugin follow these steps:

-  Wait until ``start.py`` finished. For starting the server manually
   run ``syconn.server --working_dir=<path>`` which executes
   ``syconn/kplugin/server.py`` and allows to visualize the analysis
   results of the working directory at (``<path>``) in KNOSSOS. The
   server address and port will be printed.

-  Download and run the nightly build of KNOSSOS
   (https://github.com/knossos-project/knossos/releases/tag/nightly)

-  In KNOSSOS -> File -> Choose Dataset -> browse to your working
   directory and open ``knossosdatasets/seg/mag1/knossos.conf`` with
   enabled ‘load_segmentation_overlay’ (at the bottom of the dialog).

-  Then go to Scripting (top row) -> Run file -> browse to
   ``syconn/kplugin/syconn_knossos_viewer.py``, open it and enter the
   port and address of the syconn server.

-  After the SyConnViewer window has opened, the selection of
   segmentation fragments in the slice-viewports (exploration mode) or
   in the list of cell IDs followed by pressing ‘show neurite’ will
   trigger the rendering of the corresponding cell reconstruction mesh
   in the 3D viewport. The plugin will display additional information
   about the selected cell and a list of detected synapses (shown as
   tuples of cell IDs; clicking the entry will trigger a jump to the
   synapse location) and their respective properties. In case the window
   does not pop-up check Scripting->Interpreter for errors.
