.. _examples:

********
Examples
********

This page gives an example for a full run of  then main components of SyConn.
The corresponding code is available in :ref:`/syconn/main.py`.
The example data set with all pre-trained models can be downloaded at
`to be added <http://>`_.


Running the example
-------------------
Given that the example zipfile was extracted, running the analysis would just be::

   python2 SyConn/syconn/main.py </path/to/example_folder> [--gpus {None | <int> <int> ...}]

By default the CNN prediction uses only CPUs (None option; up to as many cores as specified in the ~/.theanorc).
GPUs can be made available by adding their ids as shown in nvidia-smi.



Data Set
--------
This data set is a subset of the zebra finch area X dataset j0126 acquired by
`Jörgen Kornfeld <http://www.neuro.mpg.de/mitarbeiter/43611/3242756>`_.


.. figure::  images/raw_cube_tracings.png
   :scale:   60 %
   :align:   center

   Left: Raw data cube Right: Human cell tracings

The above figure shows cell tracings in 3D EM data cube, which is the
required data to perform automated structural analysis using SyConn.
The following section is a walk-through of all major steps.


Applying SyConn -- step by step
-------------------------------
At first, membrane and intracellular space is predicted as barrier (black) and cell
components, such as mitochondria (blue), vesicle clouds (green) and synaptic junctions (red)
are inferred using a CNN with a 3D perceptive field of view (image below). Additionally, another
3D CNN predicts symmetric and asymmetric synaptic junction regions which get combined with
the predicted synaptic junctions later on.


.. figure::  images/raw_img_overlay_153.png
   :scale:   60 %
   :align:   center

   Left: Slice of the raw cube; Right: Overlayed CNN probability maps

Thresholding and connected components analysis then reveal volumetric objects for each
class. Based on the cell tracings and the barrier prediction a cell hull sampling
is performed which enables the mapping of these objects to individual cells (embedding
tracings in biological environment). Furthermore, contact sites of adjacent
cell pairs can be distinguished from synapses combining properties of corresponding
contact site and, if existent, synaptic junction.


.. figure::  images/mapped_hull_synapse_new.png
   :scale:   40 %
   :align:   center


With these priors sub-cellular compartments can be inferred (axon/dendrite/soma and spines in dendrites)
to further assign types (EA, MSN, GP, INT) to every cell and create a wiring
diagram combining knowledge about pre and post synapses (sub-cellular compartments),
area of synaptic junctions and cell functions.

.. figure::  images/connectivity_labeled.png
   :scale:   10 %
   :align:   center