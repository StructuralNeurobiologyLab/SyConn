# Documentation

## Package structure and data classes
The basic data structures and initialization procedures are explained in the following sections:

* SyConn operates with pre-defined [working directory and config files](config.md)

* Super voxels (and cellular organelles) are stored in the SegmentationObject data class ([SO](segmentation_datasets.md)), which are
organized in [SegmentationDatasets](segmentation_datasets.md).

* SyConn principally supports different [backends](backend.md) for data storage

* Agglomerated super voxels (SVs) are implemented as SuperSegmentationObjects ([SSO](super_segmentation_objects.md)). The collection
 of super-SVs are usually defined in a region adjacency graph (RAG) which is used to initialize the SuperSegmentationDataset
  ([SSD](super_segmentation_datasets.md)).

* [Skeletons](skeletons.md) of (super) super voxels

* [Mesh](meshes.md) generation and representation of SOs

* Multi-view representation of SSOs (see [glia](glia_removal.md) and [neuron](neuron_analysis.md) analysis)


## Analysis steps
After initialization of the SDs (SVs and cellular organelles) and SSD (the segmentation defined by agglomerated SVs) SyConn allows
the application of several analysis procedures:

* [Optional] [Glia removal](glia_removal.md)

* [Neuron analysis](neuron_analysis.md) such as cellular compartment, spine and cell type classification

* [Contact site extraction](contact_site_extraction.md)

* [Identification of synapses and extraction of a wiring diagram](contact_site_classification.md)


