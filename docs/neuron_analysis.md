# Neuron analysis
All scripts used for the analysis of the neuron segmentation are located in `SyConn/scripts/multiviews_neuron/`.

## Prerequisites
* [Optional] [Glia removal](glia_removal.md)
* KNOSSOS- and SegmentationDataset of the super voxel segmentation
* SegmentationDatasets for all cellular organelles (currently mitochondria, vesicle clouds and synaptic junctions)
* Initial RAG/SV-mapping
* [Mapped cellular organelles](object_mapping.md) to SSVs

## Steps
The multi-views which contain channels for cell objects and SSV outline
 are the basis for predicting cell compartments, cell type and spines.
To generate these views run:
`start_sso_rendering.py`
