# Neuronal morphology analysis and classification
All scripts used for the analysis of the neuron segmentation are located in `SyConn/scripts/multiviews_neuron/`.

## Prerequisites
* [Optional] [Glia removal](glia_removal.md)
* [Mapped cellular organelles](object_mapping.md) to SSVs
* Knossos- and SegmentationDataset of the super voxel segmentation
* SegmentationDatasets for all cellular organelles (currently mitochondria, vesicle clouds and synaptic junctions)
* Initial RAG/SV-mapping

## Steps
The multi-views which contain channels for cell objects and SSV outline
 are the basis for predicting cell compartments, cell type and spines.
* SSV multi-views generation: `start_sso_rendering.py`
* Cell compartment prediction: `axoness_prediction.py`
* Cell type prediction: `celltype_prediction.py`
* Cell compartment prediction: `spiness_prediction.py` [TO BE DONE]


