# Neuronal morphology analysis and classification
All scripts used for the analysis of the neuron segmentation are located in `SyConn/scripts/multiviews_neuron/`.

## Prerequisites
* \[Optional\] [Glia removal](glia_removal.md)
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
* **Auto-scripts WIP:** Cell compartment prediction: `spiness_prediction.py`

![](images/axoness_3D_2855_4896_4617_28985344.002.png?resize=200,200){.left}
![](images/spine_semseg_3D_7141_6013_4838_28479489_spiness_k5_2views.png?resize=200,200){.right}