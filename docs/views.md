# (Multi-)View models
View-related QSUB scripts are located at `SyConn/scripts/glia/` and `SyConn/scripts/multi_views/`.

## Glia removal
_in progress_

For generating the multi-views prior to glia removal run:
`start_sso_rendering_glia_removal.py`

In order to start the glia prediction run:
`glia_prediction.py`

For splitting and generating the glia-free region adjacency graph (RAG) run:
`glia_splitting.py`

## Creating new SuperSegmentationDataset
Now create a new SSD, the post-glia-removal SSD, and run the analysis to
 assign cell objects (mitochondria, vesicle clouds and synaptic junctions)
 to all its SSVs #TODO: what exactly has to be called for that?

## Cellular morphology learning neural networks
Now we can extract the multi-views which contain channels for cell objects and
 are the basis for predicting cell compartments, cell type and spines (coming soon).

Run:
`start_sso_rendering.py`


# Groundtruth generation
TBD