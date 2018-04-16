# (Multi-)Views
QSUB scripts are located at `SyConn/scripts/mp/`.

For generating the multi-views prior to glia removal run:
`start_sso_rendering_glia_removal.py`

In order to start the glia prediction run:
# TODO

For splitting and generating the glia-free region adjacency graph (RAG) run:
# TODO

Now create a new SSD, the post-glia-removal SSD, and run the analysis to
 assign cell objects (mitochondria, vesicle clouds and synaptic junctions)
 to all its SSVs #TODO: what exactly has to be called for that?

Then we can extract the multi-views which contain channels for cell objects and
 are the basis for predicting cell compartments, cell type and spines (coming soon).

Run:
`start_sso_rendering.py`




