# Example run


## Prerequisites
The following data should be located in in the current directory 
(they will be copied to the `working_directory` if they are not 
there yet). Example data with all pre-trained models can be found 
in ???
* models
* RAG file 
* folders with data 

## PARAMETERS
* `scale` voxel size in nano meters
* `chunk_size` size of a cube that is processed by a single worker
* `bb` min and max coordinate of the cube of interest;'None' == the whole dataset
* `n_folders_fs` and `n_folders_fs_sc` number of folders to create 
a hierarchy for storing information about cells and subcellular elements
* in `key_val_pairs_conf`:
    * `pyopengl_platform`: string, possible 'egl' ... 
    * `batch_proc_system`: string, possible 'SLURM',...
    * `prior_glia_removal`: bool, 'True' or 'False'
    * `ncores_per_node`, `ngpus_per_node`, `nnodes_total`: intigers

## Remark

The transform functions will be applied when loading the segmentation 
data of cell organelles in order to convert them into binary fore- and 
background currently using `dill` package to support lambda expressions, 
a weak feature. 
Make sure all dependencies within the lambda expressions are imported 
in `QSUB_gauss_threshold_connected_components.py` (here: numpy)