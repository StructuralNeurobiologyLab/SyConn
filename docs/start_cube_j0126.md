# Start the SyConn pipeline


## Prerequisites
The following data should be located in 'working_directory' or in the 
current directory (they will be copied to the working_directory)
* models (list of necessary models is specified)
* RAG file 
* raw knossosdatasets

## 

## 'MORE PARAMETERS'
`scale` in nano meters \
`chunk_size` size of a cube that is processed by a single worker \
`bb` min and max coordinate of the cube of interest; 
None = the whole dataset \
`n_folders_fs` and `n_folders_fs_sc` number of folders to create 
a hierarchy for storing information about cells and subcellular elements

## PREPARING DATA



## STH

The transform functions will be applied when loading the segmentation 
data of cell organelles in order to convert them into binary fore- and 
background currently using `dill` package to support lambda expressions, 
a weak feature. 
Make sure all dependencies within the lambda expressions are imported 
in `QSUB_gauss_threshold_connected_components.py` (here: numpy)

