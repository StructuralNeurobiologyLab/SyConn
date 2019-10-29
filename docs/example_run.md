# Example run

Starting script for an exemplary SyConn run to process 3D electron 
microscope (EM) data. Some exemplary data to run the `example_run`
are available upon request 
??? MAYBE SOME CONTACT DATA ????. 

# An exemplary data structure
The folder contains an exemplary 3D EM data.
Two datasets are provided: cube1 of size 400 x 400 x 600 and 
a bigger cube3 of size 2180 x 218 x 1140.
* `data1` / `data3`- contain: 
    * h5 files that store raw data (`raw.h5`) and
    segmentation information about cells (`seg.h5`) and other cellular 
    organelles like mitochondria (`mi.h5`), synaptic junctions (`sj.h5`), 
    vesicle clouds (`vc.h5`), symmetric and asymmetric synapses (`sym.h5` and 
    `asym.h5`, respectively). Each cell and cell component has its id greater than zero.
    The id of the background (membranes and intracellular spaces) is 0. 
    * `rag.bz2` and `neuron_rag.bz2` files contain lists of edges of
    Resource Allocation Graph (RAG) that describes the neurons structure.
* `models`- folder with pretrained Convolutional Neural Network models 
to detect cell types, glia, myelins, spines, synapses types etc. 


## Prerequisites
The following data should be located in in the current directory 
(they will be copied to the `working_directory` if they are not 
there yet):
* models 
* data1/ data3

## COMMAND LINE ARGUMENTS
* `--working_dir`- can be given as a command line argument or defined 
by the user inside the script.  
* `--example_cube`- integer: 1 or 3. Id if the cube of interest. 
The default one is 1.


## BASIC PARAMETERS
* `scale` numpy array; voxel size in nano meters
* `prior_glia_removal` boolean
* `chunk_size` touple; size of a cube that is processed by a single worker
* `n_folders_fs` and `n_folders_fs_sc` number of folders in the folder structure
to create a hierarchy for storing information about cells and subcellular elements
* `experiment_name`; string
* Each working directory has its own `config.yml` file that stores dataset 
specific parameters (for more detailed information see [config](config.md)).
To add further parameters to the `config.yml` file, they have to be specified
in the `key_val_pairs_conf` list of tuples, here:
    * `pyopengl_platform`: string, possible 'egl' ... 
    * `batch_proc_system`: string, possible 'SLURM',...
    * `prior_glia_removal`: boolean
    * `ncores_per_node`, `ngpus_per_node`, `nnodes_total`: intigers

* `bb` numpy array; min and max coordinates of the cube of interest will 
be fetched from the annotation file;
* `offset` numpy array
* `bd` numpy array; effectively dimensions of the cube of interest


## LOGGING
For the example run a convenient system of log files is provided.
There is created one main log file for the whole run and other for 
subsequent steps of the pipeline.
All log files are located in `log` folder inside working directory.

## INITIALIZE DATA
Data in the form of h5 files have to be converted into knossos datasets. 
For more information about knossos data format check [KNOSSOS](http://knossostool.org/)

## START SyConn
Data processing is divided into nine steps that have to be run subsequently.

## Remark

The transform functions will be applied when loading the segmentation 
data of cell organelles in order to convert them into binary fore- and 
background currently using `dill` package to support lambda expressions, 
a weak feature. 
Make sure all dependencies within the lambda expressions are imported 
in `QSUB_gauss_threshold_connected_components.py` (here: numpy)