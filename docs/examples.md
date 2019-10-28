# Examples
This section introduces minimal examples for parts of SyConn. The corresponding python scripts
can be found in `SyConn/examples/`.

## Semantic segmentation of spines
* python script:  `SyConn/examples/semseg_spine.py`
* requires model folder in working directory


<img align="left" width="400" height="400" src="./_static/semseg_spiness_raw.png">

<img align="right" width="400" height="400" src="./_static/semseg_spiness_spines.png">
The script needs to be called with at least the `--kzip` argument to specify the location of the
 file which contains the cell reconstruction.
If your SyConn working directory of example cube 1 is not located at the default
location (`~/SyConn/example_cube1/`) the path to the parent directory of SyConn's `models` folder
 has to be specified via the `--working_dir` argument.

To run the script on a k.zip file (filename should contain at least of numeric value, e.g. `1_spineexample.k.zip`) execute the
 following:

```
python SyConn/examples/semseg_spine.py --kzip=~/1_spineexample.k.zip
```

The k.zip file must contain mesh files of the cell, its synaptic junctions, mitochondria and vesicle clouds: `sv.ply`, `sj.ply`, `mi.ply`, `vc.ply`.

After completion you will find a second file which contains the cell mesh colored according to the
model's output.

The following two images show an example of the original data (content of `1_spineexample.k.zip`) and the prediction (content of `1_spineexample_spines.k.zip`; red: spine head, black: dendritic shaft; grey: spine neck; turquoise: soma/axon)


## Semantic segmentation of axons, dendrites and somas
* python script:  `SyConn/examples/semseg_axon.py`
* requires model folder in working directory

```
python SyConn/examples/semseg_axon.py --kzip=~/2_axonexample.k.zip
```

## Semantic segmentation of cell mergers
The prediction script to determine the merge location of two cells.
* Input: kzip file of cell reconstruction
* output: mesh and skeleton kzip of prediction

* python scrpit: `SyConn/examples/semseg_false_merger.py`
* requires model in working directory

```
python SyConn/examples/semseg_false_merger.py --kzip=<path to kzip folder>
```

#### Arguments
- `--kzip`: path to the folder which contains kzip files of cell reconstructions. Each kzip file must contain the following content: `sv.ply`, `mi.ply`, `sj.ply`, `vc.ply`, `meta.pkl`, `skeleton.pkl`, `annotation.xml`. Incomplete kzip files will not be filtered out during prediction.

- `--modelpath`: path to the trained model. E.g. `~/e3training/merger_FCN_v10_2/`

- `--dest_path`: path in where the output kzip file should be stored. 

