# Integration of new cell-organelle-types

The following lists the steps that need to be taken to
integrate new types of cell organelles into the prediction
and segmentation pipeline:


1. Choose a short but clear name for the cell-organelle-type
   which is supposed to be integrated, which in the following is
   used as an identifier for the cell-organelle-type.<br>
   Examples for implemented cell-organelle-types are: 'mi'
   for mitochondria, 'er' for the endoplasmic reticulum and
   'golgi' for the golgi apparatus.<br>
   In the following, the exemplary cell-organelle-name 'co'
   is used. It will have to be replaced with the chosen name.
   

2. In `syconn/exec/exec_dense_prediction.py` add a function
   `predict_co()`, analogue to the other functions for
   prediction of cell organelles (e.g. `predict_er` or
   `predict_golgi`).<br>
   Within this function make sure to replace the
   cell-organelle-name in the parameter values passed to
   `predict_dense_to_kd()` accordingly; in particular
   `global_params.config.mpath_co` and `target_names=['co']`.<br>
   The simplest case would be to choose `n_channel=2`, one for
   the background volume and one for the corresponding 
   cell-organelle-volume. If different channels are supposed
   to be predicted in this step, these parameters will need to
   be adapted accordingly.
   

3. In `syconn/handler/config.py` in class `DynConfig` add
   property-method `kd_co_path`, which returns
   `self.entries['paths']['kd_co']`.
   
   
4. Also in `syconn/handler/config.py` in class `DynConfig` add
   property-method `mpath_co`, which returns
   `self.model_dir + '/co/model.pts'`.<br>
   Correspondingly in the model directory add a directory co, in
   which the corresponding model for the prediction is stored.
   

5. Also in `syconn/handler/config.py` give the function
   `generate_default_conf()` an additional parameter `kd_co` with
   default value `None`.<br>
   Within the function, set `kd_co` to
   `working_dir + 'knossosdatasets/co/'` if it was passed as
   `None`.<br>
   Within the function add `entries['paths']['kd_er'] = kd_er` to
   write the given path of the knossos_dataset into the
   config-file.
   

6. In `syconn/handler/config.yml` make the following changes:
   * To the list `process_cell_organelles` add the element `co`.
   * Under `paths:` with an indentation add the line `kd_co:`
     (include the `:`).
   * Under `versions:` with an indentation add the line
     `co: version_no`, where `version_no` typically is 0.
   * Under `cell_objects:`, under `min_obj_vx:`, with an
     indentation add the line `co: min_vx_no`, where `min_vx_no`
     is an integer set in a sensible manner. It specifies the
     minimum number of voxels an object needs to have, to be
     identified as an example of the corresponding
     cell-organelle-type.<br>
     E.g. it is 500 for a large structure like the mitochondrion,
     but for a small structure like the synaptic vesicles, it is 1.
   * Under `cell_objects:`, under `probathresholds:`, with an
     indentation add the line `co: thresh`, where `thresh` is the
     probability-threshold for the generation of a segmentation
     dataset from a probability dataset and needs to be set to a
     number. Without any extra information, a reasonable choice
     is `thresh=0.5`.
   * Under `cell_objects:`, under `min_seed_vx:`, with an
     indentation add the line `co: min_seed`, with `min_seed` the
     seed for the watershed-algorithm (see description in the file).
   * Under `cell_objects:`, under `extract_morph_op:`, with an
     indentation add the line `co: morph_ops`, where `morph_ops`
     is a list of possible morphological operations. See
     explanation in file.
   * Under `meshes:`, under `downsampling:`, with an
     indentation add the line `co: mesh_down`, where `mesh_down`
     is a list of length 3, describing the downsampling.
     
All the steps taken above are implemented analogously for `er`
and `golgi` already, which could be used as examples and
base for further organelles to be integrated.
