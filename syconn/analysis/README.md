# SyConn Client

---
The SyConn client starts up Neuroglancer to render the meshes of all cells and organelles.
Currently Neuroglancer can render the following meshes:
* Cells ('sv')
* Mitochondria ('mi')
* Synapses ('syn_ssv') and Synaptic Junctions ('sj')
* Vesticle Clouds ('vc')


#Input

The client can be run standalone and supports the following input: \
**1. `wd`**:
* Working directory where the SyConn client gets the desired dataset.
* Defaults to the SyConn global config working directory

```
--wd=/wholebrain/u/<your_username>/...
```

**2. `host`**:
* Host adress for Neuroglancer

```
--host=localhost
```

**2. `port`**:
* Port of the adress for Neuroglancer

```
--port=5000
```

**For no adress/port input, the default adress where Neuroglancer is found on 127.0.0.1:5000**

#How it works

The SyConn client starts up Neuroglancer and acts as a client to the Neuroglancer server, where it passes the host and port arguments.
SyConn backend is also initialized with the corresponding working directory coming from the arguments or global SyConn config working directory.
Data is parsed from Knossos and is put in `Neuroglancer.viewer` layers, then passed on to Neuroglancer for processing.

The layers with the following sources:\
**1.** `ImageLayer` for the ground truth 
* Raw Knossos dataset encapsulated in a `LocalVolume` object

**2.** `SegmentationLayer` for the segmentation data 
* Segmented Knossos dataset encapsulated in a `LocalVolume` object

**3.** `SegmentationLayer` for each type of organelle mesh mentioned above linked to the segmentation data
* Precomputed links according to [Neuroglancer Legacy single-resolution mesh format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md#legacy-single-resolution-mesh-format)

At the end the link for the data visualisation is prompted. 