from __future__ import print_function
from logging import log

from neuroglancer.local_volume import LocalVolume

from syconn.handler.logger import log_main as log_gate
from syconn import global_params
from syconn.analysis.backend import SyConnBackend
from syconn.analysis.neuroShaders import rgb, jet
from syconn.analysis.sources import SkeletonSource
from syconn.analysis.utils import _upload_individuals, mesh_task

import json
import argparse
import os
import numpy as np
from knossos_utils import KnossosDataset

import neuroglancer
import neuroglancer.cli
import webbrowser

ATTRIBUTES = ('sv', 'mi', 'sj', 'vc')
BASE_DIRECTORY = os.path.expanduser('~/mnt/wholebrain/scratch/hashirah/SyConn/syconn/meshes')

def configure_backend():
    """
    Setups SyConnBackend object and logger

    :return SyConnBackend:
    """
    global logger
    logger = log_gate
    logger.info('SyConn gate server starting up on working directory '
                '"{}".'.format(global_params.wd))

    if not np.any(['syn_ssv' in name for name in os.listdir(global_params.config.working_dir)]):
        msg = 'Could not find synapse results in working directory ' \
              f'{global_params.config.working_dir}.'
        logger.error(msg)
        raise RuntimeError(msg)

    backend = SyConnBackend(global_params.config.working_dir, logger)

    return backend


def configure_viewer(backend: SyConnBackend, state, data=None, dimensions=None):
    """
    Configures the Syconn client so it parses the desired data to Neuroglancer
    Viewer needs to have layers supported by Neuroglancer -> layer_types 
    = { 'image': ImageLayer, 
        'segmentation': SegmentationLayer,
        'pointAnnotation': PointAnnotationLayer, 
        'annotation': AnnotationLayer,   
        'mesh': SingleMeshLayer}
    Layer visibility depends on ordering. Last layer overrides the side panel
    visibility of all layers

    :param backend: SyConnBackend
    :param state: neuroglancer.viewer_state.ViewerState
    :param data: numpy.ndarray (e.g KnossosDataset)
    :param dimensions: neuroglancer.CoordinateSpace (viewer/layer dimensions)
    """
    if data is None:
        # create dummy data
        a = np.zeros((3, 100, 100, 100), dtype=np.uint8)
        ix, iy, iz = np.meshgrid(*[np.linspace(0, 1, n) for n in a.shape[1:]], indexing='ij')
        b = np.cast[np.uint32](np.floor(np.sqrt((ix - 0.5) ** 2 + (iy - 0.5) ** 2 + (iz - 0.5) ** 2) * 10))
        b = np.pad(b, 1, 'constant')

    # set local volume dimensions if not provided
    if dimensions is None:
        dimensions = neuroglancer.CoordinateSpace(
            names=['z', 'y', 'x'],
            units='nm',
            scales=[20, 10, 10],
        )

    # object id dict for color assignment 
    # segment_colors = {id: None for id in backend.ssv_list().get('ssvs')}

    sv_path = os.path.join(BASE_DIRECTORY, 'sv')
    state.layers.append(
        name='segmentation_mesh_skeleton',
        layer=neuroglancer.SegmentationLayer(
            source=[
                # neuroglancer.LocalVolume(
                #     data=data,
                #     dimensions=dimensions,
                #     backend=backend,
                #     precomputedMesh=True,
                #     object_type='sv'
                # ),
                'precomputed://http://127.0.0.1:8000' + sv_path,
                SkeletonSource(dimensions, backend),
            ],
            segment_query=', '.join(str(id) for id in backend.ssv_list().get('ssvs'))
        )
    )
    state.selected_layer.layer = 'segmentation_mesh_skeleton'
    state.selected_layer.visible = True

    # render mitochondria if required
    mito_path = os.path.join(BASE_DIRECTORY, 'mi')
    state.layers.append(
        name='mito',
        layer=neuroglancer.SegmentationLayer(
            source='precomputed://http://127.0.0.1:8000' + mito_path,
            segment_colors={id: '#FF0000' for id in backend.ssv_list().get('ssvs')}
        )
    )
    state.selected_layer.layer = 'mito'
    state.selected_layer.visible = True

    # state.layers.append(
    #     name='mitochondria',
    #     layer=neuroglancer.SegmentationLayer(
    #         source=neuroglancer.LocalVolume(
    #             data=data,
    #             dimensions=dimensions,
    #             backend=backend,
    #             precomputedMesh=True,
    #             object_type='mi'
    #         ),
    #         segment_colors={id: '#FF0000' for id in backend.ssv_list().get('ssvs')},
    #     )
    # )
    # state.selected_layer.layer = 'mitochondria'
    # state.selected_layer.visible = False

    # render vesicle clouds if required
    # state.layers.append(
    #     name='vesicle clouds',
    #     layer=neuroglancer.LocalVolume(
    #         data=data,
    #         dimensions=dimensions,
    #         backend=backend,
    #         precomputedMesh=True,
    #         object_type='vc'
    #     ),
    #     segment_colors={id: '00FF00' for id in backend.ssv_list().get('ssvs')}
    # )
    # state.selected_layer.layer = 'vesicle clouds'
    # state.selected_layer.visible = False
    #
    # # render synaptic junctions if required
    # state.layers.append(
    #     name='synapses/synaptic junctions',
    #     layer=neuroglancer.LocalVolume(
    #         data=data,
    #         dimensions=dimensions,
    #         backend=backend,
    #         precomputedMesh=True,
    #         object_type='sj'
    #     ),
    #     segment_colors={id: 'FFFF00' for id in backend.ssv_list().get('ssvs')}
    # )
    # state.selected_layer.layer = 'synapse junctions'
    # state.selected_layer.visible = False
    # to_precomputed(MESH_DIRECTORY)
    # state.layers.append(
    #     name='mito',
    #     layer=neuroglancer.SegmentationLayer(
    #         source=neuroglancer.LocalVolume(
    #             data=data,
    #             dimensions=dimensions,
    #             backend=backend,
    #             precomputedMesh=True,
    #             object_type='mi'
    #         )
    #     )
    # )
    #
    # state.selected_layer.layer = 'mito'
    # state.selected_layer.visible = True

    # state.layers.append(
    #     name='mito',
    #     layer=neuroglancer.LocalVolume(
    #         data=data,
    #         dimensions=dimensions,
    #         backend=backend,
    #         precomputedMesh=True,
    #         object_type='mi'
    #     ),
    #     mesh_silhouette_rendering=2,
    # )

    # state.selected_layer.layer = 'mito'
    # state.selected_layer.visible = True

    # skeleton and cell mesh combined
    # keep the skeleton source either the first or last layer to adjust rendering options

    # state.layers.append(
    #     name=global_params.config.working_dir.split('/')[-1],
    #     layer=neuroglancer.SegmentationLayer(
    #         source=[
    #             SkeletonSource(dimensions, backend),
    #             # TODO(hashir): independent mesh source
    #             # MeshSource(dimensions, backend, 'mi')
    #         ],
    #         skeleton_shader=jet(),
    #         selected_alpha=0,
    #         not_selected_alpha=0,
    #         # following segments are listed in the 'Seg' tab of side panel
    #         segment_query=', '.join(str(id) for id in backend.ssv_list().get('ssvs')),
    #         mesh_silhouette_rendering=2,
    #     ),
    # )

    # state.selected_layer.layer = global_params.config.working_dir.split('/')[-1]
    # state.selected_layer.visible = True

    # Configure skeleton layer
    if any(layer.name == global_params.config.working_dir.split('/')[-1] for layer in state.layers):
        # Adjust the skeleton rendering options
        state.layers[-1].skeleton_rendering.mode2d = 'lines'
        state.layers[-1].skeleton_rendering.line_width2d = 3
        state.layers[-1].skeleton_rendering.mode3d = 'lines_and_points'
        state.layers[-1].skeleton_rendering.line_width3d = 3


############################################################
# Get Syconn data and transform it to support Neuroglancer #
############################################################

if __name__ == '__main__':
    """
    Start the Neuroglancer server with the desired Viewer 
    and give the data to the Neuroglancer Tornado server
    
    [DEV] To run test server: 
    python -i cli.py --host=<HOST> --port=<PORT>
    """
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    global backend
    # dataset_path = '../../../../../../ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v2'
    if args.wd == '':
        # Andrei path
        # args.wd = '../../../../../SyConn/example_cube1'
        # Hashir path
        # args.wd = '~/SyConn/example_cube2'
        args.wd = '~/mnt/wholebrain/u/hashirah/SyConn/example_cube1'
    global_params.wd = os.path.expanduser(args.wd)
    # print(global_params.config.working_dir)
    backend = configure_backend()

    # segmentation data of example_cube1
    # seg_path = global_params.config.kd_seg_path
    seg_path = os.path.join(global_params.config.working_dir, 'knossosdatasets/seg')
    dataset = KnossosDataset(seg_path)

    # load entire segmentation with magnification 1
    seg_data = dataset.load_seg(offset=(0, 0, 0), size=dataset.boundary, mag=1)
    # seg_data = dataset.load_seg(offset=(0,0,0), size=(256, 256, 256), mag=1)
    # seg_data = backend.get_ssd()

    bounds = np.array([600 * 20, 400 * 10, 400 * 10])*1e-9
    print(bounds)
    # print(dataset.boundary)
    # _upload_individuals(MESH_DIRECTORY, True, create_mesh_binaries(), True, 0, bounds)

    logger.info('Segmentation data of shape {} loaded in memory'.format(seg_data.shape))
    
    for obj_type in ATTRIBUTES:
        mesh_dir = os.path.join(BASE_DIRECTORY, obj_type)
        mesh_task(mesh_dir, backend, 0, bounds, obj_type, force=True)

    # initialize viewer for Neuroglancer
    viewer = neuroglancer.Viewer()
    logger.info('Neuroglancer viewer object initialized')

    # configure viewer
    with viewer.txn() as s:
        configure_viewer(backend, s, seg_data)

    # webbrowser.open_new_tab(viewer.get_viewer_url())
    logger.info('Neuroglancer server running at {}'.format(viewer))
