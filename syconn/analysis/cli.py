from __future__ import print_function
from logging import log

from neuroglancer.local_volume import LocalVolume

from syconn.handler.logger import log_main as log_gate
from syconn import global_params
from syconn.analysis.backend import SyConnBackend
from syconn.analysis.neuroShaders import rgb, jet
from sources import SkeletonSource
from utils import _upload_individuals, mesh_task
from flask_server import createDownloadUrl

import json
import argparse
import os
import numpy as np
from knossos_utils import KnossosDataset

import neuroglancer
import neuroglancer.cli
import multiprocessing as mp
import threading as th

ATTRIBUTES = ('sv', 'mi', 'sj', 'vc')
BASE_DIRECTORY = os.path.expanduser('/wholebrain/u/amancu/SyConn/example_cube2/meshes')


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
        msg = f'Could not find synapse results in working directory {global_params.config.working_dir}.'
        logger.error(msg)
        raise RuntimeError(msg)

    backend = SyConnBackend(global_params.config.working_dir, logger)

    return backend


def configure_viewer(backend: SyConnBackend, state, raw_dataset=None, seg_dataset=None, dimensions=None):
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

    # set local volume dimensions if not provided
    scales = seg_dataset.scale
    if dimensions is None:
        dimensions = neuroglancer.CoordinateSpace(
            names=['z', 'y', 'x'],
            units='nm',
            scales=[scales[2], scales[1], scales[0]],
        )

    state.layers.append(
        name='img',
        layer=neuroglancer.ImageLayer(
            source=[
                neuroglancer.LocalVolume(
                    dataset=raw_dataset,
                    dimensions=dimensions,
                    backend=backend,
                    precomputedMesh=False,
                    object_type='sv',
                    volume_type='image',
                    chunk_layout='isotropic',
                    downsampling='3d',
                    task_type='highest_then_upsample'
                ),
            ],
        )
    )
    state.selected_layer.layer = 'img'
    state.selected_layer.visible = True

    # seg_data = dataset.load_seg(offset=(0, 0, 0), size=dataset.boundary, mag=1)
    # print(f"Data Shape: {seg_data.shape}")
    state.layers.append(
        name='segmentation_sv',
        layer=neuroglancer.SegmentationLayer(
            source=[
                neuroglancer.LocalVolume(
                    dataset=seg_dataset,
                    dimensions=dimensions,
                    backend=backend,
                    precomputedMesh=True,
                    object_type='sv',
                    volume_type='segmentation',
                    chunk_layout='isotropic',
                    downsampling='3d',
                    task_type='highest_then_upsample'
                ),
                SkeletonSource(dimensions, backend),
            ],
            segment_query=', '.join(str(id) for id in backend.ssv_list().get('ssvs')),
            mesh_silhouette_rendering=2,
        )
    )
    state.selected_layer.layer = 'segmentation_sv'
    state.selected_layer.visible = True
    # Configure skeleton layer
    # Adjust the skeleton rendering options
    state.layers[1].skeleton_rendering.mode2d = 'lines'
    state.layers[1].skeleton_rendering.line_width2d = 3
    state.layers[1].skeleton_rendering.mode3d = 'lines'
    state.layers[1].skeleton_rendering.line_width3d = 1

    # render mitochondria if required
    state.layers.append(
        name='mitochondria',
        layer=neuroglancer.SegmentationLayer(
            source='precomputed://http://127.0.0.1:8001/mi',
            # segment_colors={id: '#FF0000' for id in backend.ssv_list().get('ssvs')},
            # segment_query=', '.join(str(id) for id in backend.ssv_list().get('ssvs')),
            linked_segmentation_layer='segmentation_sv',
        )
    )
    state.selected_layer.layer = 'mitochondria'
    state.selected_layer.visible = True

    state.layers.append(
        name='vesicle clouds',
        layer=neuroglancer.SegmentationLayer(
            source='precomputed://http://127.0.0.1:8001/vc',
            # segment_colors={id: '#00FF00' for id in backend.ssv_list().get('ssvs')},
            # segment_query=', '.join(str(id) for id in backend.ssv_list().get('ssvs')),
            linked_segmentation_layer='segmentation_sv',
        )
    )
    state.selected_layer.layer = 'vesicle clouds'
    state.selected_layer.visible = True

    state.layers.append(
        name='synaptic junctions',
        layer=neuroglancer.SegmentationLayer(
            source='precomputed://http://127.0.0.1:8001/sj',
            # segment_colors={id: '#0000FF' for id in backend.ssv_list().get('ssvs')},
            # segment_query=', '.join(str(id) for id in backend.ssv_list().get('ssvs')),
            linked_segmentation_layer='segmentation_sv',
        )
    )
    state.selected_layer.layer = 'synaptic junctions'
    state.selected_layer.visible = True


############################################################
# Get Syconn data and transform it to support Neuroglancer #
############################################################

if __name__ == '__main__':
    """
    Start the Neuroglancer server with the desired Viewer 
    and give the data to the Neuroglancer Tornado server
    
    [DEV] To run test server: 
    python -i cli.py --wd=<WORKING_DIRECTORY> --host=<HOST> --port=<PORT>
    """
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    global backend
    # dataset_path = '../../../../../../ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v2'
    if args.wd == '':
        logger.error('No working directory selected... Aborting')

    global_params.wd = os.path.expanduser(args.wd)
    # print(global_params.config.working_dir)
    backend = configure_backend()

    # load seg and raw data
    seg_path = global_params.config.kd_seg_path
    if os.path.basename(os.path.dirname(seg_path)) == 'latest_seg':
        raw_path = '/wholebrain/songbird/j0251/j0251_72_clahe2'
    else:
        raw_path = seg_path

    print(f"Raw path: {raw_path}")
    print(f"Seg path: {seg_path}")
    seg_dataset = KnossosDataset(seg_path)
    raw_dataset = KnossosDataset(raw_path)

    # flask server here
    flask_server = th.Thread(target=createDownloadUrl, args=('127.0.0.1', 8001, backend, logger, seg_path, True,))
    flask_server.start()

    viewer = neuroglancer.Viewer()
    logger.info('Neuroglancer viewer object initialized')

    # configure viewer
    with viewer.txn() as s:
        configure_viewer(backend, s, raw_dataset=raw_dataset, seg_dataset=seg_dataset)

    logger.info('Neuroglancer server running at {}'.format(viewer))
    flask_server.join()
