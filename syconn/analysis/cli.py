from __future__ import print_function
from logging import log
from neuroglancer.local_volume import LocalVolume
from syconn.handler.logger import log_main as log_gate
from syconn import global_params
from syconn.analysis.backend import SyConnBackend
from syconn.analysis.neuroShaders import rgb, jet
from syconn.analysis.utils import handle_layer_args
from syconn.analysis.flask_server import start_flask_server
import argparse
import os
import numpy as np
from knossos_utils import KnossosDataset
import neuroglancer
import neuroglancer.cli

flask_PORT = 8000

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

class SyConnClient(object):
    '''
    Client used to visualize the segmented data.
    Contains embedded neuroglancer viewer
    One SyConn client = one Neuroglancer viewer
    '''

    def __init__(self, logger, backend, seg_path, organelles):
        if os.path.basename(os.path.dirname(seg_path)) == 'latest_seg':
            raw_path = '/wholebrain/songbird/j0251/j0251_72_clahe2'
        else:
            raw_path = seg_path

        logger.info(f"Raw path: {raw_path}")
        logger.info(f"Seg path: {seg_path}")

        self.backend = backend
        self.seg_dataset = KnossosDataset(seg_path)
        self.raw_dataset = KnossosDataset(raw_path)

        viewer = self.viewer = neuroglancer.Viewer()
        logger.info('Neuroglancer viewer object initialized')

        # warn the user for no organelles
        if organelles == []:
            logger.info('No organelles selected')

        # start flask server with desired seg_dataset
        self.flask_server = start_flask_server(flask_PORT, backend, self.seg_dataset)

        # configure viewer
        with viewer.txn() as s:
            self.configure_viewer(self.backend, s, raw_dataset=self.raw_dataset, seg_dataset=self.seg_dataset,
                                  flask_PORT=flask_PORT, organelles=organelles)

    def __del__(self):
        self.flask_server.join()

    def configure_viewer(self, backend: SyConnBackend, state, raw_dataset=None, seg_dataset=None, dimensions=None,
                         flask_PORT=8000, organelles=[]):
        """
        Configures the Syconn client so it parses the desired data to Neuroglancer
        Viewer. Layer visibility depends on ordering. Last layer overrides the side panel visibility of all layers
        :param backend: SyConnBackend
        :param state: neuroglancer.viewer_state.ViewerState
        :param data: numpy.ndarray (e.g KnossosDataset)
        :param dimensions: neuroglancer.CoordinateSpace (viewer/layer dimensions)
        """
        def append_organelle_layer(state, organelle):

            # handle names
            if organelle == 'mi':
                name = 'mitochondria'
            elif organelle == 'sj':
                name = 'synaptic junctions'
            elif organelle == 'vc':
                name = 'vesticle clouds'
            else:
                logger.error('Unsupported organelle layer requested')
                return

            state.layers.append(
                name=name,
                layer=neuroglancer.SegmentationLayer(
                    source=f'precomputed://http://127.0.0.1:{flask_PORT}/{organelle}',
                    linked_segmentation_layer='segmentation_sv',
                )
            )

            state.selected_layer.layer = name
            state.selected_layer.visible = True



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
                    f'precomputed://http://127.0.0.1:{flask_PORT}/skeletons'
                ],
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

        for organelle in organelles:
            append_organelle_layer(state, organelle)

############################################################
# Get Syconn data and transform it to support Neuroglancer #
############################################################

if __name__ == '__main__':
    """
    Start the Neuroglancer server with the desired Viewer 
    and give the data to the Neuroglancer Tornado server
    Supported organelles = ('sv', 'mi', 'sj', 'vc')
    [DEV] To run test server: 
    python -i cli.py --wd=<WORKING_DIRECTORY> --host=<HOST> --port=<PORT> --organelles=<list of organelles>
    """
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    handle_layer_args(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    global backend

    if args.wd == '':
        logger.error('No working directory selected... Aborting')

    global_params.wd = os.path.expanduser(args.wd)

    backend = configure_backend()

    # load seg and raw data
    seg_path = global_params.config.kd_seg_path

    client = SyConnClient(logger, backend, seg_path, args.organelles)

    logger.info('Neuroglancer server running at {}'.format(client.viewer))
