from __future__ import print_function
from logging import log
from neuroglancer.local_volume import LocalVolume
from syconn.handler.logger import log_main as logger
from syconn import global_params
from syconn.analysis.backend import SyConnBackend
from syconn.analysis.utils import handle_layer_args
# from syconn.analysis.flask_server import start_flask_server
import argparse
import os
import numpy as np
from knossos_utils import KnossosDataset
import neuroglancer
import neuroglancer.cli
import neuroglancer.server as srv
import neuroglancer.settings as st

flask_PORT = 8000 # for development environment

def configure_backend():
    """Setups SyConnBackend object and logger
    
    :return backend:
    :rtype backend: SyConnBackend
    """

    logger.info('SyConn gate server starting up on working directory '
                '"{}".'.format(global_params.wd))

    if not np.any(['syn_ssv' in name for name in os.listdir(global_params.config.working_dir)]):
        msg = f'Could not find synapse results in working directory {global_params.config.working_dir}.'
        logger.error(msg)
        raise RuntimeError(msg)

    backend = SyConnBackend(global_params.config.working_dir, logger, 0.9)

    return backend

class SyConnClient(object):
    """
    Client used to visualize the segmented data.Contains embedded neuroglancer 
    viewer. 

    :param backend: to retrieve the skeleton and mesh data in neuroglancer.LocalVolume
    :type backend: SyConnBackend
    :param seg_path: path of the segmentation data (global_params.config.kd_seg_path)
    :type seg_path: str
    :param organelles: command line arg; supported organelles
    :type organelles: list
    :param clargs: command line args; used for retrieving <HOST> and <PORT> 
    :type clargs: dict

    .. note:: One client instance corresponds to one neuroglancer viewer.
    """

    def __init__(self, backend, seg_path, organelles, clargs, token):
        
        self.backend = backend
        self._seg_path = seg_path

        if os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(seg_path)))) == 'j0251':
            raw_path = '/wholebrain/songbird/j0251/j0251_72_clahe2'
        else:
            raw_path = seg_path

        self._raw_path = raw_path

        logger.info(f"Raw path: {raw_path}")
        logger.info(f"Seg path: {seg_path}")

        self.seg_dataset = KnossosDataset(seg_path)
        self.raw_dataset = KnossosDataset(raw_path)

        # warn the user for no organelles
        if organelles == []:
            logger.info('No organelles selected')

        # self.host = clargs['host']
        # self.port = clargs['port']
        
        viewer = self.viewer = neuroglancer.Viewer(token=token, global_srv=st.global_server)
        logger.info('Neuroglancer viewer object initialized')

        # start flask server with desired seg_dataset
        # self.flask_server = start_flask_server(flask_PORT, backend, self.seg_dataset)
        
        # configure viewer
        with viewer.txn() as s:
            self.configure_viewer(self.backend, s, raw_dataset=self.raw_dataset, seg_dataset=self.seg_dataset,
                                  flask_PORT=flask_PORT, organelles=organelles)

    def __del__(self):
        self.flask_server.join()

    @property
    def seg_name(self):
        """Segmentation name"""

        return self._seg_path.split('/')[-2]

    @property
    def raw_name(self):
        """Raw data name"""

        return self._raw_path.split('/')[-1]

    def configure_viewer(self, backend: SyConnBackend, state, raw_dataset=None, seg_dataset=None, dimensions=None,
                         flask_PORT=8000, organelles=[]):
        """
        Configures the Syconn client so it parses the desired data to 
        Neuroglancer Viewer. Layer visibility depends on ordering. Last layer
        overrides the side panel visibility of all layers.
        
        :param backend: SyConnBackend
        :param state: neuroglancer.viewer_state.ViewerState
        :param data: numpy.ndarray (e.g KnossosDataset)
        :param dimensions: neuroglancer.CoordinateSpace 
            (viewer/layer dimensions)

        .. note::

           The precomputed source uses http://syconn.esc.mpcdf.mpg.de/ in
           the production environment. For a local development environment,
           use http://localhost:[PORT]/ and configure flask-CORS in 
           neuroglancer/python/neuroglancer/flask_server.py
        """

        def append_organelle_layer(state, organelle):
            """
            Organelle mesh colors referenced from 
            SyConn/syconn/analysis/syconn_knossos_viewer.py#L878
            """

            # handle names
            if organelle == 'mi':
                name = 'mitochondria'
                color = "#0099ff" # rgba(0, 153, 255, 255)
            elif organelle == 'sj':
                name = 'synaptic junctions'
                color = "#f03232" # rgba(240, 50, 50, 255)
            elif organelle == 'vc':
                name = 'vesicle clouds'
                color = "#2d954d" # rgba(45, 149, 77, 255)
            else:
                logger.error('Unsupported organelle layer requested')
                return

            state.layers.append(
                name=name,
                layer=neuroglancer.SegmentationLayer(
                    source=f'precomputed://http://syconn.esc.mpcdf.mpg.de/{organelle}',
                    segment_default_color=color,
                    linked_segmentation_group=self.seg_name,
                    linked_segmentation_color_group=False,
                )
            )

            state.selected_layer.layer = name
            state.selected_layer.visible = False

        # set local volume dimensions if not provided
        scales = seg_dataset.scale
        if dimensions is None:
            dimensions = neuroglancer.CoordinateSpace(
                names=['z', 'y', 'x'],
                units='nm',
                scales=[scales[2], scales[1], scales[0]],
            )

        # raw image
        state.layers.append(
            name=self.raw_name,
            layer=neuroglancer.ImageLayer(
                source=[
                    neuroglancer.LocalVolume(
                        dataset=raw_dataset,
                        dimensions=dimensions,
                        backend=backend,
                        precomputedMesh=False,
                        object_type='sv',
                        volume_type='image',
                        # chunk_layout='isotropic',
                        downsampling='3d',
                        task_type='lowest_then_downsample'
                    ),
                ],
            )
        )

        state.selected_layer.layer = self.raw_name
        state.selected_layer.visible = True
        
        # segmentation 
        state.layers.append(
            name=self.seg_name,
            layer=neuroglancer.SegmentationLayer(
                source=[
                    neuroglancer.LocalVolume(
                        dataset=seg_dataset,
                        dimensions=dimensions,
                        backend=backend,
                        precomputedMesh=False,
                        object_type='sv',
                        volume_type='segmentation',
                        # chunk_layot='isotropic',
                        downsampling='3d',
                        task_type='lowest_then_downsample'
                        ),
                    f'precomputed://http://syconn.esc.mpcdf.mpg.de/sv', # ssv mesh
                    f'precomputed://http://syconn.esc.mpcdf.mpg.de/skeletons'
                ],
                mesh_silhouette_rendering=2,
            )
        )

        state.selected_layer.layer = self.seg_name
        state.selected_layer.visible = True

        # Configure skeleton layer
        # Adjust the skeleton rendering options
        # state.layers[1].skeleton_rendering.mode2d = 'lines'
        # state.layers[1].skeleton_rendering.line_width2d = 3
        # state.layers[1].skeleton_rendering.mode3d = 'lines'
        # state.layers[1].skeleton_rendering.line_width3d = 1

        for organelle in organelles:
            append_organelle_layer(state, organelle)

##############################################################
# Retrieve and transform SyConn data to support Neuroglancer #
##############################################################

if __name__ == '__main__':
    """
    Start the SyConn client with the desired Viewer and provide
    segmentation and raw data to the Neuroglancer Tornado server
    Supported organelles = ('mi', 'sj', 'vc')
    
    To run client: 
    python -i cli.py --wd=<WORKING_DIRECTORY> --host=<HOST> --port=<PORT>
        --organelles <list of organelles>
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

    client = SyConnClient(backend, seg_path, args.organelles, dict(host=args.host, port=args.port))
    logger.info('Neuroglancer server running at {}'.format(client.viewer))
