import argparse
import os
import threading
import time
from typing import Optional
import numpy as np

from knossos_utils import KnossosDataset
from syconn.handler.logger import log_main as logger
from syconn import global_params
from syconn.analysis.backend import SyConnBackend
import neuroglancer
import neuroglancer.cli
from neuroglancer.config import NeuroConfig
from neuroglancer import config


def configure_backend() -> SyConnBackend:
    """Setups SyConnBackend object and logger

    Raises:
        RuntimeError: If synapse segmentations are not available

    Returns:
        SyConnBackend object initialized with logger and working directory
    """    

    logger.info('SyConn gate server starting up in working directory '
                '"{}".'.format(global_params.wd))

    if not np.any(['syn_ssv' in name for name in os.listdir(global_params.config.working_dir)]):
        msg = f'Could not find synapse results in working directory {global_params.config.working_dir}.'
        logger.error(msg)
        raise RuntimeError(msg)

    backend = SyConnBackend(global_params.config.working_dir, logger, 0.9)

    return backend


class SyConnClient(object):
    """Base class for the web client with a configurable viewer state.
    Each instance of this class represents a single viewer instance.
    Inherit this class to create a custom neuroglancer viewer instance
    (see :class:`~syconn.analysis.property_filter.PropertyFilter`).

    The viewer state can be configured by overriding the 
    ``configure_viewer`` method.
    """

    def __init__(self, params: NeuroConfig, token: str, organelles: Optional[list] = None, **kwargs):
        """Initialize the client with the given parameters.

        Args:
            params: neuroglancer server configuration
            token: 40-character hex token for the client
            organelles: list of organelle meshes to be displayed
        """        
        self.params = params
        self.acquisition = params.acquisition
        self.version = params.version

        # warn the user if no organelles are provided
        if organelles == None:
            logger.warning('No organelles selected')

        self.seg_src = kwargs["seg_src"]
        self.img_src = kwargs["img_src"]
        
        initial_state = neuroglancer.ViewerState(
            title=self.acquisition + '_' + self.version,
            position=np.array(self.params["boundary"], dtype=np.float32) // 2,
            crossSectionScale=1e-9,
        )
        
        viewer = self.viewer = neuroglancer.Viewer(token=token)
        # viewer.set_state(initial_state)
    
        with viewer.txn() as s:
            # s.title = self.acquisition + '_' + self.version
            # s.position=np.array(self.params["boundary"], dtype=np.float32) // 2
            self.configure_viewer(s, organelles)

        # with viewer.txn() as s:
        # add layers with different data sources
            

    @property
    def token(self):
        return self.viewer.token

    @property
    def seg_name(self):
        """Segmentation name"""
        
        return self.acquisition + '_' + self.version

    @property
    def raw_name(self):
        """Raw name"""

        if "example_cube" in self.acquisition:
            return self.acquisition + '_' + self.version
        elif "j0251" in self.acquisition:
            return "j0251_72_clahe2"
        elif "j0126" in self.acquisition:
            return "j0126_realigned"

    def configure_viewer(self, state, organelles: Optional[list] = None):
        """Configures the viewer state with the desired image and
        segmentation volumes, skeletons, cell and organelle meshes.
        These objects are datasources and can be provided by the 
        supported datasource protocols (see ``neuroglancer/src/\
            neuroglancer/datasource``).

        Args:
            state: neuroglancer viewer state
            organelles: list of organelle meshes to be displayed

        Note:
            1. The default implementation of this method adds all the
            datasources. Override this method to configure the viewer
            state

            2. Layer visibility is set for the SegmentationLayer with
            the segmentation volume, mesh and skeleton data. Changing
            it will affect the properties behavior in the side panel of
            the viewer.

            3. The source (base url) in the development server is the
            nginx endpoint. The tornado server `host` and `port` are
            set in the `server.py` file. If running the server in
            development mode, make sure the port matches the value in
            the `dev.syconn.esc.mpcdf.mpg.de` nginx config file.  
        """        

        if config.dev_environ:  # development environment
            source = f'http://localhost:9002'  
        
        else:  # production environment
            source = f'https://syconn.esc.mpcdf.mpg.de'

        def append_organelle_layer(state, organelle: str):
            """Creates a mesh datasource layer for the given organelle
            
            Organelle mesh colors referenced from 
            `~syconn/analysis/syconn_knossos_viewer.py#L878`

            Args:
                state: neuroglancer viewer state
                organelle: organelle name

            Returns:
                None: if organelle is invalid 
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
                    source=f'precomputed://' + source + f'/{self.token}/{self.acquisition}/{self.version}/{organelle}',  # organelle mesh
                    segment_default_color=color,
                    linked_segmentation_group=self.seg_name,
                    linked_segmentation_color_group=False,
                )
            )

        # raw image volume
        state.layers.append(
            name=self.raw_name,
            layer=neuroglancer.ImageLayer(
                source=[
                    self.img_src
                ]
            )
        )

        # segmentation volume
        state.layers.append(
            name=self.seg_name,
            layer=neuroglancer.SegmentationLayer(
                source=[
                    self.seg_src,
                    f'precomputed://' + source + f'/{self.token}/{self.acquisition}/{self.version}/sv',  # ssv mesh
                    f'precomputed://' + source + f'/{self.token}/{self.acquisition}/{self.version}/skeletons',  # ssv skeleton
                    f'precomputed://' + source + f'/{self.acquisition}/{self.version}/properties',  # segment properties
                ],
                mesh_silhouette_rendering=2,
            ),
            tab='segments'  # show segments tab with properties
        )

        state.selected_layer.layer = self.seg_name
        state.selected_layer.visible = True

        for organelle in organelles:
            append_organelle_layer(state, organelle)
        
        # Configure skeleton layer
        # Adjust the skeleton rendering options
        # state.layers[1].skeleton_rendering.mode2d = 'lines'
        # state.layers[1].skeleton_rendering.line_width2d = 3
        # state.layers[1].skeleton_rendering.mode3d = 'lines'
        # state.layers[1].skeleton_rendering.line_width3d = 1

        # Uncomment to test the knossos segmentation data source
        # name = 'knossos'
        # state.layers.append(
        #     name=name,
        #     layer=neuroglancer.SegmentationLayer(
        #         source=f'knossos://' + source + f'/{self.acquisition}/{self.version}/segmentation', # knossos segmentation
        #     )
        # )

        # state.selected_layer.layer = name
        # state.selected_layer.visible = True