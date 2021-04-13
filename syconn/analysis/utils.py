from syconn import global_params
from storage import MeshStorage
from syconn.handler.logger import log_main as logger
from syconn import global_params
import json
import numpy as np
import os
import shutil
from knossos_utils import KnossosDataset
from neuroglancer.chunks import encode_npz


def get_encoded_mesh(backend, ssv_id, obj_type):
    """
    Get encoded mesh of a specific obj type for ssv_id.
    :param ssv_id: int
    :param obj_type: str
    :return: bytes
    """
    logger.info('Getting binary encoded {} mesh for ssv_id {}'.format(obj_type,
                                                                      global_params.config.working_dir.split('/')[-1]))
    mesh = {}
    if obj_type == 'sv':
        try:
            mesh = backend.ssv_mesh(ssv_id)
        except:
            logger.error('{} mesh not available for ssv_id: {}'.format(obj_type, ssv_id))
    else:
        try:
            object_vert = backend.ssv_obj_vert(ssv_id, obj_type)
            object_ind = backend.ssv_obj_ind(ssv_id, obj_type)
        except:
            logger.error('Precomputed mesh not available for ssv_id: {}'.format(obj_type))
        mesh['vertices'] = object_vert['vert']
        mesh['indices'] = object_ind['ind']
    if not mesh:
        logger.error('Mesh could not be retrieved for ssv_id: {}'.format(ssv_id))
    vertices = np.array(mesh['vertices'], dtype=np.float32).reshape(-1, 3)
    indices = np.array(mesh['indices'], dtype=np.uint32).reshape(-1, 3)
    num_vert = len(vertices)
    data = [
        np.uint32(num_vert),
        vertices,
        indices
    ]
    encoded_mesh = b''.join([array.tobytes('C') for array in data])
    return encoded_mesh


def _to_filename(bounds):
    """
    Converts boundaries to file names for file storage
    """
    return '_'.join('0' + '-' + str(bounds[i]) for i in range(len(bounds)))


def _upload_individuals(mesh_dir, progress, mesh_binaries, generate_manifests, lod, boundaries):
    """
    Saves meshes for the Neuroglancer format to files
    """
    boundaryFilename = _to_filename(boundaries)
    storage = MeshStorage(mesh_dir, progress)
    for segid, mesh_binary in mesh_binaries.items():
        storage.put_files([(
            '{}/{}:{}:{}'.format(  # file_path
                storage.get_path(), segid, lod,
                boundaryFilename
            ),
            mesh_binary)],  # content
            content_type=None,
            compress='precomputed',
            compress_level=9,
        )
        if generate_manifests:
            fragments = []
            fragments.append('{}:{}:{}'.format(segid, lod, boundaryFilename))
            storage.put_file(
                file_path='{}/{}:{}'.format(
                    storage.get_path(), segid, lod
                ),
                content=json.dumps({"fragments": fragments}),
                content_type='application/json',
                compress=None
            )


def _dump_encoded_mesh(obj_mesh_path, backend, lod, obj_type):
    mesh_binaries = get_encoded_mesh(backend, obj_type)
    logger.info('Dumping mesh binaries for {}'.format(global_params.config.working_dir.split('/')[-1]))
    for segid, mesh_binary in mesh_binaries.items():
        fragments = []
        fragments.append('{}:{}:{}_mesh'.format(segid, lod, segid))
        content = json.dumps({"fragments": fragments})
        metadata_path = f"{obj_mesh_path}/{segid}:{lod}"
        try:
            with open(metadata_path, "w") as f:
                f.write(content)
        except IOError as err:
            print('Error writing meta data file for ssv id {}'.format(segid))
        mesh_binary_path = f"{obj_mesh_path}/{segid}:{lod}:{segid}_mesh"
        try:
            with open(mesh_binary_path, "wb") as f:
                f.write(mesh_binary)
        except IOError as err:
            print('Error writing binary mesh data for ssv id {}'.format(segid))


def mesh_task(mesh_dir, backend, lod, obj_type, force=False):
    if not os.path.exists(mesh_dir):  # make meshes subdirectory if not present
        os.mkdir(mesh_dir)
    obj_mesh_path = os.path.join(mesh_dir, obj_type)
    if force:  # rewrite binary mesh data of every object type
        if os.path.exists(obj_mesh_path):
            logger.info(
                '[Force=True] Deleting {} meshes subdirectory. Creating info file and dumping binary {} mesh data'.format(
                    obj_type, obj_type))
            shutil.rmtree(obj_mesh_path)
        os.mkdir(obj_mesh_path)
        with open(os.path.join(obj_mesh_path, 'info'), 'w') as f:
            f.write(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
        _dump_encoded_mesh(obj_mesh_path, backend, lod, obj_type)
        return
    else:
        if not os.path.exists(obj_mesh_path):
            logger.info(
                '{} mesh subdirectory does not exist! Creating directory and info file and dumping binary {} mesh data'.format(
                    obj_type, obj_type))
            os.mkdir(obj_mesh_path)
            with open(os.path.join(obj_mesh_path, 'info'), 'w') as f:
                f.write(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
            _dump_encoded_mesh(obj_mesh_path, backend, lod, obj_type)
        else:
            if len(os.listdir(obj_mesh_path)) > 1:
                logger.info('Found non-empty {} meshes subdirectory. Skipping!'.format(obj_type))
                return
            elif len(os.listdir(obj_mesh_path)) == 1:
                logger.info('Dumping binary {} mesh data'.format(obj_type))
                _dump_encoded_mesh(obj_mesh_path, backend, lod)
                return
            elif len(os.listdir(obj_mesh_path)) == 0:
                logger.info(
                    'Found empty {} mesh subdirectory! Creating info file and dumping binary {} mesh data'.format(
                        obj_type, obj_type))
                with open(os.path.join(obj_mesh_path, 'info'), 'w') as f:
                    f.write(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
                _dump_encoded_mesh(obj_mesh_path, backend, lod)
                return