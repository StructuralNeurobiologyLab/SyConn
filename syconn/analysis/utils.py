from syconn import global_params
from syconn.analysis.storage import MeshStorage
from syconn.handler.logger import log_main as logger
from syconn import global_params
import json
import numpy as np
import os
import shutil

def _create_mesh_binaries(backend, obj_type):
    logger.info('Creating mesh binaries for {}'.format(global_params.config.working_dir.split('/')[-1]))

    meshBinaries = {}
    for ssv_id in backend.ssv_list().get('ssvs'):
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
            logger.error('Mesh could not be built for given object_id: {}'.format(ssv_id))

        # vertices = np.array(mesh['vertices'], dtype=np.float32).reshape(-1, 3)[:, [2, 1, 0]] * 1e-9
        vertices = np.array(mesh['vertices'], dtype=np.float32).reshape(-1, 3)
        vertices[:, 0] *= 1
        vertices[:, 1] *= 1
        vertices[:, 2] *= 1

        indices = np.array(mesh['indices'], dtype=np.uint32).reshape(-1, 3)
        # num_vert = len(vertices)
        num_vert = len(vertices)

        data = [
            np.uint32(num_vert),
            vertices,
            indices
        ]
        encoded_mesh = b''.join([array.tobytes('C') for array in data])

        meshBinaries[ssv_id] = encoded_mesh

    return meshBinaries

def _to_filename(bounds):
    """converts boundaries to file names for file storage"""
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

def _dump_mesh_binaries(mesh_dir, backend, lod, boundaries, obj_type):
    mesh_binaries = _create_mesh_binaries(backend, obj_type)

    logger.info('Dumping mesh binaries for {}'.format(global_params.config.working_dir.split('/')[-1]))
    # boundary_file = _to_filename(boundaries)

    for segid, mesh_binary in mesh_binaries.items():
        fragments = []
        fragments.append('{}:{}:{}_mesh'.format(segid, lod, segid))
        
        content = json.dumps({"fragments": fragments})
        
        metadata_path = f"{mesh_dir}/{segid}:{lod}"

        try:
            with open(metadata_path, "w") as f:
                f.write(content)
        except IOError as err:
            print('Error writing meta data file for ssv id {}'.format(segid))

        mesh_binary_path = f"{mesh_dir}/{segid}:{lod}:{segid}_mesh"

        try:
            with open(mesh_binary_path, "wb") as f:
                f.write(mesh_binary)
        except IOError as err:
            print('Error writing binary mesh data for ssv id {}'.format(segid))

def mesh_task(mesh_dir, backend, lod, boundaries, obj_type, force=False):
    subdir = os.path.dirname(mesh_dir)
    if not os.path.exists(subdir): # make meshes subdirectory if not present
        os.mkdir(subdir)

    if force: # rewrite binary mesh data of every object type
        if os.path.exists(mesh_dir): 
            logger.info('[Force=True] Deleting precomputed subdirectory {}. Creating info file and dumping binary mesh data of object type {}'.format(subdir, mesh_dir.split('/')[-1]))
            shutil.rmtree(mesh_dir)
            
        os.mkdir(mesh_dir)

        with open(os.path.join(mesh_dir, 'info'), 'w') as f:
            f.write(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
            
        _dump_mesh_binaries(mesh_dir, backend, lod, boundaries, obj_type)
        return

    elif len(os.listdir(mesh_dir)) > 1:
        logger.info('Found non-empty meshes subdirectory. Skipping!')
        return

    elif len(os.listdir(mesh_dir)) == 1:
        _dump_mesh_binaries(mesh_dir, backend, lod, boundaries)
        return

    elif len(os.listdir(mesh_dir)) == 0:
        with open(mesh_dir+'/info', 'w') as f:
            f.write(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
        
        _dump_mesh_binaries(mesh_dir, backend, lod, boundaries)
        return