from syconn.handler.logger import log_main as logger
import numpy as np
import time
import json

def get_encoded_skeleton(backend, ssv_id, scales):
    """Gets encoded skeleton for ssv_id.

    :param backend: 
    :type backend: SyConnBackend
    :param ssv_id: 
    :type ssv_id: int
    :param scales: KnossosDataset.scale
    :type scales: numpy.array
    :return encoded_skeleton:
    :rtype encoded_skeleton: bytes, -1 (skeleton not available)
    """

    logger.info('Getting binary encoded skeleton for ssv_id {}'.format(ssv_id))
    
    skeleton = {}
    
    try:
        start = time.time()
        skeleton = backend.ssv_skeleton(ssv_id)
        dtime = time.time() - start
        logger.debug('Got ssv skeleton {} after {:.2f}'.format(ssv_id, dtime))
        nodes = np.array(skeleton["nodes"], dtype=np.float32).reshape(-1, 3)

    except:
        return -1
        
    # accomodate dimension scaling
    nodes[:, 0] *= scales[0] # z
    nodes[:, 1] *= scales[1] # y
    nodes[:, 2] *= scales[2] # x
    
    edges = np.array(skeleton["edges"], dtype=np.uint32).reshape(-1, 2)
    num_vert = nodes.shape[0]
    num_edges = edges.shape[0]

    data = [
        np.uint32(num_vert),
        np.uint32(num_edges),
        nodes,
        edges
    ]

    encoded_skeleton = b''.join([array.tobytes('C') for array in data])

    return encoded_skeleton

def get_encoded_mesh(backend, ssv_id, obj_type):
    """Gets encoded mesh of a specific obj type for ssv_id.

    :param backend: 
    :type backend: SyConnBackend
    :param ssv_id: 
    :type ssv_id: int
    :param obj_type: 'sv', 'mi', 'vc', 'sj'
    :type obj_type: str
    :return encoded_mesh: 
    :rtype encoded_mesh: bytes, -1 (mesh not available)
    """

    logger.info('Getting binary encoded {} mesh {}'.format(obj_type, ssv_id))

    mesh = {}

    if obj_type == 'sv':
        try:
            mesh = backend.ssv_mesh(ssv_id)

        except:
            # logger.error('{} mesh not available for ssv_id: {}'.format(obj_type, ssv_id))
            return -1
    else:
        try:
            start = time.time()
            object_vert = backend.ssv_obj_vert(ssv_id, obj_type)
            object_ind = backend.ssv_obj_ind(ssv_id, obj_type)
            mesh['vertices'] = object_vert['vert']
            mesh['indices'] = object_ind['ind']
            dtime = time.time() - start
            logger.debug('Got {} mesh {} after {:.2f}'.format(obj_type, ssv_id, dtime))

        except:
            # logger.error('{} mesh not available for ssv_id: {}'.format(obj_type, ssv_id))
            return -1

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

def get_mesh_meta(ssv_id, lod):
    fragments = []
    fragments.append("{}:{}:{}_mesh".format(ssv_id, lod, ssv_id))
    meta = json.dumps({"fragments": fragments})

    return meta

_ordinal_mags = False

def mag_scale(self, mag): # get scale in specific mag
    index = mag - 1 if _ordinal_mags else int(np.log2(mag))
    return self.scales[index]

def scale_ratio(mag, base_mag): # ratio between scale in mag and scale in base_mag
    return (mag_scale(mag) / mag_scale(base_mag)) if _ordinal_mags else np.array(3 * [float(mag) / base_mag])

def load_segmentation(path, offset, size, cube_type, from_overlay, mag, expand_area_to_mag=False, padding=0, datatype=None):
    """ Extracts a 3D matrix from the KNOSSOS-dataset NOTE: You should use one of the two wrappers below
    :param offset: 3 sequence of ints
        mag 1 coordinate of the corner closest to (0, 0, 0)
    :param size: 3 sequence of ints
        mag 1 size of requested data block
    :param from_overlay: bool
        loads overlay instead of raw cubes
    :param mag: int
        magnification of the requested data block
        Enlarges area to true voxels of mag in case offset and size don't exist in that mag.
    :param expand_area_to_mag: bool, int
        Enlarges area to true voxels of specified mag in case offset and size don't exist in that mag.
        False: no expansion, True: expansion to ``mag``, int: expansion to ``expand_area_to_mag``
    :param padding: str or int
        Pad mode for matrix parts outside the dataset. See https://www.pydoc.io/pypi/numpy-1.9.3/autoapi/numpy/lib/arraypad/index.html?highlight=pad#numpy.lib.arraypad.pad
        When passing an it, will pad with that int in 'constant' mode
    :param datatype: numpy datatype
        typically: for mode 'raw' this is np.uint8, and for 'overlay' np.uint64
    :return: 3D numpy array or nothing
    """
    available_mags = (1, 2, 4, 8, 16, 32, 64)
    if mag not in (1, 2, 4, 8, 16, 32):
        raise Exception(f'Requested mag {mag} not available, only mags {available_mags} are available.')

    if 0 in size:
        raise Exception(f'The second parameter is size! - at least one dimension was set to 0 ({size})')

    ratio = scale_ratio(mag, 1)

    def mag_scale(mag):  # get scale in specific mag
        index = mag - 1 if ordinal_mags else int(np.log2(mag))
        return self.scales[index]

    def scale_ratio(mag, base_mag):  # ratio between scale in mag and scale in base_mag
        return (mag_scale(mag) / mag_scale(base_mag)) if ordinal_mags else np.array(
            3 * [float(mag) / base_mag])

    def get_intervals(offset, size, cube_coord):
        global_end = offset + size
        out_start = np.maximum(0, cube_coord * self.cube_shape - offset)
        out_end = (cube_coord + 1) * self.cube_shape - global_end
        out_end = size * (out_end >= 0) + out_end * (out_end < 0)  # cube contains this output edge
        incube_start = np.maximum(0, offset - cube_coord * self.cube_shape)
        incube_end = global_end - (cube_coord + 1) * self.cube_shape
        incube_end = self.cube_shape * (incube_end >= 0) + incube_end * (
                    incube_end < 0)  # output contains this cube edge
        return out_start, out_end, incube_start, incube_end

    def read_cube(cube_type, cube_coord):
        out_start, out_end, incube_start, incube_end = get_intervals(offset, size, cube_coord)

        valid_values = False

        # check cache first
        # values = self._cube_from_cache(cube_coord, from_overlay)
        # from_cache = values is not None
        # TODO make caching work
        from_cache = False

        if not from_cache:
            filename = f'{self.experiment_name}_{self.name_mag_folder}{mag}_x{cube_coord[0]:04d}_y{cube_coord[1]:04d}_z{cube_coord[2]:04d}.{"seg.sz.zip" if from_overlay else self._raw_ext}'
            path = f'{path}/{self.name_mag_folder}{mag}/x{cube_coord[0]:04d}/y{cube_coord[1]:04d}/z{cube_coord[2]:04d}/{filename}'

            # if self.in_http_mode:
            #     for tries in range(1, self.http_max_tries + 1):
            #         try:
            #             request = requests.get(path, auth=self.http_auth, timeout=60)
            #             request.raise_for_status()
            #             if not from_overlay:
            #                 if self._raw_ext == 'raw':
            #                     values = np.fromstring(request.content, dtype=np.uint8).astype(datatype)
            #                 else:
            #                     values = imageio.imread(request.content)
            #             else:
            #                 with zipfile.ZipFile(BytesIO(request.content), 'r') as zf:
            #                     snappy_cube = zf.read(zf.namelist()[0]) # seg.sz (without .zip)
            #                     raw_cube = self.module_wide['snappy'].decompress(snappy_cube)
            #                     values = np.fromstring(raw_cube, dtype=np.uint64).astype(datatype)
            #             try:# check if requested values match shape
            #                 values.reshape(self.cube_shape[::-1])
            #                 valid_values = True
            #                 break
            #             except ValueError:
            #                 self._print(f'Reshape error encountered for {1 + tries} time. ({path}). Content length: {len(request.content)}')
            #                 time.sleep(random.uniform(0.1, 1.0))
            #                 if tries == self.http_max_tries:
            #                     raise Exception(f'Reshape errors exceed http_max_tries ({self.http_max_tries}).')
            #         except requests.exceptions.RequestException as e:
            #             if isinstance(e, requests.exceptions.ConnectionError) and tries < self.http_max_tries:
            #                 time.sleep(random.uniform(0.1, 1.0))
            #                 continue
            #             return e
            #         self._print(f'[{path}] Error occured ({tries}/{self.http_max_tries})')
            #     if not valid_values:
            #         raise Exception(f'Max. #tries reached. ({self.http_max_tries})')
            # else:
            if os.path.exists(path):
                try:
                    if from_overlay:
                        with zipfile.ZipFile(path, 'r') as zf:
                            snappy_cube = zf.read(zf.namelist()[0]) # seg.sz (without .zip)
                        raw_cube = self.module_wide['snappy'].decompress(snappy_cube)
                        values = np.fromstring(raw_cube, dtype=np.uint64).astype(datatype)
                    elif cube_type == 'image':
                        flat_shape = int(np.prod(self.cube_shape))
                        values = np.fromfile(path, dtype=np.uint8, count=flat_shape).astype(datatype)
                    else: # snappy compressed file
                        with zipfile.ZipFile(path, 'r') as zf:
                            values = zf.read(zf.namelist()[0])
                    valid_values = True
                except Exception as e:
                    print(f'Reading cube failed: {path}')
                    raise e
            else:
                print(f'Cube »{path}« does not exist, cube with zeros only assigned')

        if valid_values:
            values = values.reshape(self.cube_shape[::-1])
            # if not from_cache:
            #     self._add_to_cube_cache(cube_coord, from_overlay, values)
            output[out_start[2]:out_end[2], out_start[1]:out_end[1], out_start[0]:out_end[0]] \
                = values[incube_start[2]:incube_end[2], incube_start[1]:incube_end[1], incube_start[0]:incube_end[0]]

    # t0 = time.time()
    #
    # assert self.initialized, 'Dataset is not initialized'
    #
    # if mag not in self.available_mags:
    #     raise Exception(f'Requested mag {mag} not available, only mags {self.available_mags} are available.')
    #
    # if 0 in size:
    #     raise Exception(f'The second parameter is size! - at least one dimension was set to 0 ({size})')

    ratio = self.scale_ratio(mag, 1)
    if expand_area_to_mag:
        if expand_area_to_mag is True:
            expand_area_to_mag = mag
        expand_ratio = self.scale_ratio(expand_area_to_mag, 1)
        # mag1 coords rounded such that when converting back from target mag to mag1 the specified offset and size can be extracted.
        # i.e. for higher mags the matrix will be larger rather than smaller
        boundary = np.ceil(np.array(self.boundary, dtype=np.int) / expand_ratio).astype(int)
        end = np.ceil(np.add(offset, size) / expand_ratio) * expand_ratio
        offset = np.floor(np.array(offset, dtype=np.int) / expand_ratio) * expand_ratio
        # offset and size in target mag
        size = ((end - offset) // ratio).astype(int)
        offset = (offset // ratio).astype(int)
    else:
        size = (np.array(size, dtype=np.int) // ratio).astype(int)
        offset = (np.array(offset, dtype=np.int) // ratio).astype(int)
        boundary = (np.array(self.boundary, dtype=np.int) // ratio).astype(int)
    orig_size = np.copy(size)

    mirror_overlap = [[0, 0], [0, 0], [0, 0]]

    for dim in range(3):
        if offset[dim] < 0:
            size[dim] += offset[dim]
            mirror_overlap[dim][0] = -offset[dim]
            offset[dim] = 0

        if offset[dim] + size[dim] > boundary[dim]:
            mirror_overlap[dim][1] = offset[dim] + size[dim] - boundary[dim]
            size[dim] = boundary[dim] - offset[dim]

        if size[dim] < 0:
            raise Exception("Given block is totally out ouf bounds with "
                            "offset: [%d, %d, %d]!" %
                            (offset[0], offset[1], offset[2]))

    start = self.get_first_blocks(offset).astype(int)
    end = self.get_last_blocks(offset, size).astype(int)

    output = np.zeros(size[::-1], dtype=datatype)

    offset_start = offset % self.cube_shape
    offset_end = (self.cube_shape - (offset + size)
                    % self.cube_shape) % self.cube_shape

    nb_cubes_to_process = int(np.prod(end - start))
    if nb_cubes_to_process == 0:
        return np.zeros(orig_size[::-1], dtype=datatype)

    cube_coordinates = []
    zipped_read_params = []

    for z in range(start[2], end[2]):
        for y in range(start[1], end[1]):
            for x in range(start[0], end[0]):
                zipped_read_params.append((cube_type , np.array([x, y, z])))

    with ThreadPoolExecutor() as pool:
        results = list(pool.map(read_cube, zipped_read_params)) # convert generator to list so we can count

    if results.count(None) < len(results):
        errors = defaultdict(int)
        for result in results: # None results are no error
            if result is not None and result.response is not None: # errors with server response
                errors[result.response.status_code] += 1
            elif result is not None: # errors without server response
                errors[result.__class__.__name__] += 1
        print(f'{len(errors)} non-ok http responses: {list(errors.items())}')

    if self.show_progress:
        dt = time.time() - t0
        speed = np.product(output.shape) * 1.0/1000000/dt
        print(f'\rSpeed: {speed:.2f} Mvx/s, time {dt}')

    if not np.all(output.shape == size[::-1]):
        raise Exception(f'Incorrect shape! Should be {size[::-1]}; got {output.shape}')

    if np.any(mirror_overlap):
        if isinstance(padding, int):
            output = np.pad(output, mirror_overlap[::-1], 'constant', constant_values=padding)
        else:
            output = np.pad(output, mirror_overlap[::-1], mode=padding)

    return output
