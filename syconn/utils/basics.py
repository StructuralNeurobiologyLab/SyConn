import h5py
import os


def chunkify(lst, n):
    """
    Splits list into n sub-lists.

    Parameters
    ----------
    lst : list
    n : int

    Returns
    -------

    """
    return [lst[i::n] for i in range(n)]


def switch_array_entries(this_array, entries):
    entry_0 = this_array[entries[0]]
    this_array[entries[0]] = this_array[entries[1]]
    this_array[entries[1]] = entry_0
    return this_array


def load_from_h5py(path, hdf5_names=None, as_dict=False):
    """
    Loads data from a h5py File

    Parameters
    ----------
    path: str
    hdf5_names: list of str
        if None, all keys will be loaded
    as_dict: boolean
        if False a list is returned

    Returns
    -------
    data: dict or np.array

    """
    if as_dict:
        data = {}
    else:
        data = []
    try:
        f = h5py.File(path, 'r')
        if hdf5_names is None:
            hdf5_names = f.keys()
        for hdf5_name in hdf5_names:
            if as_dict:
                data[hdf5_name] = f[hdf5_name].value
            else:
                data.append(f[hdf5_name].value)
    except:
        raise Exception("Error at Path: %s, with labels:" % path, hdf5_names)
    f.close()
    return data


def save_to_h5py(data, path, hdf5_names=None):
    """
    Saves data to h5py File

    Parameters
    ----------
    data: list of np.arrays
    path: str
    hdf5_names: list of str
        has to be the same length as data

    Returns
    -------
    nothing

    """
    if (not type(data) is dict) and hdf5_names is None:
        raise Exception("hdf5names has to be set, when data is a list")
    if os.path.isfile(path):
        os.remove(path)
    f = h5py.File(path, "w")
    if type(data) is dict:
        for key in data.keys():
            f.create_dataset(key, data=data[key],
                             compression="gzip")
    else:
        if len(hdf5_names) != len(data):
            f.close()
            raise Exception("Not enough or to much hdf5-names given!")
        for nb_data in range(len(data)):
            f.create_dataset(hdf5_names[nb_data], data=data[nb_data],
                             compression="gzip")
    f.close()


def cut_array_in_one_dim(array, start, end, dim):
    """
    Cuts an array along a dimension

    Parameters
    ----------
    array: np.array
    start: int
    end: int
    dim: int

    Returns
    -------
    array: np.array

    """
    start = int(start)
    end = int(end)
    if dim == 0:
        array = array[start: end, :, :]
    elif dim == 1:
        array = array[:, start: end, :]
    elif dim == 2:
        array = array[:, :, start:end]
    else:
        raise NotImplementedError()

    return array