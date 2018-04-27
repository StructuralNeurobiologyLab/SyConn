import numpy as np
from syconn.reps.super_segmentation import SuperSegmentationObject
#from scripts.rendering.rendering_example import str2intconverter
import re

#retrieving id from files
#retrieving vertices from the ids of SSOs
#the aim is paint with a unique colour every single vertex of sso


def id2rgb(vertex_id):
    """
        Transforms ID value of single sso vertex into the unique RGD colour.

        Parameters
        ----------
        vertex_id : int

        Returns
        -------
        np.array
            RGB values [1, 3]
        """
    red = vertex_id % 255
    green = (vertex_id/255) % 255
    blue = (vertex_id/255/255) % 255
    colour = np.array([red, green, blue], dtype=np.uint8)
    return colour

def id2rgb_array(id_arr):
    """
        Transforms ID values into the array of RGBs labels based on 'idtorgb'.

        Parameters
        ----------
        id_arr : np.array
            ID values [N, 1]

        Returns
        -------
        np.array
            RGB values.squeezed [N, 3]
        """

    if np.max(id_arr) > 16646654:
        raise ValueError("Overflow in vertex ID array.")
    if id_arr.ndim == 1:
        id_arr = id_arr[:, None]
    elif id_arr.ndim == 2:
        assert id_arr.shape[1] == 1, "ValueError: unsupported shape"
    else:
        raise ValueError("Unsupported shape")
    rgb_arr = np.apply_along_axis(id2rgb, 1, id_arr)
    return rgb_arr.squeeze()


def rgb2id(rgb):
    """
        Transforms unique RGB values into soo vertex ID.

        Parameters
        ----------
        rgb: np.array
            RGB values [1, 3]

        Returns
        -------
        np.array
            ID values [1, 1]
        """
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    vertex_id = red + green*255 + blue*(255**2)
    return np.array([vertex_id], dtype=np.uint32)


def rgb2id_array(rgb_arr):
    """
    Transforms RGB values into IDs based on 'rgb2id'.

    Parameters
    ----------
    rgb_arr : np.array
        RGB values [N, 3]

    Returns
    -------
    np.array
        ID values [N, ]
    """
    if rgb_arr.ndim == 2:
        assert rgb_arr.shape[1] == 3, "ValueError: unsupported shape"
    else:
        raise ValueError("Unsupported shape")
    id_arr = np.apply_along_axis(rgb2id, 1, rgb_arr)
    print(id_arr.dtype, id_arr.shape)
    return id_arr.squeeze()


if __name__ == "__main__":
    # vertex_id = 1000056
    # rgb = (21, 96, 15)
    # print(rgb2id(rgb))
    label_file_folder = "/u/shum/spine_label_files/"
    kzip_path = label_file_folder + "/28985344.001.k.zip"
    # kzip_path = label_file_folder + "/4741011.k.zip"
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    sso = SuperSegmentationObject(sso_id)
    indices, vertices, normals = sso.mesh
    vertex_ids = np.arange(vertices.size, dtype=np.uint32)[:20000, None]
    rgb_arr = id2rgb_array(vertex_ids)
    print("mapped ids to rgb")
    remapped_vertex_ids = rgb2id_array(rgb_arr)
    print("mapped rgb to ids")

# print()
# imsave("/u/shum/rev_map_to_rgb.png",)
# imsave("/u/shum/rev_map_to_id.png", )