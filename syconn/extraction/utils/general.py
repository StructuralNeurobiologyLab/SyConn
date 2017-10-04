import numpy as np
import scipy.ndimage


def single_conn_comp_img(img, background=1.0):
    """
    Returns connected component in image which is located at the center.
    TODO: add 'max component' option

    Parameters
    ----------
    img : np.array
    background : float

    Returns
    -------
    np.array
    """
    orig_shape = img.shape
    img = np.squeeze(img)
    labeled, nr_objects = scipy.ndimage.label(img != background)
    new_img = np.ones_like(img) * background
    center = np.array(img.shape) / 2
    ixs = [labeled == labeled[tuple(center)]]
    new_img[ixs] = img[ixs]
    return new_img.reshape(orig_shape)