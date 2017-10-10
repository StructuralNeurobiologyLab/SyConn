import glob
import os


def find_missing_overlaycubes_thread(args):
    paths = args[0]

    m_paths = []
    for path in paths:
        if len(glob.glob(path + "/*seg.sz.zip")) == 0:
            m_paths.append(path)
    return m_paths
