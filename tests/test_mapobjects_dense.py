# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert


import numpy as np
from syconn.reps.find_object_properties_C import find_object_propertiesC, \
    map_subcell_extract_propsC


def test_map_subcell_extract_propsC():
    # TODO: actual value tests
    edge_s = 50
    toy = np.random.randint(low=0, size=edge_s**3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint64)
    _ = map_subcell_extract_propsC(toy, toy[None])


def test_find_object_propertiesC():
    # TODO: actual value tests
    edge_s = 50
    toy = np.random.randint(low=0, size=edge_s ** 3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint64)
    _ = find_object_propertiesC(toy)


if __name__ == "__main__":
    edge_s = 50
    toy = np.random.randint(low=0, size=edge_s**3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint64)
    _ = find_object_propertiesC(toy)
    _ = map_subcell_extract_propsC(toy, toy[None])
