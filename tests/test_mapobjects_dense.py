# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert


import numpy as np
from syconn.extraction.find_object_properties import find_object_properties, \
    map_subcell_extract_props


def test_map_subcell_extract_props():
    # TODO: actual value tests
    edge_s = 50
    toy = np.random.randint(low=0, size=edge_s**3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint64)
    _ = map_subcell_extract_props(toy, toy[None])


def test_find_object_properties():
    # TODO: actual value tests
    edge_s = 50
    toy = np.random.randint(low=0, size=edge_s ** 3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint64)
    _ = find_object_properties(toy)


if __name__ == "__main__":
    edge_s = 50
    toy = np.random.randint(low=0, size=edge_s**3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint64)
    _ = find_object_properties(toy)
    _ = map_subcell_extract_props(toy, toy[None])
