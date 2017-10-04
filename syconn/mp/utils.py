# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

def multi_helper_obj(args):
    """
    Generic helper emthod for multiprocessed jobs. Calls the given object 
    method.
    
    Parameters
    ----------
    args : iterable
        object, method name, optional: kwargs        

    Returns
    -------
    
    """
    attr_str = args[0]
    obj = args[1]
    if len(args) == 3:
        kwargs = args[2]
    else:
        kwargs = {}
    attr = getattr(obj, attr_str)
    # check if attr is callable, i.e. a method to be called
    if not hasattr(attr, '__call__'):
        return attr
    return attr(**kwargs)


def negative_to_zero(a):
    if a > 0:
        return a
    else:
        return 0