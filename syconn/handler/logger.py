# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import logging
import coloredlogs
import os

from .. import global_params

__all__ = ['initialize_logging']


def get_main_log():
    logger = logging.getLogger('syconn')
    coloredlogs.install(level=global_params.log_level, logger=logger)
    level = logging.getLevelName(global_params.log_level)
    logger.setLevel(level)

    if not global_params.DISABLE_FILE_LOGGING:
        # create file handler which logs even debug messages
        log_dir = os.path.expanduser('~') + "/SyConn/logs/"

        try:
            os.makedirs(log_dir, exist_ok=True)
        except TypeError:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
        fh = logging.FileHandler(log_dir + 'syconn.log')
        fh.setLevel(level)

        # add the handlers to logger
        if os.path.isfile(log_dir + 'syconn.log'):
            os.remove(log_dir + 'syconn.log')
        logger.addHandler(fh)
        logger.info("Initialized file logging. Log-files are stored at"
                    " {}.".format(log_dir))
    logger.info("Initialized stdout logging (level: {}).".format(global_params.log_level))
    return logger


def initialize_logging(log_name, log_dir=global_params.default_log_dir,
                       overwrite=True):
    """
    Logger for each package module. For import processing steps individual
    logger can be defined (e.g. multiviews, skeleton)
    Parameters
    ----------
    log_name : str
        Name of logger
    log_dir : str
        Set log_dir specifically. Will then create a filehandler and ignore the
         state of global_params.DISABLE_FILE_LOGGING state.
    overwrite : bool
        Previous log file will be overwritten

    Returns
    -------

    """
    logger = logging.getLogger(log_name)
    coloredlogs.install(level=global_params.log_level, logger=logger,
                        reconfigure=False)  # True possibly leads to stderr output
    level = logging.getLevelName(global_params.log_level)
    logger.setLevel(level)

    if not global_params.DISABLE_FILE_LOGGING or log_dir is not None:
        # create file handler which logs even debug messages
        if log_dir is None:
            log_dir = os.path.expanduser('~') + "/.SyConn/logs/"
        try:
            os.makedirs(log_dir, exist_ok=True)
        except TypeError:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
        if overwrite and os.path.isfile(log_dir + log_name + '.log'):
            os.remove(log_dir + log_name + '.log')
        # add the handlers to logger
        fh = logging.FileHandler(log_dir + log_name + ".log")
        fh.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# init main logger
log_main = get_main_log()

# TODO: might be interesting to redirect output of syconn modules
# (proc, handler, ...) more dynamically.