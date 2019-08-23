# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.handler.config import DynConfig, generate_default_conf, Config
import os
import shutil
import logging

test_dir = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=test_dir + '/test_config.log',
                    level=logging.DEBUG, filemode='w')


def test_load_write_conf():
    generate_default_conf(test_dir, scaling=(1, 1, 1), force_overwrite=True)
    assert os.path.isfile(test_dir + '/config.yml')
    try:
        generate_default_conf(test_dir, scaling=(1, 1, 1))
        raise AssertionError
    except ValueError:
        pass
    conf = Config(test_dir)
    conf2 = Config(test_dir)
    print(conf, conf2)
    assert conf == conf2
    entries = conf.entries
    entries['scaling'] = (2, 2, 2)
    assert conf != conf2
    conf.write_config(test_dir)
    conf2 = Config(test_dir)
    assert conf == conf2
    os.remove(conf.path_config)
