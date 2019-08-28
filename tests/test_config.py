# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

import pytest
from syconn.handler.config import DynConfig, generate_default_conf, Config
import os
import logging

test_dir = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=test_dir + '/test_config.log',
                    level=logging.DEBUG, filemode='w')


def test_load_write_conf():
    generate_default_conf(test_dir, scaling=(1, 1, 1), force_overwrite=True)
    assert os.path.isfile(test_dir + '/config.yml'), "Error creating config file."
    with pytest.raises(ValueError):
        generate_default_conf(test_dir, scaling=(1, 1, 1))
    conf = Config(test_dir)
    conf2 = Config(test_dir)
    assert conf == conf2, "Mismatch between config objects."
    entries = conf.entries
    entries['scaling'] = (2, 2, 2)
    assert conf != conf2, "Expected mismatch between config objects but " \
                          "they are the same."
    conf.write_config(test_dir)
    conf2 = Config(test_dir)
    assert conf == conf2, "Mismatch between config objects after re-loading " \
                          "modified config file."
    os.remove(conf.path_config)
    print('PASSED: Load and write `Config` class.')


if __name__ == '__main__':
    test_load_write_conf()
