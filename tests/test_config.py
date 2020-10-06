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
    logging.info('PASSED: Load and write `Config` class.')


def test_key_value_pairs():
    prior_glia_removal = True
    key_val_pairs_conf = [
        ('glia', {'prior_glia_removal': prior_glia_removal}),
        ('cell_objects', {'asym_label': 10})
    ]

    generate_default_conf(test_dir, scaling=(1, 1, 1),
                          key_value_pairs=key_val_pairs_conf)
    conf = Config(test_dir)
    # test if value was modified and written correctly
    assert conf.entries['cell_objects']['asym_label'] == 10, \
        'Modified value for asymmetric synapses is incorrect.'
    # test if default values were modified or removed
    assert conf.entries['cell_objects']['sym_label'] is None, \
        'Default value for symmetric synapses is incorrect.'
    conf = DynConfig(test_dir)
    assert conf.sym_label is None, \
        'Default value for symmetric synapses is incorrect (DynConf).'
    assert conf.asym_label == 10, \
        'Default value for symmetric synapses is incorrect (DynConf).'
    os.remove(conf.path_config)


def test_key_value_pairs_fail():
    key_val_pairs_conf = [
        ('asym_label', 10)
    ]
    try:
        generate_default_conf(test_dir, scaling=(1, 1, 1),
                              key_value_pairs=key_val_pairs_conf)
        conf = Config(test_dir)
        os.remove(conf.path_config)
        assert 'Key-value pair was added to config which does not exist ' \
               'in the default config.'
    except KeyError:
        pass


if __name__ == '__main__':
    test_key_value_pairs_fail()
