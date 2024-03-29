import os
import logging
import zipfile
import sys
import time
import pytest
import tempfile

import numpy as np
from multiprocessing import Process, Queue
from syconn import global_params
# TODO: test VoxelStorageDyn
from syconn.backend.storage import AttributeDict, CompressedStorage, VoxelStorageL, MeshStorage, \
    VoxelStorageClass, BinarySearchStore, VoxelStorageLazyLoading
from syconn.handler.basics import write_txt2kzip, write_data2kzip,\
     read_txt_from_zip, remove_from_zip

# TODO: use tempfile
dir_path = os.path.dirname(os.path.realpath(__file__))


def _setup_testfile(fname):
    test_p = f"{dir_path}/{fname}.pkl"
    if os.path.isfile(test_p):
        os.remove(test_p)

    if os.path.isfile(f'{dir_path}/.{fname}.pkl.lk'):
        os.remove(f'{dir_path}/.{fname}.pkl.lk')
    return test_p


def test_VoxelStorageLazyLoading():
    test_p = f"{dir_path}/vx_dc_lazy.npz"
    if os.path.isfile(test_p):
        os.remove(test_p)
    arr = np.arange(90).reshape((30, 3))
    vx_dc_lazy = VoxelStorageLazyLoading(test_p)
    assert len(vx_dc_lazy) == 0
    vx_dc_lazy[10] = arr
    vx_dc_lazy.push()
    del vx_dc_lazy
    vx_dc_lazy = VoxelStorageLazyLoading(test_p)
    assert 10 in vx_dc_lazy
    np.array_equal(vx_dc_lazy[10], arr)
    assert len(vx_dc_lazy) == 1
    os.remove(test_p)


def test_BinarySearchStore():
    np.random.seed(0)
    n_shards = 5
    n_elements = int(1e6)
    max_int = int(2e6)  # choice becomes slow for large max values
    max_int_attr = int(1e12)
    ids = np.random.choice(max_int, n_elements, replace=False).astype(np.uint64)
    attr = dict(ssv_ids=np.random.randint(1, max_int_attr, n_elements).astype(np.uint64))
    tf = tempfile.TemporaryFile()
    bss = BinarySearchStore(tf, ids, attr, n_shards=n_shards)
    ixs_sample = np.random.permutation(len(ids))[:1000]
    attrs = bss.get_attributes(ids[ixs_sample], 'ssv_ids')
    if not np.array_equal(attr['ssv_ids'][ixs_sample], attrs):
        not_equal = attr['ssv_ids'][ixs_sample] != attrs
        print(attrs[not_equal], attr['ssv_ids'][ixs_sample][not_equal])
    assert np.array_equal(attr['ssv_ids'][ixs_sample], attrs)
    assert bss.n_shards == n_shards, "Number of shards differ."
    assert len(bss.id_array) == len(ids), "Unequal ID array lengths."
    assert np.max(ids) == bss.id_array[-1], 'Maxima do not match.'  # captured by test below, but important detail
    assert np.array_equal(bss.id_array, np.sort(ids)), "Sort failed."
    del bss
    tf.close()


def get_attr_newinstances(args):
    tf, samples, key = args
    binstore = BinarySearchStore(tf)
    return binstore.get_attributes(samples, key)


def get_attr(args):
    bss, samples, key = args
    return bss.get_attributes(samples, key)


def test_BinarySearchStore_multiprocessed():
    np.random.seed(0)
    n_shards = 5
    n_elements = int(2e6)
    n_queries = int(1e3)  # single process query becomes slow for >=1e5
    max_int = int(9e6)  # choice becomes slow for large max values
    max_int_attr = int(1e12)
    ids = np.random.choice(max_int, n_elements, replace=False).astype(np.uint64)
    attr = dict(ssv_ids=np.random.randint(1, max_int_attr, n_elements).astype(np.uint64))
    test_p = f"{dir_path}/.binstore"
    start = time.time()
    bss = BinarySearchStore(test_p, ids, attr, n_shards=n_shards, overwrite=True)
    print(f'build BSS: {(time.time() - start):.2f} s')
    ixs_sample = np.random.permutation(len(ids))[:n_queries]
    ids_samples = ids[ixs_sample]

    start = time.time()
    attrs = bss.get_attributes(ids_samples, 'ssv_ids')
    print(f'get_attr_orig: {(time.time() - start):.2f}')

    from syconn.mp.mp_utils import start_multiprocess
    start = time.time()
    attrs_multi = start_multiprocess(
        get_attr_newinstances, [(test_p, ch, 'ssv_ids') for ch in np.array_split(ids_samples, 5)], nb_cpus=5,
        debug=True)
    print(f'get_attr_newinstances: {(time.time() - start):.2f}')
    attrs_multi = np.concatenate(attrs_multi)
    assert np.array_equal(attrs_multi, attrs)

    start = time.time()
    attrs_multi = start_multiprocess(
        get_attr, [(bss, ch, 'ssv_ids') for ch in np.array_split(ids_samples, 5)], nb_cpus=5,
        debug=False)
    print(f'get_attr_pickle: {(time.time() - start):.2f} s')
    attrs_multi = np.concatenate(attrs_multi)
    assert np.array_equal(attrs_multi, attrs)

    del bss
    os.remove(test_p)


# TODO: requires revision
@pytest.mark.xfail(strict=False)
def test_created_then_blocking_LZ4Dict_for_3s_2_fail_then_one_successful():
    """
      Creates a file then blocks it for 3 seconds. In parallel it creates 3 files.
      First one after 1s , 2nd after 2 seconds and the third one after 3s.
      The first two creations are EXPECTED to fail. The last one is expected to be
      successful.

      Returns:
          str:
              An assertion error in case any of the test case fails.
              Logged to: logs/unitests.log
      """
    test_p = _setup_testfile('test1')

    def create_LZ4Dict_wait_for_3s_then_close():
        # created and locked LZ4Dict for 3s
        pkl1 = CompressedStorage(test_p, read_only=False, disable_locking=False)
        pkl1[1] = np.ones((5, 5))
        time.sleep(1.5)
        pkl1.push()

    def create_fail_expected_runtime_error1(q1):
        try:
            _ = CompressedStorage(test_p, read_only=True, timeout=0.5, disable_locking=False,
                                  max_delay=1)  # timeout sets the maximum time before failing, not max_delay

            logging.warning('FAILED: create_fail_expected_runtime_error1')
            q1.put('FAILED: create_fail_expected_runtime_error1')
        except RuntimeError as e:
            logging.info('PASSED: create_fail_expected_runtime_error1')
            q1.put(0)

    def create_fail_expected_runtime_error2(q2):
        try:
            _ = CompressedStorage(test_p, read_only=True,  disable_locking=False, timeout=0.75)
            logging.warning('FAILED: create_fail_expected_runtime_error2')
            q2.put('FAILED: create_fail_expected_runtime_error2')
        except RuntimeError as e:
            logging.info('PASSED: create_fail_expected_runtime_error2')
            q2.put(0)

    def create_success_expected(q3):
        try:
            _ = CompressedStorage(test_p, read_only=True,  disable_locking=False, timeout=1)
            logging.info('PASSED: create_success_expected')
            q3.put(0)
        except RuntimeError as e:
            logging.warning('FAILED: create_success_expected')
            q3.put('FAILED: create_success_expected')

    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    p = Process(target=create_LZ4Dict_wait_for_3s_then_close)
    p.start()
    time.sleep(0.05)
    p2 = Process(target=create_fail_expected_runtime_error1, args=(q1,))
    p2.start()
    p3 = Process(target=create_fail_expected_runtime_error2, args=(q2,))
    p3.start()
    p.join()  # wait for maximum timeout
    time.sleep(0.05)
    p4 = Process(target=create_success_expected, args=(q3,))
    p4.start()
    p4.join()
    r1, r2, r3 = q1.get(), q2.get(), q3.get()
    if r1 != 0 or r2 != 0 or r3 != 0:
        raise AssertionError(f'{r1}\n{r2}\n{r3}')


def test_saving_loading_and_copying_process_for_Attribute_dict():
    """
    Checks the saving,loading and copying  functionality for an attribute dict
    
    Returns:
        An Assertion Error in case an exception is thrown

    """
    test_p = _setup_testfile('test2')

    try:
        ad = AttributeDict(test_p, read_only=False)
        ad[1]["b"] = 2
        assert "b" in ad[1]
        for i in range(100):
            ad[i] = {"glia_probability": np.ones((10, 2)).astype(np.uint8)}
        start = time.time()
        ad.push()
        logging.debug("Saving AttributeDict took %0.4f." % (time.time() - start))
        logging.debug("AttributeDict file size:\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del ad
        logging.info('PASSED: test_saving_process_for_Attribute_dict')
    except Exception as e:
        logging.warning('FAILED: test_saving_process_for_Attribute_dict. ' + str(e))
        raise AssertionError
    try:
        start = time.time()
        ad = AttributeDict(test_p, read_only=True)
        logging.debug("Loading AttributeDict took %0.4f." % (time.time() - start))
        assert len(list(ad.keys())) == 100
        assert np.all(ad[0]["glia_probability"] == np.ones((10, 2)).astype(np.uint8))
        ad.update({100: "b"})
        assert 100 in ad
        start = time.time()
        dc_constr = ad.copy_intern()
        logging.debug("Copying dict from AttributeDict took %0.4f." % (time.time() - start))
        assert len(list(dc_constr.keys())) == 101
        del ad
        os.remove(test_p)
        logging.info('PASSED: test_loading_and_copying_process_for_Attribute_dict')
    except Exception as e:
        logging.warning('FAILED: test_loading_and_copying_process_for_Attribute_dict. ' + str(e))
        raise AssertionError


def test_compression_and_decompression_for_mesh_dict():
    test_p = _setup_testfile('test3')

    try:
        md = MeshStorage(test_p, read_only=False, disable_locking=False)
        md[1] = [np.ones(100).astype(np.uint32), np.zeros(200).astype(np.float32),
                 np.zeros(200).astype(np.float32), np.zeros((200)).astype(np.uint8)]

        logging.debug("MeshDict arr size (zeros, uncompr.):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in md[1]]) / 1.e3))
        logging.debug("MeshDict arr size (zeros, uncompr.):\t%s" % ([a.shape for a in md[1]]))

        md.push()

    except Exception as e:
        logging.debug('FAILED: test_compression_and_decompression_for_mesh_dict: STEP 1 ' + str(e))
        raise AssertionError

    # checks if lock release after saving works by saving a second time without acquiring lock
    try:
        md.push()
        logging.debug('FAILED: test_compression_and_decompression_for_mesh_dict: STEP 2 ')
        raise AssertionError
    except Exception as e:
        # obey typo in upstream package
        assert str(e) in ["Unable to release an unacquired lock", "Unable to release an unaquired lock"]

    logging.debug("MeshDict file size:\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))

    # checks mesh dict compression with highest entropy data
    md = MeshStorage(test_p, read_only=False)
    mesh = [np.random.randint(0, 100, 1000).astype(np.uint32), np.random.rand(2000, ).astype(np.float32)]
    md[1] = mesh
    try:
        md.update({0: 1})
        assert 0 in md
    except NotImplementedError:
        pass
    logging.debug(
        "MeshDict arr size (random, uncompr.):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in md[1]]) / 1.e3))
    logging.debug("MeshDict arr size (random, uncompr.):\t%s" % (([a.shape for a in md[1]])))

    md.push()
    logging.debug("MeshDict file size:\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del md

    # checks decompression
    try:
        md = MeshStorage(test_p, read_only=True)
        assert np.allclose(md[1][0], mesh[0])
        assert np.allclose(md[1][1], mesh[1])
        os.remove(test_p)
        logging.info('PASSED: test_compression_and_decompression_for_mesh_dict')
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_mesh_dict: STEP 3 ' + str(e))
        raise AssertionError


def test_compression_and_decompression_for_voxel_storage():
    test_p = _setup_testfile('test4')

    try:
        # tests least entropy data
        start = time.time()
        vd = VoxelStorageClass(test_p, read_only=False, cache_decomp=True)
        voxel_masks = [np.zeros((128, 128, 100)).astype(np.uint8),
                       np.zeros((10, 50, 20)).astype(np.uint8)] * 2
        offsets = np.random.randint(0, 1000, (4, 3))
        logging.debug("VoxelDict arr size (zeros):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
        logging.debug("VoxelDict arr size (zeros):\t%s" % ([a.shape for a in voxel_masks]))
        start_comp = time.time()
        vd[8192734] = [voxel_masks, offsets]
        vd.push()
        logging.debug("VoxelDict compression and file writing took %0.4fs." % (time.time() - start_comp))
        logging.debug("VoxelDict file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_voxel_dict: STEP 1 ' + str(e))
        raise AssertionError

    # tests reading
    try:
        start_loading = time.time()
        vd = VoxelStorageClass(test_p, read_only=True, cache_decomp=True)
        logging.debug("Finished loading of compressed VoxelDict after %0.4fs." % (time.time() - start_loading))
        start = time.time()
        _ = vd[8192734]
        logging.debug("Finished decompression of VoxelDict after %0.4fs." % (time.time() - start))

        logging.info("\nAccurarcy tests...")
        for i in range(len(offsets)):
            assert np.allclose(vd[8192734][0][i], voxel_masks[i])
        assert np.allclose(vd[8192734][1], offsets)
        logging.info("... passed.")
        voxel_masks = [np.zeros((128, 128, 100)).astype(np.uint8),
                       np.zeros((10, 50, 20)).astype(np.uint8)] * 2
        offsets = np.random.randint(1000, 2000, (4, 3))
        start_appending = time.time()
        for i_voxel_mask in range(len(voxel_masks)):
            vd.append(8192734, voxel_masks[i_voxel_mask], offsets[i_voxel_mask])
        vd.push()
        logging.debug("VoxelDict appending, compression and file writing took %0.4fs." % (time.time() - start_comp))
        logging.debug("VoxelDict file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
        os.remove(test_p)
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_voxel_dict: STEP 2 ' + str(e))
        raise AssertionError

    # checks high entropy data
    try:
        vd = VoxelStorageClass(test_p, read_only=False)
        voxel_masks = [np.random.randint(0, 1, (128, 128, 100)).astype(np.uint8),
                       np.random.randint(0, 1, (10, 50, 20)).astype(np.uint8)] * 2
        offsets = np.random.randint(0, 1000, (4, 3))
        logging.debug("\nVoxelDict arr size (random):\t%0.2f kB" %
                     (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
        logging.debug("VoxelDict arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
        vd[8192734] = [voxel_masks, offsets]
        vd.push()
        logging.debug("VoxelDict file size (random):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
        # tests decompressing
        vd = VoxelStorageClass(test_p, read_only=True, cache_decomp=True)
        start = time.time()
        _ = vd[8192734]
        logging.debug("Finished decompression of VoxelDict after %0.4fs." % (time.time() - start))
        # accuracy tests
        logging.info("Accurarcy tests...")
        for i in range(len(offsets)):
            assert np.allclose(vd[8192734][0][i], voxel_masks[i])
        assert np.allclose(vd[8192734][1], offsets)
        logging.info("... passed.")
        del vd
        os.remove(test_p)
        logging.info('PASSES: test_compression_and_decompression_for_voxel_dict')
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_voxel_dict: STEP 3 ' + str(e))
        raise AssertionError


def test_compression_and_decompression_for_voxel_dictL():
    test_p = _setup_testfile('test5')

    # tests least entropy data
    try:
        start = time.time()
        vd = VoxelStorageL(test_p, read_only=False, cache_decomp=True)
        voxel_masks = [np.zeros((128, 128, 100)).astype(np.uint8),
                       np.zeros((10, 50, 20)).astype(np.uint8)] * 2
        offsets = np.random.randint(0, 1000, (4, 3))
        logging.debug("VoxelDictL arr size (zeros):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
        logging.debug("VoxelDictL arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
        start_comp = time.time()
        vd[8192734] = [voxel_masks, offsets]
        vd.push()
        logging.debug("VoxelDictL compression and file writing took %0.4fs." % (time.time() - start_comp))
        logging.debug("VoxelDictL file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_voxel_dictL: STEP 1 ' + str(e))
        raise AssertionError

    # tests decompressing
    try:
        start_loading = time.time()
        vd = VoxelStorageL(test_p, read_only=True, cache_decomp=True)
        logging.debug("Finished loading of compressed VoxelDictL after %0.4fs." % (time.time() - start_loading))
        start = time.time()
        _ = vd[8192734]
        logging.debug("Finished decompression of VoxelDictL after %0.4fs." % (time.time() - start))
        logging.info("\nAccurarcy tests...")
        for i in range(len(offsets)):
            assert np.allclose(vd[8192734][0][i], voxel_masks[i])
        assert np.allclose(vd[8192734][1], offsets)
        logging.info("... passed.")
        voxel_masks = [np.zeros((128, 128, 100)).astype(np.uint8),
                       np.zeros((10, 50, 20)).astype(np.uint8)] * 2
        offsets = np.random.randint(1000, 2000, (4, 3))
        start_appending = time.time()
        for i_voxel_mask in range(len(voxel_masks)):
            vd.append(8192734, voxel_masks[i_voxel_mask], offsets[i_voxel_mask])
        vd.push()
        logging.debug("VoxelDictL appending, compression and file writing took %0.4fs." % (time.time() - start_comp))
        logging.debug("VoxelDictL file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
        os.remove(test_p)
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_voxel_dictL: STEP 2 ' + str(e))
        raise AssertionError

    # checks high entropy data
    try:
        vd = VoxelStorageL(test_p, read_only=False)
        voxel_masks = [np.random.randint(0, 1, (128, 128, 100)).astype(np.uint8),
                       np.random.randint(0, 1, (10, 50, 20)).astype(np.uint8)] * 2
        offsets = np.random.randint(0, 1000, (4, 3))
        logging.debug("\nVoxelDictL arr size (random):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
        logging.debug("VoxelDictL arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
        vd[8192734] = [voxel_masks, offsets]
        vd.push()
        logging.debug("VoxelDict file size (random):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
        # tests decompressing
        vd = VoxelStorageL(test_p, read_only=True, cache_decomp=True)
        start = time.time()
        _ = vd[8192734]
        logging.debug("Finished decompression of VoxelDictL after %0.4fs." % (time.time() - start))
        # accuracy tests
        logging.info("Accurarcy tests...")
        for i in range(len(offsets)):
            assert np.allclose(vd[8192734][0][i], voxel_masks[i])
        assert np.allclose(vd[8192734][1], offsets)
        del vd
        os.remove(test_p)
        logging.info('PASSED: test_compression_and_decompression_for_voxel_dictL')
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_voxel_dictL: STEP 3 ' + str(e))
        raise AssertionError


def test_basics_write_txt2kzip():

    """
    Checks the write_txt2kzip function in syconnfs.handler.basics

    Returns:
        An Assertion Error in case an exception is thrown

    """

    try:
        txt = 'test'
        write_txt2kzip(dir_path + '/test6.k.zip', txt, "test")

        if os.path.isfile(dir_path + '/test6.k.zip'):
            os.remove(dir_path + '/test6.k.zip')
        if os.path.isfile(dir_path + '/test6.txt'):
            os.remove(dir_path + '/test6.txt')
        logging.info('PASSED: test_basics_write_txt2kzip')
    except Exception as e:
        logging.warning('FAILED: test_basics_write_txt2kzip' + str(e))
        raise AssertionError


def test_basics_write_data2kzip():

    """
    Checks the write_data2kzipfunction in syconn.handler.basics

    Returns:
        An Assertion Error in case an exception is thrown

    """

    try:
        test_file = open(dir_path + '/test7.txt', "w+")
        test_file.write('This is line test.')
        write_data2kzip(dir_path + '/test7.k.zip', dir_path + '/test7.txt', fname_in_zip='test')
        logging.info('PASSED: test_basics_write_data2kzip')
    except Exception as e:
        logging.warning('FAILED: test_basics_write_data2kzip' + str(e))
        raise AssertionError
    if os.path.isfile(dir_path + '/test7.k.zip'):
        os.remove(dir_path + '/test7.k.zip')
    if os.path.isfile(dir_path + '/test7.txt'):
        os.remove(dir_path + '/test7.txt')


def test_read_txt_from_zip():
    """
    Tests reading of a text file from zip.

    Returns:
        An Assertion Error in case an exception is thrown

    """
    test_str = "testing_" + sys._getframe().f_code.co_name
    try:
        with zipfile.ZipFile(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip', mode='w') as zf:
            zf.writestr(sys._getframe().f_code.co_name + '.txt', test_str)
            zf.close()
        txt = read_txt_from_zip(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip', sys._getframe().f_code.co_name + '.txt')
        txt = txt.decode('utf-8')
        if not txt == test_str:
            raise ValueError('Invalid text found in test file "{}"'.format(txt))
        logging.info("PASSED:" + sys._getframe().f_code.co_name)
        remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
    except Exception as e:
        remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
        logging.info("FAILED:" + sys._getframe().f_code.co_name + str(e))
        raise e


def test_remove_from_zip():
    """
    Tests removing of a text file from zip.

    Returns:
        An Assertion Error in case an exception is thrown

    """
    try:
        with zipfile.ZipFile(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip', mode='w') as zf:
            zf.writestr(sys._getframe().f_code.co_name + '.txt', "testing_" + sys._getframe().f_code.co_name)
            zf.close()
        remove_from_zip(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip',
                        sys._getframe().f_code.co_name + '.txt')
        with pytest.raises(KeyError):
            with zipfile.ZipFile(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip', mode='r') as zf:
                zf.open(sys._getframe().f_code.co_name + '.txt')
            remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
            pass
        logging.info("PASSED:" + sys._getframe().f_code.co_name)
        remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
    except Exception as e:
        remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
        logging.info("FAILED:" + sys._getframe().f_code.co_name + str(e))
        raise e


def remove_files_after_test(file_name):
    if os.path.isfile(str(dir_path) + '/' + file_name):
        os.remove(str(dir_path) + '/' + file_name)


if __name__ == '__main__':
    test_BinarySearchStore()
    test_BinarySearchStore_multiprocessed()
