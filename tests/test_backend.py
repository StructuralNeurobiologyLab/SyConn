import pytest
import numpy as np
import time
from multiprocessing import Process, Queue
from syconn.backend.storage import AttributeDict, CompressedStorage, VoxelStorageL, MeshStorage, \
    VoxelStorage
from syconn.handler.basics import write_txt2kzip,write_data2kzip,\
    read_txt_from_zip, remove_from_zip
import os
import logging
import traceback
import zipfile
import sys


dir_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=dir_path + '/unit_tests.log',
                    level=logging.DEBUG, filemode='w')

test_p = dir_path + "/test.pkl"
if os.path.isfile(test_p):
    os.remove(test_p)


def test_created_then_blocking_LZ4Dict_for_3s_2_fail_then_one_successful():

    """
      Creates a file then blocks it for 3 seconds. In parallel it creates 3 files.
      First one after 1s , 2nd after 2 seconds and the third one after 3s.
      The first two creations are EXPECTED to fail. The last one is expected to be
      successful.

      Parameters
      ----------
      None

      Returns
      -------
      str
          An assertion error in case any of the test case fails.
          Logged to: logs/unitests.log
      """

    def create_LZ4Dict_wait_for_3s_then_close():

        #created and locked LZ4Dict for 3s
        pkl1 = CompressedStorage(test_p, read_only=False)
        pkl1[1] = np.ones((5, 5))
        time.sleep(3)
        pkl1.push()

    def create_fail_expected_runtime_error_at_1s(a, b, q1):
        # logging.debug("Started worker to access file for 1s"
        time.sleep(0)
        start = time.time()
        try:
            pkl2 = CompressedStorage(test_p, read_only=True, timeout=1,
                                     max_delay=1)  # timeout sets the maximum time before failing, not max_delay

            logging.warning('FAILED: create_fail_expected_runtime_error_at_1s')
            q1.put(1)
        except RuntimeError as e:
            logging.info('PASSED: create_fail_expected_runtime_error_at_1s')
            q1.put(0)

    def create_fail_expected_runtime_error_at_2s(a, b, q2):
        # logging.debug("Started worker to access file for 2s."
        time.sleep(0)
        start = time.time()
        try:
            pkl2 = CompressedStorage(test_p, read_only=True, timeout=2)
            logging.warning('FAILED: create_fail_expected_runtime_error_at_2s')
            q2.put(1)
        except RuntimeError as e:
            logging.info('PASSED: create_fail_expected_runtime_error_at_2s')
            q2.put(0)

    def create_success_expected(a, b, q3):
        time.sleep(0)
        start = time.time()

        try:
            pkl2 = CompressedStorage(test_p, read_only=True, timeout=1)
            logging.info('PASSED: create_success_expected')
            q3.put(0)
        except RuntimeError as e:
            logging.warning('FAILED: create_success_expected')
            q3.put(1)

    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    p = Process(target=create_LZ4Dict_wait_for_3s_then_close)
    p.start()
    time.sleep(0.01)
    p2 = Process(target=create_fail_expected_runtime_error_at_1s, args=(1, 2, q1))
    p2.start()
    p3 = Process(target=create_fail_expected_runtime_error_at_2s, args=(1, 2, q2))
    p3.start()
    p.join()
    p4 = Process(target=create_success_expected, args=(1, 2, q3))
    p4.start()
    p4.join()
    if q1.get() == 1 or q2.get() == 1 or q3.get() == 1:
       raise AssertionError


def test_saving_loading_and_copying_process_for_Attribute_dict():

    """
    Checks the saving,loading and copying  functionality for an attribute dict
    Returns An Assertion Error in case an exception is thrown
    -------

    """

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

    try:
        md = MeshStorage(test_p, read_only=False)
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
        logging.debug('FAILED: test_compression_and_decompression_for_mesh_dict: STEP 2 ' + str(e))
    except Exception as e:
        assert str(e) ==  "Unable to release an unacquired lock"
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


def test_compression_and_decompression_for_voxel_dict():

    try:
        # tests least entropy data
        start = time.time()
        vd = VoxelStorage(test_p, read_only=False, cache_decomp=True)
        voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                       np.zeros((300, 400, 200)).astype(np.uint8)] * 5
        offsets = np.random.randint(0, 1000, (10, 3))
        logging.debug("VoxelDict arr size (zeros):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
        logging.debug("VoxelDict arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
        start_comp = time.time()
        vd[8192734] = [voxel_masks, offsets]
        vd.push()
        logging.debug("VoxelDict compression and file writing took %0.4fs." % (time.time() - start_comp))
        logging.debug("VoxelDict file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
    except Exception as e:
        logging.warning('FAILED: test_compression_and_decompression_for_voxel_dict: STEP 1 ' + str(e))
        raise AssertionError

    # tests decompressing
    try:
        start_loading = time.time()
        vd = VoxelStorage(test_p, read_only=True, cache_decomp=True)
        logging.debug("Finished loading of compressed VoxelDict after %0.4fs." % (time.time() - start_loading))
        start = time.time()
        _ = vd[8192734]
        logging.debug("Finished decompression of VoxelDict after %0.4fs." % (time.time() - start))

        logging.info("\nAccurarcy tests...")
        for i in range(len(offsets)):
            assert np.allclose(vd[8192734][0][i], voxel_masks[i])
        assert np.allclose(vd[8192734][1], offsets)
        logging.info("... passed.")
        voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                       np.zeros((300, 400, 200)).astype(np.uint8)] * 5
        offsets = np.random.randint(1000, 2000, (10, 3))
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
        vd = VoxelStorage(test_p, read_only=False)
        voxel_masks = [np.random.randint(0, 1, (512, 512, 256)).astype(np.uint8),
                       np.random.randint(0, 1, (300, 400, 200)).astype(np.uint8)] * 5
        offsets = np.random.randint(0, 1000, (10, 3))
        logging.debug("\nVoxelDict arr size (random):\t%0.2f kB" %
                     (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
        logging.debug("VoxelDict arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
        vd[8192734] = [voxel_masks, offsets]
        vd.push()
        logging.debug("VoxelDict file size (random):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
        del vd
        # tests decompressing
        vd = VoxelStorage(test_p, read_only=True, cache_decomp=True)
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

    # tests least entropy data
    try:
        start = time.time()
        vd = VoxelStorageL(test_p, read_only=False, cache_decomp=True)
        voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                       np.zeros((300, 400, 200)).astype(np.uint8)] * 5
        offsets = np.random.randint(0, 1000, (10, 3))
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
        voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                       np.zeros((300, 400, 200)).astype(np.uint8)] * 5
        offsets = np.random.randint(1000, 2000, (10, 3))
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
        voxel_masks = [np.random.randint(0, 1, (512, 512, 256)).astype(np.uint8),
                       np.random.randint(0, 1, (300, 400, 200)).astype(np.uint8)] * 5
        offsets = np.random.randint(0, 1000, (10, 3))
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
    Returns an assertion error if an exception is thrown
    -------

    """

    try:
        txt = 'test'
        write_txt2kzip(dir_path + '/test.kzip',txt, "test")

        if os.path.isfile(dir_path + '/test.kzip'):
            os.remove(dir_path + '/test.kzip')
        if os.path.isfile(dir_path + '/test.txt'):
            os.remove(dir_path + '/test.txt')
        logging.info('PASSED: test_basics_write_txt2kzip')
    except Exception as e:
        logging.warning('FAILED: test_basics_write_txt2kzip' + str(e))
        raise AssertionError


def test_basics_write_data2kzip():

    """
    Checks the write_data2kzipfunction in syconn.handler.basics
    Returns an assertion error if an exception is thrown
    -------

    """

    try:
        test_file = open(dir_path + '/test.txt', "w+")
        test_file.write('This is line test.')
        write_data2kzip(dir_path + '/test.kzip', dir_path + '/test.txt', fname_in_zip='test')
        logging.info('PASSED: test_basics_write_data2kzip')
    except Exception as e:
        logging.warning('FAILED: test_basics_write_data2kzip' + str(e))
        raise AssertionError
    if os.path.isfile(dir_path + '/test.kzip'):
        os.remove(dir_path + '/test.kzip')
    if os.path.isfile(dir_path + '/test.txt'):
        os.remove(dir_path + '/test.txt')


def test_read_txt_from_zip():
    """
    Tests reading of a text file from zip.
    Returns an assertion error if an exception is thrown
    -------

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
        logging.info("FAILED:" + sys._getframe().f_code.co_name)
        raise e


def test_remove_from_zip():
    """
    Tests removing of a text file from zip.
    Returns an assertion error if an exception is thrown
    -------

    """
    try:
        with zipfile.ZipFile(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip', mode = 'w') as zf:
            zf.writestr(sys._getframe().f_code.co_name + '.txt', "testing_" + sys._getframe().f_code.co_name)
            zf.close()
        remove_from_zip(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip',sys._getframe().f_code.co_name + '.txt' )
        with pytest.raises(KeyError, message = " Expecting Key Error as the file should have been removed"):
            with zipfile.ZipFile(str(dir_path) + '/' + sys._getframe().f_code.co_name + '.zip', mode='r') as zf:
                zf.open(sys._getframe().f_code.co_name + '.txt')
            remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
            pass
        logging.info("PASSED:" + sys._getframe().f_code.co_name)
        remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
    except Exception as e:
        remove_files_after_test(sys._getframe().f_code.co_name + '.zip')
        logging.info("FAILED:" + sys._getframe().f_code.co_name)
        raise e


def remove_files_after_test(file_name):
    if os.path.isfile(str(dir_path) + '/' + file_name):
        os.remove(str(dir_path) + '/' + file_name)


