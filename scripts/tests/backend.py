# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import numpy as np
import time
from multiprocessing import Process
from syconn.handler.compression import LZ4Dict, MeshDict, \
    VoxelDict, VoxelDictL, AttributeDict
import os

test_p = "/wholebrain/scratch/areaxfs/test.pkl"
if os.path.isfile(test_p):
    os.remove(test_p)

if 0:
    print("\n---------------------------------------------------\n"
         "Checking locking...\n")

    def create_waite_close():
        print("Created blocking LZ4Dict for 3s.")
        lala = LZ4Dict(test_p, read_only=False) # check if blocking times out after 0.5
        lala[1] = np.ones((5, 5))
        print(lala[1])
        time.sleep(3)
        lala.save2pkl()

    def create_fail():
        time.sleep(0)
        start = time.time()
        print("Started worker to access file for 1s")
        try:
            lala2 = LZ4Dict(test_p, read_only=True, timeout=1, max_delay=1) #timeout sets the maximum time before failing, not max_delay
            print(lala2)
            raise(AssertionError)
        except RuntimeError as e:
            print("\n%s\nStopped loading attempt after %0.1fs.\n" % (str(e), (time.time() - start)))

    def create_fail2():
        time.sleep(0)
        start = time.time()
        print("Started worker to access file for 2s.")
        try:
            lala2 = LZ4Dict(test_p, read_only=True, timeout=2)
            print(lala2)
            raise (AssertionError)
        except RuntimeError as e:
            print("\n%s\nStopped loading attempt after %0.1fs.\n" % (str(e), (time.time() - start)))

    def create_succeed():
        time.sleep(0)
        start = time.time()
        print("Started worker to access file for 2s.")
        try:
            lala2 = LZ4Dict(test_p, read_only=True, timeout=1)
            print(lala2[1])
        except Exception as e:
            print("\n%s\nStopped loading attempt after %0.1fs.\n" % (e, (time.time() - start)))
            raise (AssertionError)

    p = Process(target=create_waite_close)
    p.start()
    time.sleep(0.01)
    p2 = Process(target=create_fail)
    p2.start()
    p3 = Process(target=create_fail2)
    p3.start()
    p.join()

    print("Lock released. Loading should work now.")
    p4 = Process(target=create_succeed)
    p4.start()
    p4.join()
    os.remove(test_p)


if 1:
    print("\n---------------------------------------------------\n"
          "Checking global AttributeDict...\n")
    # checking saving time
    ad = AttributeDict(test_p, read_only=False)
    print(ad[1])
    ad[1]["b"] = 2
    assert "b" in ad[1]
    for i in range(10000):
        ad[i] = {"glia_proba": np.ones((10, 2)).astype(np.uint8)}
    start = time.time()
    ad.save2pkl()
    print("Saving AttributeDict took %0.4f." % (time.time() - start))
    print("AttributeDict file size:\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del ad
    # checking loading time
    start = time.time()
    ad = AttributeDict(test_p, read_only=True)
    print("Loading AttributeDict took %0.4f." % (time.time() - start))
    assert len(ad.keys()) == 10000
    assert np.all(ad[0]["glia_proba"] == np.ones((10, 2)).astype(np.uint8))
    ad.update({10000: "b"})
    assert 10000 in ad
    # checking copy
    start = time.time()
    dc_constr = ad.copy_intern()
    print("Copying dict from AttributeDict took %0.4f." % (time.time() - start))
    assert len(dc_constr.keys()) == 10001
    del ad
    os.remove(test_p)


if 1:
    print("\n---------------------------------------------------\n"
          "Checking compression...")

    print("... for MeshDict")
    # check mesh dict compression with least entropy data
    md = MeshDict(test_p, read_only=False)
    md[1] = [np.ones(10000).astype(np.uint32), np.zeros(20000).astype(np.float32)]
    print("MeshDict arr size (zeros, uncompr.):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in md[1]]) / 1.e3))
    print("MeshDict arr size (zeros, uncompr.):\t%s" % (([a.shape for a in  md[1]])))
    md.save2pkl()

    #  check if lock release after saving works by saving a second time without acquiring lock
    try:
        md.save2pkl()
    except Exception as e:
        assert str(e) == "Unable to release an unacquired lock"
    print("MeshDict file size:\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))

    # check mesh dict compression with highest entropy data
    md = MeshDict(test_p, read_only=False)
    mesh = [np.random.randint(0, 1000, 10000).astype(np.uint32), np.random.rand(20000,).astype(np.float32)]
    md[1] = mesh
    try:
        md.update({0: 1})
        assert 0 in md
    except NotImplementedError:
        pass
    print("MeshDict arr size (random, uncompr.):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in md[1]]) / 1.e3))
    print("MeshDict arr size (random, uncompr.):\t%s" % (([a.shape for a in  md[1]])))
    md.save2pkl()
    print("MeshDict file size:\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del md

    # check decompression
    md = MeshDict(test_p, read_only=True)
    assert np.allclose(md[1][0], mesh[0])
    assert np.allclose(md[1][1], mesh[1])
    os.remove(test_p)

    print("\n... for VoxelDict")
    # check voxel dict funtionality

    # test least entropy data
    start = time.time()
    vd = VoxelDict(test_p, read_only=False, cache_decomp=True)
    voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                   np.zeros((300, 400, 200)).astype(np.uint8)] * 5
    offsets = np.random.randint(0, 1000, (10, 3))
    print("VoxelDict arr size (zeros):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
    print("VoxelDict arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
    start_comp = time.time()
    vd[8192734] = [voxel_masks, offsets]
    vd.save2pkl()
    print("VoxelDict compression and file writing took %0.4fs." % (time.time() - start_comp))
    print("VoxelDict file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del vd
    # test decompressing
    start_loading = time.time()
    vd = VoxelDict(test_p, read_only=True, cache_decomp=True)
    print("Finished loading of compressed VoxelDict after %0.4fs." % (time.time() - start_loading))
    start = time.time()
    _ = vd[8192734]
    print("Finished decompression of VoxelDict after %0.4fs." % (time.time() - start))

    print("\nAccurarcy tests...")
    for i in range(len(offsets)):
        assert np.allclose(vd[8192734][0][i], voxel_masks[i])
    assert np.allclose(vd[8192734][1], offsets)
    print("... passed.")
    voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                   np.zeros((300, 400, 200)).astype(np.uint8)] * 5
    offsets = np.random.randint(1000, 2000, (10, 3))
    start_appending = time.time()
    for i_voxel_mask in range(len(voxel_masks)):
        vd.append(8192734, voxel_masks[i_voxel_mask], offsets[i_voxel_mask])
    vd.save2pkl()
    print("VoxelDict appending, compression and file writing took %0.4fs." % (time.time() - start_comp))
    print("VoxelDict file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del vd
    os.remove(test_p)

    # check high entropy data
    vd = VoxelDict(test_p, read_only=False)
    voxel_masks = [np.random.randint(0, 1, (512, 512, 256)).astype(np.uint8),
                   np.random.randint(0, 1, (300, 400, 200)).astype(np.uint8)] * 5
    offsets = np.random.randint(0, 1000, (10, 3))
    print("\nVoxelDict arr size (random):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
    print("VoxelDict arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
    vd[8192734] = [voxel_masks, offsets]
    vd.save2pkl()
    print("VoxelDict file size (random):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del vd
    # test decompressing
    vd = VoxelDict(test_p, read_only=True, cache_decomp=True)
    start = time.time()
    _ = vd[8192734]
    print("Finished decompression of VoxelDict after %0.4fs." % (time.time() - start))
    # accuracy tests
    print("Accurarcy tests...")
    for i in range(len(offsets)):
        assert np.allclose(vd[8192734][0][i], voxel_masks[i])
    assert np.allclose(vd[8192734][1], offsets)
    print("... passed.")
    del vd
    os.remove(test_p)

    print("\n... for VoxelDictL")
    # check voxel dict funtionality

    # test least entropy data
    start = time.time()
    vd = VoxelDictL(test_p, read_only=False, cache_decomp=True)
    voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                   np.zeros((300, 400, 200)).astype(np.uint8)] * 5
    offsets = np.random.randint(0, 1000, (10, 3))
    print("VoxelDictL arr size (zeros):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
    print("VoxelDictL arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
    start_comp = time.time()
    vd[8192734] = [voxel_masks, offsets]
    vd.save2pkl()
    print("VoxelDictL compression and file writing took %0.4fs." % (time.time() - start_comp))
    print("VoxelDictL file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del vd
    # test decompressing
    start_loading = time.time()
    vd = VoxelDictL(test_p, read_only=True, cache_decomp=True)
    print("Finished loading of compressed VoxelDictL after %0.4fs." % (time.time() - start_loading))
    start = time.time()
    _ = vd[8192734]
    print("Finished decompression of VoxelDictL after %0.4fs." % (time.time() - start))

    print("\nAccurarcy tests...")
    for i in range(len(offsets)):
        assert np.allclose(vd[8192734][0][i], voxel_masks[i])
    assert np.allclose(vd[8192734][1], offsets)
    print("... passed.")
    voxel_masks = [np.zeros((512, 512, 256)).astype(np.uint8),
                   np.zeros((300, 400, 200)).astype(np.uint8)] * 5
    offsets = np.random.randint(1000, 2000, (10, 3))
    start_appending = time.time()
    for i_voxel_mask in range(len(voxel_masks)):
        vd.append(8192734, voxel_masks[i_voxel_mask], offsets[i_voxel_mask])
    vd.save2pkl()
    print("VoxelDictL appending, compression and file writing took %0.4fs." % (time.time() - start_comp))
    print("VoxelDictL file size (zeros):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del vd
    os.remove(test_p)

    # check high entropy data
    vd = VoxelDictL(test_p, read_only=False)
    voxel_masks = [np.random.randint(0, 1, (512, 512, 256)).astype(np.uint8),
                   np.random.randint(0, 1, (300, 400, 200)).astype(np.uint8)] * 5
    offsets = np.random.randint(0, 1000, (10, 3))
    print("\nVoxelDictL arr size (random):\t%0.2f kB" % (np.sum([a.__sizeof__() for a in voxel_masks]) / 1.e3))
    print("VoxelDictL arr size (zeros):\t%s" % (([a.shape for a in voxel_masks])))
    vd[8192734] = [voxel_masks, offsets]
    vd.save2pkl()
    print("VoxelDict file size (random):\t%0.2f kB" % (os.path.getsize(test_p) / 1.e3))
    del vd
    # test decompressing
    vd = VoxelDictL(test_p, read_only=True, cache_decomp=True)
    start = time.time()
    _ = vd[8192734]
    print("Finished decompression of VoxelDictL after %0.4fs." % (time.time() - start))
    # accuracy tests
    print("Accurarcy tests...")
    for i in range(len(offsets)):
        assert np.allclose(vd[8192734][0][i], voxel_masks[i])
    assert np.allclose(vd[8192734][1], offsets)
    print("... passed.")
    del vd
    os.remove(test_p)


