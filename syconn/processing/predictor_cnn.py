from elektronn.training import predictor
from syconn.processing import initialization
from knossos_utils import knossosdataset
from knossos_utils import chunky
import glob
import h5py
import numpy as np
import os
import re
import sys
import time
import theano


def interpolate(data, mag=2):
    ds = data.shape
    new_data = np.zeros([ds[0], ds[1]*mag, ds[2]*mag, ds[3]*mag])
    for x in range(0, mag):
        for y in range(0, mag):
            for z in range(0, mag):
                new_data[:, x::mag, y::mag, z::mag] = data
    return new_data


def create_chunk_checklist(head_path, nb_chunks, names, subfolder=0):
    checklist = np.zeros((nb_chunks, len(names)), dtype=np.uint8)
    folders_in_path = glob.glob(head_path+"/chunky*")
    for folder in folders_in_path:
        if len(re.findall('[\d]+', folder)) > 0:
            chunk_nb = int(re.findall('[\d]+', folder)[-1])
            if subfolder == 0:
                existing_files = glob.glob(folder+"/*.h5")
            else:
                existing_files = glob.glob(folder+"/*")
            for file in existing_files:
                for name_nb in range(len(names)):
                    if names[name_nb] in file:
                        checklist[chunk_nb, name_nb] = 1

    return checklist


def search_for_chunk(head_path, nb_chunks, name, subfolder=1, max_age_min=100):
    checklist = create_chunk_checklist(head_path, nb_chunks,
                                       [name], subfolder=subfolder)
    left_chunks = []
    for chunk_nb in range(nb_chunks):
        if checklist[chunk_nb] == 0:
            left_chunks.append(chunk_nb)
    if len(left_chunks) > 0:
        for ii in range(4):
            this_max_age_min = max_age_min/(ii+1)
            np.random.shuffle(left_chunks)
            for left_chunk in left_chunks:
                folder_path = head_path + "chunky_%d/mutex_%s" % (left_chunk, name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    return left_chunk
                else:
                    if time.time()-os.stat(folder_path).st_mtime > this_max_age_min*60:
                        os.rmdir(folder_path)
                        os.makedirs(folder_path)
                        return left_chunk
    return -1


def create_recursive_data(labels, labels_data=None, labels_path="",
                          raw_path="", raw_data=None, use_labels=[],
                          data_shape=None):
    try:
        len(labels)
    except:
        raise Exception("labels has to be a list")

    if len(use_labels) == 0:
        use_labels = np.ones(len(labels))
    else:
        use_labels = np.array(use_labels)

    if labels_data is None and len(labels_path) == 0:
        raise Exception("No labels_data or labels_path given")

    labels_dict = {}
    new_labels = []
    if len(labels_path) > 0:
        f = h5py.File(labels_path)
        for nb_label in range(len(labels)):
            if use_labels[nb_label]:
                label = labels[nb_label]
                labels_dict[label] = f[label].value
                new_labels.append(label)
        f.close()
    else:
        for nb_label in range(len(labels)):
            if use_labels[nb_label]:
                label = labels[nb_label]
                labels_dict[label] = labels_data[nb_label]
                new_labels.append(label)

    labels = new_labels

    recursive_data = []
    if len(raw_path) > 1 or not raw_data is None:
        if raw_data is None:
            f = h5py.File(raw_path)
            raw_data = f["raw"].value
            f.close()
            print "Raw data is none"

        labels_shape = np.array(labels_dict[labels[0]].shape)
        raw_shape = np.array(raw_data.shape)
        offset = (raw_shape - labels_shape) / 2
        recursive_data.append(raw_data[offset[0]: raw_shape[0] - offset[0],
                              offset[1]: raw_shape[1] - offset[1],
                              offset[2]: raw_shape[2] - offset[2]])

    for nb_label in range(len(labels)):
        label = labels[nb_label]
        if not label == "none":
            print label
            recursive_data.append(labels_dict[label])

    print len(recursive_data)
    return np.array(recursive_data)


def join_chunky_inference(cset, config_path, param_path, names,
                          labels, offset, desired_input, gpu=None, MFP=True,
                          invert_data=False, kd=None, mag=1):

    sys.setrecursionlimit(10000)

    offset = np.array(offset)

    if kd is None:
        kd = cset.dataset

    nb_chunks = len(cset.chunk_dict)
    print "Number of chunks:", nb_chunks

    if len(names) > 1:
        n_ch = len(labels)
    else:
        n_ch = 1

    cnn = predictor.create_predncnn(config_path, n_ch, len(labels),
                                    imposed_input_size=desired_input,
                                    override_MFP_to_active=MFP,
                                    param_file=param_path)

    name = names[0]

    while True:
        try:
            print "Time per chunk: %.3f" % (time.time() - time_start)
        except:
            pass
        time_start = time.time()

        while True:
            nb_chunk = search_for_chunk(cset.path_head_folder, nb_chunks, name)
            if nb_chunk == -1:
                break
            chunk = cset.chunk_dict[nb_chunk]
            if len(names) == 1:
                break
            if os.path.exists(chunk.folder + names[1] + ".h5"):
                break
            time.sleep(2)

        if nb_chunk != -1:
            if not os.path.exists(chunk.folder):
                os.makedirs(chunk.folder)

            out_path = chunk.folder + name + ".h5"
            print "Processing Chunk: %d" % nb_chunk
            if len(names) == 1:
                raw_data = kd.from_raw_cubes_to_matrix(
                    (np.array(chunk.size) + 2 * offset) / mag,
                    (chunk.coordinates - offset) / mag,
                    mag=mag,
                    invert_data=invert_data)
                raw_data = raw_data[None, :, :, :]
            else:
                raw_data = kd.from_raw_cubes_to_matrix(
                    (np.array(chunk.size) + 2 * offset) / mag,
                    (chunk.coordinates - offset) / mag,
                    mag=mag,
                    invert_data=invert_data)
                time_rec = time.time()
                rec_labels = []
                for label in labels:
                    if label != "none":
                        rec_labels.append(label)
                raw_data = create_recursive_data(rec_labels,
                                                 labels_path=chunk.folder +
                                                             names[1] + ".h5",
                                                 raw_data=raw_data)
                print "Time for creating recursive data: %.3f" % (
                time.time() - time_rec)

            print raw_data.shape
            inference_data = cnn.predictDense(raw_data, as_uint8=True)

            if mag > 1:
                inference_data = interpolate(inference_data, mag=mag)

            f = h5py.File(out_path, "w")
            for ii in range(len(labels)):
                if not labels[ii] == "none":
                    f.create_dataset(labels[ii],
                                     data=inference_data[ii],
                                     compression="gzip")
            f.close()

            folder_path = chunk.folder + "/mutex_%s" % name
            try:
                os.rmdir(folder_path)
            except:
                pass
        else:
            break


if __name__ == "__main__":
    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path("/mnt/axon/home/sdorkenw/SyConnDenseCube/knossodatasets/raw/")
    cset = initialization.initialize_cset(kd, "/mnt/axon/home/sdorkenw/SyConnDenseCube/", [500, 500, 250])
    join_chunky_inference(cset, "/mnt/axon/home/sdorkenw/SyConnDenseCube/models/BIRD_MIGA_config.py",
                          "/mnt/axon/home/sdorkenw/SyConnDenseCube/models/BIRD_MIGA.param",
                          ["MIGA"], ["sj", "vc", "mi"], [200, 200, 100], [50, 270, 270], kd=kd)
