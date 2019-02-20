# distutils: language = c++
from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t, uint32_t
cimport cython
from libc.stdlib cimport rand
from libcpp.map cimport map
from cython.operator import dereference, postincrement
import numpy as np
from libcpp.vector cimport vector
import timeit


def uniqueCython(chunk, return_index=False, return_inverse=False,return_counts=False, axis=None):

    if axis == None:
            chunk.flatten()
    else:
        chunk = chunk ## check how to choose good axis maybe np.take()


    cdef int [:] chunk_view = chunk
    cdef map[int, int] mapa
    cdef int [:] unique_ids
    cdef int [:] index
    cdef int [:] counts
    cdef int [:] inverse
    cdef int act_size=0, iterator=0, ind=0
    cdef int key = 0

    for i in range(chunk_view.shape[0]):
        key = chunk_view[i]
        if mapa.find(key) == mapa.end(): #element doesn't exist
            mapa[key] = act_size
            index[act_size]=iterator
            unique_ids[act_size]=key
            counts[act_size]=1
            inverse[iterator]=act_size
            act_size+=1
        else:
            ind=unique_ids[key]
            counts[ind]+=1
            inverse[iterator]=ind
        iterator+=1

    return 1

cdef uniqueCython_elements_counts(chunky):

    cdef uint32_t [:] chunk_view = chunky.flatten()
    cdef map[uint32_t, int] mapa
    cdef vector[uint32_t] unique_ids
    cdef vector[int] counts

    for i in range(chunk_view.shape[0]):
        mapa[chunk_view[i]] = mapa[chunk_view[i]] + 1

    cdef map[uint32_t, int].iterator it = mapa.begin()

    while it != mapa.end():
        unique_ids.push_back(dereference(it).first)
        counts.push_back(dereference(it).second)
        postincrement(it)
    return unique_ids, counts


cdef uniqueCython_most_often(chunky):
    cdef uint32_t[:, :, :] chunk = chunky
    cdef map[uint32_t, int] mapa

    for i in range(chunk.shape[0]):
        for j in range(chunk.shape[1]):
            for k in range(chunk.shape[2]):
                mapa[chunk[i][j][k]] += 1

    cdef map[uint32_t, int].iterator it = mapa.begin()
    cdef uint32_t key = 0
    cdef int counts = 0

    while it != mapa.end():
        if dereference(it).second > counts:
            key = dereference(it).first
            counts = dereference(it).second
        postincrement(it)
    return key


def abc(chunk):
    u_i, c = np.unique(chunk, return_counts = True)

    return u_i[np.argmax(c)]



def create_toy_data(size1, size2, size3, moduloo):

   matrix = np.zeros((size1, size2, size3), dtype = np.uint32)
   for i in range(size1):
       for j in range(size2):
            matrix[i,j,:] = np.random.randint(moduloo, size=size3)
   return matrix

def wrapper(func, *args, **kwargs):
   def wrapped():
       return func(*args, **kwargs)
   return wrapped

def main():

    chunk = create_toy_data(3,3,3,10)


    print("numpy")
    wrapped = wrapper(np.unique, chunk, return_counts=True)
    print (timeit.timeit(wrapped, number=10))


    print ("uniqueCython_most_often")
    wrapped = wrapper(uniqueCython_elements_counts, chunk)
    print (timeit.timeit(wrapped, number=10))


'''
    bounding_box = create_toy_data(2,3, 10)
    print ("in_bounding_box__Numba")
    wrapped = wrapper(in_bounding_box, coords, bounding_box)
    print (timeit.timeit(wrapped, number=10000))   
    print ("in_bounding_box__Maria")
    wrapped = wrapper(in_bounding_boxC, coords, bounding_box)
    print (timeit.timeit(wrapped, number=10000))
    '''


main()



































