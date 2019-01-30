from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t
cimport cython
from libcpp.map cimport map
from cython.operator import dereference, postincrement
import timeit
from libc.stdlib cimport rand



cdef struct point:
	int xMin, yMin, zMin
	int xMax, yMax, zMax

def extract_bounding_box(uint64_t[:,:,:] chunk):
	cdef map[uint64_t, point] bB #bB = boundingBox
	
	for x in range(chunk.shape[0]):
		for y in range(chunk.shape[1]):
			for z in range(chunk.shape[2]):
				bB[chunk[x,y,z]].xMin = min(bB[chunk[x,y,z]].xMin, x)
				bB[chunk[x,y,z]].xMax = max(bB[chunk[x,y,z]].xMax, x)
			
				bB[chunk[x,y,z]].yMin = min(bB[chunk[x,y,z]].yMin, y)
				bB[chunk[x,y,z]].yMax = min(bB[chunk[x,y,z]].yMax, y)
			
				bB[chunk[x,y,z]].zMin = min(bB[chunk[x,y,z]].zMin, z)
				bB[chunk[x,y,z]].zMax = min(bB[chunk[x,y,z]].zMax, z)
	return bB


def create_toy_data(int size):
	cdef uint64_t[:, :, :] matrix = cvarray(shape=(size, size, size), itemsize=sizeof(uint64_t), format='Q')
	for i in range(size):
		for j in range(size):
			for k in range(size):
				matrix[i, j, k] = rand() % 10
	return matrix


def printMemView(uint64_t[:, :, :] myArray):
	for i in range(myArray.shape[0]):
		for j in range(myArray.shape[1]):
			for k in range(myArray.shape[2]):
				print myArray[i,j,k],
			print("")
		print("")

cdef uint64_t[:, :, :] chunk
chunk = create_toy_data(8)
extract_bounding_box(chunk)
