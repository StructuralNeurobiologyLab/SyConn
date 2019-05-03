from zmesh import Mesher
import numpy as np


def find_meshes():
    labels = create_toy_data(5, 5, 5, 10)
    mesher = Mesher((1, 1, 1))  # anisotropy of image
    mesher.mesh(labels)  # initial marching cubes pass

    meshes = {}
    for obj_id in mesher.ids():
      meshes[obj_id] = mesher.get_mesh(obj_id, normals=True, simplification_factor=0,
                        max_simplification_error=8)
      print("v=     ", meshes[obj_id].vertices, "\n\n\n m=   ", meshes[obj_id].faces, "\n\n\n normals  =", meshes[obj_id].normals)
      mesher.erase(obj_id)  # delete high res mesh

    mesher.clear()  # clear memory retained by mesher
    print(meshes)
    return meshes


def create_toy_data(size1, size2, size3, moduloo):
    np.random.seed(0)
    matrix = np.zeros(shape=(size1, size2, size3), dtype=int)
    for i in range(size1):
        for j in range(size2):
            for k in range(size3):
                matrix[i, j, k] = np.random.randint(moduloo, size=1)
    return matrix


find_meshes()
