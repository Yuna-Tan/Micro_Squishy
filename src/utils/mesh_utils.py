import pyvista as pv
import numpy as np
from skimage.measure import marching_cubes


def field_to_mesh(field, level=0.0, spacing=(1,1,1)):
    verts, faces, normals, _ = marching_cubes(field, level=level, spacing=spacing)

    faces = faces.reshape(-1, 3)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    mesh = pv.PolyData(verts, faces)
    return mesh


def save_mesh(mesh, path):
    mesh.save(path)