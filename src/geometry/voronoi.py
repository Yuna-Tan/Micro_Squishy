import numpy as np
from scipy.spatial import cKDTree
from src.utils.mesh_utils import field_to_mesh


def generate_voronoi(param_field, n_points=200):
    
    shape = param_field.shape

    # random seeds
    pts = np.random.rand(n_points, 3)
    pts *= np.array(shape)

    grid = np.indices(shape).reshape(3, -1).T

    tree = cKDTree(pts)
    dist, _ = tree.query(grid)

    field = dist.reshape(shape)

    field = field - param_field * field.max()

    return field_to_mesh(field, level=0.0)