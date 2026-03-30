import numpy as np
from src.utils.mesh_utils import field_to_mesh


def generate_gyroid(param_field, size=100, cell_size=10):

    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    z = np.linspace(0, 2*np.pi, size)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    freq = 2*np.pi / cell_size
    X *= freq
    Y *= freq
    Z *= freq

    field = (
        np.sin(X)*np.cos(Y) +
        np.sin(Y)*np.cos(Z) +
        np.sin(Z)*np.cos(X)
    )

    thickness = param_field

    solid = (np.abs(field) < thickness)

    field = solid.astype(float)

    return field_to_mesh(field, level=0.5)