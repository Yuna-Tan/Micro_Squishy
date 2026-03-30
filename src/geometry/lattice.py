import numpy as np
from src.utils.mesh_utils import field_to_mesh


def generate_lattice(param_field, spacing=10):

    shape = param_field.shape
    X, Y, Z = np.indices(shape)

    lattice = (
        ((X % spacing) < 2) |
        ((Y % spacing) < 2) |
        ((Z % spacing) < 2)
    ).astype(float)

    field = lattice - param_field

    return field_to_mesh(field, level=0.0)