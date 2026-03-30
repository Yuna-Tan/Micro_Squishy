import numpy as np
import pyvista as pv

def compute_safe_bounds(mesh: pv.DataSet, resolution: int):
    domain_width_x = mesh.bounds[1] - mesh.bounds[0]
    voxel_size = domain_width_x / resolution

    safe_min = voxel_size * 8.0
    safe_max = domain_width_x

    return voxel_size, safe_min, safe_max