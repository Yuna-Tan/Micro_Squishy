import pyvista as pv

def compute_safe_bounds(mesh: pv.DataSet, resolution: int):
    domain_width = mesh.bounds[1] - mesh.bounds[0]

    voxel_size = domain_width / resolution

    safe_min = voxel_size * 8.0
    safe_max = domain_width

    print(f"[Safe Bounds]")
    print(f"Voxel size: {voxel_size:.3f}")
    print(f"Safe min: {safe_min:.3f}")
    print(f"Safe max: {safe_max:.3f}")

    return voxel_size, safe_min, safe_max