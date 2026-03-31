import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


# =========================================================
# Geometry helpers
# =========================================================
def distance_to_segment_points(points, a, b):
    """
    points: (N, 3)
    a, b: (3,)
    return: (N,) distance from points to line segment ab
    """
    pa = points - a[None, :]
    ba = b - a
    denom = np.dot(ba, ba)
    if denom < 1e-12:
        return np.linalg.norm(pa, axis=1)

    t = np.clip((pa @ ba) / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ba[None, :]
    return np.linalg.norm(points - proj, axis=1)


def make_grid_centers(nx, ny, nz):
    X, Y, Z = np.meshgrid(
        np.arange(nx, dtype=np.float32) + 0.5,
        np.arange(ny, dtype=np.float32) + 0.5,
        np.arange(nz, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    return X, Y, Z


# =========================================================
# Global lattice graph construction
# =========================================================
def _add_node(node_map, node_list, p, tol=1e-6):
    """
    Add a node to global node list with quantized hashing.
    p is in voxel coordinates.
    """
    key = tuple(np.round(np.asarray(p, dtype=np.float64) / tol).astype(np.int64))
    if key not in node_map:
        node_map[key] = len(node_list)
        node_list.append(np.asarray(p, dtype=np.float32))
    return node_map[key]


def _build_cubic_graph(nx, ny, nz, cell_size):
    """
    Global graph for cubic lattice:
    nodes at regular grid points
    edges along x, y, z directions
    """
    node_map = {}
    nodes = []
    edges = set()

    gx = list(np.arange(0, nx + 1, cell_size, dtype=np.float32))
    gy = list(np.arange(0, ny + 1, cell_size, dtype=np.float32))
    gz = list(np.arange(0, nz + 1, cell_size, dtype=np.float32))

    if gx[-1] != nx:
        gx.append(np.float32(nx))
    if gy[-1] != ny:
        gy.append(np.float32(ny))
    if gz[-1] != nz:
        gz.append(np.float32(nz))

    idx = {}
    for i, x in enumerate(gx):
        for j, y in enumerate(gy):
            for k, z in enumerate(gz):
                nid = _add_node(node_map, nodes, [x, y, z])
                idx[(i, j, k)] = nid

    nxg = len(gx)
    nyg = len(gy)
    nzg = len(gz)

    for i in range(nxg):
        for j in range(nyg):
            for k in range(nzg):
                a = idx[(i, j, k)]

                if i + 1 < nxg:
                    b = idx[(i + 1, j, k)]
                    edges.add(tuple(sorted((a, b))))
                if j + 1 < nyg:
                    b = idx[(i, j + 1, k)]
                    edges.add(tuple(sorted((a, b))))
                if k + 1 < nzg:
                    b = idx[(i, j, k + 1)]
                    edges.add(tuple(sorted((a, b))))

    return np.array(nodes, dtype=np.float32), list(edges)


def _build_bcc_graph(nx, ny, nz, cell_size):
    """
    Global graph for BCC:
    cell corner nodes + cell centers
    edges from each center to its 8 corners
    """
    node_map = {}
    nodes = []
    edges = set()

    xs = list(np.arange(0, nx + 1, cell_size, dtype=np.float32))
    ys = list(np.arange(0, ny + 1, cell_size, dtype=np.float32))
    zs = list(np.arange(0, nz + 1, cell_size, dtype=np.float32))

    if xs[-1] != nx:
        xs.append(np.float32(nx))
    if ys[-1] != ny:
        ys.append(np.float32(ny))
    if zs[-1] != nz:
        zs.append(np.float32(nz))

    corner_idx = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                nid = _add_node(node_map, nodes, [x, y, z])
                corner_idx[(i, j, k)] = nid

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[j], ys[j + 1]
                z0, z1 = zs[k], zs[k + 1]

                center = np.array(
                    [(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5],
                    dtype=np.float32,
                )
                c = _add_node(node_map, nodes, center)

                corners = [
                    corner_idx[(i, j, k)],
                    corner_idx[(i + 1, j, k)],
                    corner_idx[(i + 1, j + 1, k)],
                    corner_idx[(i, j + 1, k)],
                    corner_idx[(i, j, k + 1)],
                    corner_idx[(i + 1, j, k + 1)],
                    corner_idx[(i + 1, j + 1, k + 1)],
                    corner_idx[(i, j + 1, k + 1)],
                ]
                for q in corners:
                    edges.add(tuple(sorted((c, q))))

    return np.array(nodes, dtype=np.float32), list(edges)


def _build_octet_graph(nx, ny, nz, cell_size):
    """
    Global graph for octet:
    corner nodes + face centers for each cell
    connect face centers to the 4 corners of that face
    """
    node_map = {}
    nodes = []
    edges = set()

    xs = list(np.arange(0, nx + 1, cell_size, dtype=np.float32))
    ys = list(np.arange(0, ny + 1, cell_size, dtype=np.float32))
    zs = list(np.arange(0, nz + 1, cell_size, dtype=np.float32))

    if xs[-1] != nx:
        xs.append(np.float32(nx))
    if ys[-1] != ny:
        ys.append(np.float32(ny))
    if zs[-1] != nz:
        zs.append(np.float32(nz))

    corner_idx = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                nid = _add_node(node_map, nodes, [x, y, z])
                corner_idx[(i, j, k)] = nid

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[j], ys[j + 1]
                z0, z1 = zs[k], zs[k + 1]

                # 6 face centers
                face_centers = [
                    np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z0], dtype=np.float32),  # bottom
                    np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z1], dtype=np.float32),  # top
                    np.array([(x0 + x1) * 0.5, y0, (z0 + z1) * 0.5], dtype=np.float32),  # front
                    np.array([(x0 + x1) * 0.5, y1, (z0 + z1) * 0.5], dtype=np.float32),  # back
                    np.array([x0, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=np.float32),  # left
                    np.array([x1, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=np.float32),  # right
                ]
                fc_ids = [_add_node(node_map, nodes, p) for p in face_centers]

                c000 = corner_idx[(i, j, k)]
                c100 = corner_idx[(i + 1, j, k)]
                c110 = corner_idx[(i + 1, j + 1, k)]
                c010 = corner_idx[(i, j + 1, k)]
                c001 = corner_idx[(i, j, k + 1)]
                c101 = corner_idx[(i + 1, j, k + 1)]
                c111 = corner_idx[(i + 1, j + 1, k + 1)]
                c011 = corner_idx[(i, j + 1, k + 1)]

                face_corner_sets = [
                    [c000, c100, c110, c010],  # bottom
                    [c001, c101, c111, c011],  # top
                    [c000, c100, c101, c001],  # front
                    [c010, c110, c111, c011],  # back
                    [c000, c010, c011, c001],  # left
                    [c100, c110, c111, c101],  # right
                ]

                for fc, corners in zip(fc_ids, face_corner_sets):
                    for q in corners:
                        edges.add(tuple(sorted((fc, q))))

    return np.array(nodes, dtype=np.float32), list(edges)


def build_global_lattice_graph(shape, cell_size, cell_type):
    nx, ny, nz = shape

    if cell_type == "cubic":
        return _build_cubic_graph(nx, ny, nz, cell_size)
    elif cell_type == "bcc":
        return _build_bcc_graph(nx, ny, nz, cell_size)
    elif cell_type == "octet":
        return _build_octet_graph(nx, ny, nz, cell_size)
    else:
        raise ValueError(f"Unknown lattice type: {cell_type}")


# =========================================================
# Global solid-beam signed field
# =========================================================
def build_lattice_signed_field(
    param_field,
    cell_size=10,
    thickness=0.4,
    cell_type="cubic",
    node_radius_factor=1.15,
    smooth_sigma=0.6,
):
    """
    Build signed field from a GLOBAL graph-based solid beam lattice.

    Parameters
    ----------
    param_field : ndarray
        Scalar field in [0, 1], used as mild local thickness modulation.
    cell_size : int
        Unit-cell size in voxels.
    thickness : float
        Strut diameter as fraction of cell size.
        e.g. 0.4 -> diameter = 0.4 * cell_size voxels
    node_radius_factor : float
        Node radius multiplier relative to local strut radius.
    smooth_sigma : float
        Small smoothing to reduce voxel aliasing.
    """
    s = np.clip(param_field.astype(np.float32), 0.0, 1.0)
    nx, ny, nz = s.shape

    nodes, edges = build_global_lattice_graph(
        shape=(nx, ny, nz),
        cell_size=int(cell_size),
        cell_type=cell_type,
    )

    # global voxel-center coordinates
    X, Y, Z = make_grid_centers(nx, ny, nz)

    field = np.full((nx, ny, nz), np.inf, dtype=np.float32)

    # base radius in voxel units
    base_radius_vox = 0.5 * float(thickness) * float(cell_size)

    # avoid too-thin beams
    base_radius_vox = max(base_radius_vox, 1.5)

    # -----------------------------------------------------
    # Add solid beams (capsule-like)
    # -----------------------------------------------------
    for i, j in edges:
        a = nodes[i]
        b = nodes[j]

        # bbox around segment
        rmax = base_radius_vox * 1.25 + 2.0
        pmin = np.floor(np.minimum(a, b) - rmax).astype(int)
        pmax = np.ceil(np.maximum(a, b) + rmax).astype(int)

        x0, y0, z0 = np.maximum(pmin, 0)
        x1, y1, z1 = np.minimum(pmax, [nx - 1, ny - 1, nz - 1])

        if x1 < x0 or y1 < y0 or z1 < z0:
            continue

        xs = X[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        ys = Y[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        zs = Z[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        pts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)

        d = distance_to_segment_points(pts, a, b).reshape(xs.shape)

        # use local scalar field to mildly modulate beam radius
        local_s = s[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        radius_scale = np.clip(1.0 + 0.45 * (local_s - 0.5), 0.80, 1.20)
        local_radius = base_radius_vox * radius_scale

        local_field = d - local_radius
        field[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = np.minimum(
            field[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1],
            local_field.astype(np.float32),
        )

    # -----------------------------------------------------
    # Add solid nodes (spheres)
    # -----------------------------------------------------
    for p in nodes:
        rmax = base_radius_vox * float(node_radius_factor) * 1.25 + 2.0
        pmin = np.floor(p - rmax).astype(int)
        pmax = np.ceil(p + rmax).astype(int)

        x0, y0, z0 = np.maximum(pmin, 0)
        x1, y1, z1 = np.minimum(pmax, [nx - 1, ny - 1, nz - 1])

        if x1 < x0 or y1 < y0 or z1 < z0:
            continue

        xs = X[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        ys = Y[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        zs = Z[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]

        d = np.sqrt((xs - p[0]) ** 2 + (ys - p[1]) ** 2 + (zs - p[2]) ** 2)

        local_s = s[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        radius_scale = np.clip(1.0 + 0.45 * (local_s - 0.5), 0.80, 1.20)
        node_radius = base_radius_vox * float(node_radius_factor) * radius_scale

        local_field = d - node_radius
        field[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = np.minimum(
            field[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1],
            local_field.astype(np.float32),
        )

    if smooth_sigma and smooth_sigma > 0:
        field = gaussian_filter(field, sigma=float(smooth_sigma))

    return field.astype(np.float32)


# =========================================================
# Mesh generation
# =========================================================
def generate_lattice_implicit(
    param_field,
    cell_size=10,
    thickness=0.4,
    cell_type="cubic",
    node_radius_factor=1.15,
    smooth_sigma=0.6,
):
    """
    Generate lattice mesh from a GLOBAL graph-based solid beam field.
    """
    field = build_lattice_signed_field(
        param_field=param_field,
        cell_size=int(cell_size),
        thickness=float(thickness),
        cell_type=cell_type,
        node_radius_factor=float(node_radius_factor),
        smooth_sigma=float(smooth_sigma),
    )

    verts, faces, _, _ = marching_cubes(field, level=0.0)
    faces = faces.reshape(-1, 3)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    mesh = pv.PolyData(verts, faces)
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.connectivity("largest")
    mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)
    return mesh