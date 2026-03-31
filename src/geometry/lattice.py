import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv


# ===============================
# 基础工具
# ===============================
def distance_to_segment(p, a, b):
    """
    计算点 p 到线段 ab 的距离
    p: (..., 3)
    a, b: (3,)
    """
    pa = p - a
    ba = b - a

    t = np.clip(np.sum(pa * ba, axis=-1) / np.sum(ba * ba), 0.0, 1.0)
    proj = a + t[..., None] * ba
    return np.linalg.norm(p - proj, axis=-1)


# ===============================
# 单元结构定义（可扩展）
# ===============================
def get_unit_cell_edges(cell_type):
    """
    返回 unit cell 的边（节点连接）
    """
    # cube corners
    V = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]
    ])

    if cell_type == "cubic":
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]

    elif cell_type == "bcc":
        center = np.array([[0.5,0.5,0.5]])
        V = np.vstack([V, center])
        c = len(V) - 1
        edges = [(i, c) for i in range(8)]

    elif cell_type == "octet":
        edges = [
            (0,6),(1,7),(2,4),(3,5),
            (0,7),(1,6),(2,5),(3,4)
        ]

    else:
        raise ValueError("Unknown lattice")

    return V, edges



# ===============================
# 主函数（核心）
# ===============================
def generate_lattice_implicit(
    param_field,
    cell_size=10,
    thickness=0.5,
    cell_type="cubic"
):
    """
    生成可打印 lattice（implicit 方法）
    """

    nx, ny, nz = param_field.shape

    # 3D grid
    X, Y, Z = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij"
    )

    points = np.stack([X, Y, Z], axis=-1).astype(float)

    # normalize to cell coordinates
    local = (points % cell_size) / cell_size

    # unit cell
    V, edges = get_unit_cell_edges(cell_type)

    # 初始化 distance field
    dist_field = np.full((nx, ny, nz), np.inf)

    # 对每条 strut 计算距离
    for i, j in edges:
        a = V[i]
        b = V[j]

        d = distance_to_segment(local, a, b)
        dist_field = np.minimum(dist_field, d)

    # ===============================
    # 🔥 核心：thickness + scalar mapping
    # ===============================

    # ===============================
    # mild general scalar control
    # ===============================
    s = param_field.astype(np.float32)

    # param_field should already be normalized in generate_sample.py,
    # but clip again for safety
    s = np.clip(s, 0.0, 1.0)

    # 0.5 is neutral
    centered = s - 0.5

    # mild contrast, not aggressive
    # low side stays present, high side does not become solid too quickly
    scale = 1.0 + 0.6 * centered

    # keep thickness variation narrow
    scale = np.clip(scale, 0.75, 1.25)

    local_thickness = thickness * scale
    field = dist_field - local_thickness

    # ===============================
    # marching cubes
    # ===============================
    verts, faces, _, _ = marching_cubes(field, level=0)

    faces = faces.reshape(-1, 3)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    mesh = pv.PolyData(verts, faces)

    return mesh