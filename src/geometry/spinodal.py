import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter


def generate_spinodal(
    shape,
    sigma=3.0,
    threshold=0.0,
    seed=42
):
    """
    Generate spinodal bicontinuous structure
    """

    np.random.seed(seed)

    # 1️⃣ Gaussian random field
    noise = np.random.randn(*shape)

    # 2️⃣ smoothing → control feature size
    field = gaussian_filter(noise, sigma=sigma)

    # normalize（可选）
    field = (field - field.mean()) / (field.std() + 1e-8)

    # 3️⃣ threshold → binary structure
    binary = field > threshold

    # 4️⃣ marching cubes
    verts, faces, _, _ = marching_cubes(binary.astype(float), level=0.5)

    faces = faces.reshape(-1, 3)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    mesh = pv.PolyData(verts, faces)

    # clean mesh（非常重要）
    mesh = mesh.clean().triangulate()

    return mesh