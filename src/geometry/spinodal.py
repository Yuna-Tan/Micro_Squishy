import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if abs(mx - mn) < 1e-8:
        return np.full_like(x, 0.5, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _safe_marching(binary: np.ndarray) -> pv.PolyData:
    # marching_cubes needs both phases present
    if binary.min() == binary.max():
        raise ValueError(
            "Spinodal threshold produced an all-solid or all-void volume. "
            "Try lowering encode_strength or adjusting base_threshold."
        )

    verts, faces, _, _ = marching_cubes(binary.astype(np.float32), level=0.5)
    faces = faces.reshape(-1, 3)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(verts, faces)
    return mesh.clean().triangulate()


def generate_spinodal(
    scalar_field: np.ndarray,
    sigma: float = 3.0,
    base_threshold: float = 0.0,
    encode_strength: float = 0.8,
    data_smoothing: float = 1.0,
    seed: int = 42,
) -> pv.PolyData:
    """
    Data-driven spinodal generation.

    Parameters
    ----------
    scalar_field : np.ndarray
        Raw or normalized scalar field. Higher values should become denser/stiffer.
    sigma : float
        GRF smoothing. Larger sigma -> larger spinodal features.
    base_threshold : float
        Global threshold offset for the GRF.
    encode_strength : float
        How strongly the scalar field modulates local density.
        0.0 = no data encoding, larger = stronger local contrast.
    data_smoothing : float
        Optional smoothing of the scalar control field to avoid noisy local transitions.
    seed : int
        Random seed.

    Returns
    -------
    pv.PolyData
        Spinodal mesh.
    """
    rng = np.random.default_rng(seed)

    # 1) normalize input scalar to [0, 1]
    s = _normalize01(scalar_field)

    # 2) smooth scalar control field slightly, so encoding is region-level not voxel-noise
    if data_smoothing > 0:
        s = gaussian_filter(s, sigma=data_smoothing)

    s = _normalize01(s)

    # 3) Gaussian random field carrier
    noise = rng.standard_normal(size=s.shape).astype(np.float32)
    grf = gaussian_filter(noise, sigma=sigma)
    grf = (grf - grf.mean()) / (grf.std() + 1e-8)

    # 4) local threshold field
    # high scalar -> lower threshold -> more solid
    # low scalar  -> higher threshold -> more void
    threshold_field = base_threshold + encode_strength * (0.5 - s)

    # 5) threshold against local field
    binary = grf > threshold_field

    return _safe_marching(binary)