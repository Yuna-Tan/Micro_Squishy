import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
from typing import Union
from scipy.ndimage import gaussian_filter
from scipy.ndimage import laplace

# def generate_spinodal_field(shape, feature_size=12.0, seed=42):
#     """
#     Generate spinodal-like bicontinuous field using
#     band-limited Gaussian random field (spectral method).

#     Parameters
#     ----------
#     shape : tuple
#         grid shape (nx, ny, nz)

#     feature_size : float
#         controls characteristic pore size

#     seed : int
#         random seed
#     """

#     rng = np.random.default_rng(seed)

#     nx, ny, nz = shape

#     # random complex spectrum
#     real = rng.standard_normal(shape)
#     imag = rng.standard_normal(shape)
#     spectrum = real + 1j * imag

#     # frequency grids
#     kx = np.fft.fftfreq(nx)[:, None, None]
#     ky = np.fft.fftfreq(ny)[None, :, None]
#     kz = np.fft.fftfreq(nz)[None, None, :]

#     k = np.sqrt(kx**2 + ky**2 + kz**2)

#     # dominant wavelength
#     k0 = 1.0 / feature_size

#     # bandwidth of frequency peak
#     bandwidth = 0.4 * k0

#     envelope = np.exp(-((k - k0)**2) / (2 * bandwidth**2))

#     envelope[0,0,0] = 0.0

#     filtered_spectrum = spectrum * envelope

#     field = np.fft.ifftn(filtered_spectrum).real

#     # normalize
#     field = (field - field.mean()) / (field.std() + 1e-8)

#     return field.astype(np.float32)

def get_lattice_field(
    lattice_type: str, 
    px: np.ndarray, 
    py: np.ndarray, 
    pz: np.ndarray
) -> np.ndarray:
    """
    Computes the scalar field values for the specified lattice topology.
    
    Args:
        lattice_type (str): 'gyroid', 'diamond', 'primitive', or 'lidinoid'.
        px, py, pz (np.ndarray): The phase maps (coordinates * frequency or integrated phase).
        
    Returns:
        np.ndarray: The computed scalar field.
    """
    # Use phase coordinates directly
    sx, sy, sz = px, py, pz

    if lattice_type == 'lidinoid':
        # Lidinoid Special Case:
        # The logic maintains gx = 2 * phase to preserve the 2x density 
        # relative to the map naturally found in the sin(2x) term.
        
        # Double frequency terms (2 * phase)
        gx, gy, gz = 2.0 * sx, 2.0 * sy, 2.0 * sz
        
        term1 = 0.5 * (np.sin(gx)*np.cos(sy)*np.sin(sz) + 
                       np.sin(gy)*np.cos(sz)*np.sin(sx) + 
                       np.sin(gz)*np.cos(sx)*np.sin(sy))
        
        term2 = 0.5 * (np.cos(gx)*np.cos(gy) + 
                       np.cos(gy)*np.cos(gz) + 
                       np.cos(gz)*np.cos(gx))
                       
        return term1 - term2 + 0.15

    elif lattice_type == 'gyroid':
        return np.sin(sx) * np.cos(sy) + np.sin(sy) * np.cos(sz) + np.sin(sz) * np.cos(sx)

    elif lattice_type == 'diamond':
        # Schwarz D
        return (np.sin(sx) * np.sin(sy) * np.sin(sz) + 
                np.sin(sx) * np.cos(sy) * np.cos(sz) + 
                np.cos(sx) * np.sin(sy) * np.cos(sz) + 
                np.cos(sx) * np.cos(sy) * np.sin(sz))

    elif lattice_type == 'primitive':
        # Schwarz P
        return np.cos(sx) + np.cos(sy) + np.cos(sz)

    # elif lattice_type == 'spinodal':

    #     shape = px.shape

    #     # pore size relative to domain
    #     feature_size = max(shape[0] / 10.0, 6.0)

    #     return generate_spinodal_field(
    #         shape=shape,
    #         feature_size=feature_size,
    #         seed=42
    #     )
    
    else:
        raise ValueError(f"Unsupported lattice_type: '{lattice_type}'. "
                         "Choose from 'gyroid', 'diamond', 'primitive', 'lidinoid'.")

def generate_adaptive_lattice(
    mesh: pv.DataSet,
    field_name: str,
    lattice_type: str = 'gyroid',
    structure_mode: str = 'sheet',
    resolution: int = 50,
    base_scale: float = 10.0,
    dense_scale: float = 25.0,
    threshold: float = 0.3,
    pad_width: int = 2,
    gradient_strategy: str = 'blend'
) -> pv.PolyData:
    """
    Generates an adaptive lattice (TPMS) using either Field Blending or Domain Warping.

    Args:
        mesh (pv.DataSet): The source mesh containing the scalar field.
        field_name (str): The name of the scalar field in point_data.
        lattice_type (str): Topology type ('gyroid', 'diamond', 'primitive', 'lidinoid').
        structure_mode (str): 'sheet' (Shell) or 'strut' (Skeletal). 
                              Defaults to 'sheet'.
        resolution (int): Resolution of the voxel grid (cubed).
        base_scale (float): The frequency k for low-stress areas (k_min).
        dense_scale (float): The frequency k for high-stress areas (k_max).
        threshold (float): Controls density/thickness.
                           - In 'sheet' mode: Defines Wall Thickness (must be > 0).
                           - In 'strut' mode: Defines Volume Fraction/Isovalue.
        pad_width (int): Number of voxel layers to force to void at the boundaries.
                         This ensures the mesh is watertight (capped). Default is 2.
        gradient_strategy (str): Strategy for transitioning between scales.
                                 - 'blend': Generates two full fields and blends them using a sigmoid.
                                            Better for preserving lattice integrity (default).
                                 - 'warp': Modulates frequency directly (Domain Warping).
                                           Uses Phase Integration to prevent coordinate drift artifacts.

    Returns:
        pv.PolyData: The extracted isosurface mesh of the lattice.
    """
    
    # 1. Define the Bounding Box and Voxel Grid
    bounds = mesh.bounds
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    z = np.linspace(bounds[4], bounds[5], resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 2. Interpolate the Field Data onto the Voxel Grid
    grid = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(
            (bounds[1] - bounds[0]) / (resolution - 1),
            (bounds[3] - bounds[2]) / (resolution - 1),
            (bounds[5] - bounds[4]) / (resolution - 1),
        ),
        origin=(bounds[0], bounds[2], bounds[4])
    )
    
    sampled_grid = grid.sample(mesh)
    field_values = sampled_grid.point_data[field_name].reshape((resolution, resolution, resolution), order='F')

    if np.any(np.isnan(field_values)):
        field_values = np.nan_to_num(field_values, nan=np.nanmin(field_values))

    # 3. Normalize Field to Weight Map [0.0, 1.0]
    f_min, f_max = field_values.min(), field_values.max()
    if f_max > f_min:
        w = (field_values - f_min) / (f_max - f_min)
    else:
        w = np.zeros_like(field_values)

    # 4. Generate Lattice Field based on Gradient Strategy
    k_min = base_scale
    k_max = dense_scale

    if gradient_strategy == 'warp':
        # Domain Warping: Modulate frequency spatially using Phase Integration
        k_map = k_min + (k_max - k_min) * w
        
        # Calculate Instantaneous Phase by numerically integrating the frequency map
        # This prevents "coordinate drift" artifacts far from the origin
        phase_x = np.cumsum(k_map, axis=0) * grid.spacing[0]
        phase_y = np.cumsum(k_map, axis=1) * grid.spacing[1]
        phase_z = np.cumsum(k_map, axis=2) * grid.spacing[2]
        
        result = get_lattice_field(lattice_type, phase_x, phase_y, phase_z)

    elif gradient_strategy == 'blend':
        # Field Blending: Generate two discrete fields and blend
        
        # Calculate phases (simple multiplication)
        px_low, py_low, pz_low = k_min * X, k_min * Y, k_min * Z
        px_high, py_high, pz_high = k_max * X, k_max * Y, k_max * Z

        field_low = get_lattice_field(lattice_type, px_low, py_low, pz_low)
        field_high = get_lattice_field(lattice_type, px_high, py_high, pz_high)

        # Apply Sigmoid function to weight map to sharpen the transition
        # This reduces the "transition zone" where destructive interference occurs
        # 10 is the steepness factor
        # w_sharp = 1.0 / (1.0 + np.exp(-10.0 * (w - 0.5)))
        w_sharp = w
        
        # Linear Blend of the fields using sharpened weights
        result = field_low * (1.0 - w_sharp) + field_high * w_sharp

    else:
        raise ValueError(f"Unknown gradient_strategy: '{gradient_strategy}'. Use 'blend' or 'warp'.")

    # 5. Apply Structure Mode Logic
    # if lattice_type == "spinodal":
    #     base_threshold = threshold * 1.2
    #     dense_threshold = threshold * 0.4

    #     local_threshold = base_threshold - (base_threshold - dense_threshold) * w

    #     if structure_mode == "sheet":
    #         scalar_field = np.abs(result) - local_threshold

    #     elif structure_mode == "strut":
    #         scalar_field = result - local_threshold

    # else:
    if structure_mode == 'sheet':
        # Sheet/Shell: Wall around the zero isosurface
        # Values < 0 are inside the wall (solid), Values > 0 are air (void)
        scalar_field = np.abs(result) - threshold
    elif structure_mode == 'strut':
        # Strut/Network: Solidify one domain
        # Using the requested formula: blended_lattice - threshold
        scalar_field = result - threshold
    else:
        raise ValueError(f"Unknown structure_mode: '{structure_mode}'. Use 'sheet' or 'strut'.")

    # 6. Apply Padding to force watertight mesh
    if pad_width > 0:
        scalar_field[:pad_width, :, :] = 1.0
        scalar_field[-pad_width:, :, :] = 1.0
        scalar_field[:, :pad_width, :] = 1.0
        scalar_field[:, -pad_width:, :] = 1.0
        scalar_field[:, :, :pad_width] = 1.0
        scalar_field[:, :, -pad_width:] = 1.0

    
    # 7. Extract Surface using Marching Cubes
    verts, faces, normals, values = marching_cubes(
        scalar_field, 
        level=0.0, 
        spacing=grid.spacing
    )

    # Offset vertices by the origin
    verts += np.array([bounds[0], bounds[2], bounds[4]])

    # 8. Convert to PyVista Mesh
    n_faces = faces.shape[0]
    padding = np.full((n_faces, 1), 3)
    pv_faces = np.hstack((padding, faces)).flatten()

    surface_mesh = pv.PolyData(verts, pv_faces)
    
    return surface_mesh
