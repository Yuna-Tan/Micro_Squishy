import pyvista as pv
from fieldlat.core import generate_adaptive_lattice


def generate_tpms_from_raw(raw_mesh,
                           lattice_type="gyroid",
                           min_cell_size=5.0,
                           max_cell_size=15.0,
                           threshold=0.4,
                           resolution=100):

    # convert cell size → frequency
    k_min = 2.0 * 3.1415926 / max_cell_size
    k_max = 2.0 * 3.1415926 / min_cell_size

    lattice = generate_adaptive_lattice(
        mesh=raw_mesh,
        field_name="stress",
        resolution=resolution,
        dense_scale=k_max,
        base_scale=k_min,
        threshold=threshold,
        lattice_type=lattice_type,
        structure_mode="sheet",   
        gradient_strategy="blend"
    )

    return lattice