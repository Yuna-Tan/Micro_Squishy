from src.load_raw import load_raw
from src.core.scalar_field import normalize_scalar
from src.core.mapping import scalar_to_target
from src.core.calibration import map_to_param

from src.geometry.gyroid import generate_gyroid
from src.geometry.voronoi import generate_voronoi
from src.geometry.lattice import generate_lattice_implicit

from src.geometry.tpms import generate_tpms_from_raw
from src.load_raw import load_raw_to_fieldlat_mesh

import numpy as np


def generate_sample(raw_path, family, structure, params): 

    if family == "TPMS":
        mesh = load_raw_to_fieldlat_mesh(raw_path)

        return generate_tpms_from_raw(
            raw_mesh=mesh,
            resolution=int(params["resolution"]),
            max_cell_size=params["min_cell_size"],
            min_cell_size=params["max_cell_size"],
            threshold=params["threshold"],
            lattice_type=structure
        )
    
    elif family == "voronoi":
        scalar = load_raw(raw_path)
        scalar = normalize_scalar(scalar)

        k = scalar
        param = map_to_param(k, family)

        mesh = generate_voronoi(param)

    elif family == "Lattice":
        scalar = load_raw(raw_path)
        scalar = normalize_scalar(scalar)

        mesh = generate_lattice_implicit(
            param_field=scalar,
            cell_size=params["cell_size"],
            thickness=params["thickness"],
            cell_type=structure
        )

        return mesh

    else:
        raise ValueError("Unknown")

    return mesh