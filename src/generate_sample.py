from src.load_raw import load_raw
from src.core.scalar_field import normalize_scalar, normalize_scalar_lattice
from src.core.mapping import scalar_to_target
from src.core.calibration import map_to_param

from src.geometry.gyroid import generate_gyroid
from src.geometry.lattice import generate_lattice_implicit
from src.geometry.tpms import generate_tpms_from_raw
from src.geometry.voronoi import generate_voronoi
from src.geometry.spinodal import generate_spinodal

from src.load_raw import load_raw_to_fieldlat_mesh

import numpy as np


def generate_sample(raw_path, family, structure, params): 

    if family == "TPMS":
        mesh = load_raw_to_fieldlat_mesh(raw_path)

        return generate_tpms_from_raw(
            raw_mesh=mesh,
            resolution=int(params["resolution"]),
            max_cell_size=params["max_cell_size"],
            min_cell_size=params["min_cell_size"],
            threshold=params["threshold"],
            lattice_type=structure
        )
    
    elif family == "Voronoi":

        scalar = load_raw(raw_path)
        scalar = normalize_scalar(scalar)

        mesh = generate_voronoi(
            scalar, 
            n_seed=params["seed_count"], 
            n_final=params["final_points"]
        )

        return mesh
        
    elif family == "Spinodal":
        scalar = load_raw(raw_path)
        scalar = normalize_scalar(scalar)

        mesh = generate_spinodal(
            scalar_field=scalar,
            sigma=float(params["sigma"]),
            base_threshold=float(params["threshold"]),
            encode_strength=float(params["encode_strength"]),
            data_smoothing=float(params["data_smoothing"]),
            seed=int(params["seed"]),
        )

        return mesh

    elif family == "Lattice":
        scalar = load_raw(raw_path)
        scalar = normalize_scalar_lattice(scalar)

        mesh = generate_lattice_implicit(
            param_field=scalar,
            cell_size=params["cell_size"],
            thickness=params["thickness"],
            cell_type=structure
        )

        return mesh

    else:
        raise ValueError("Unknown Family")