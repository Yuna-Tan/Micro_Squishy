from src.load_raw import load_raw
from src.core.scalar_field import normalize_scalar
from src.core.mapping import scalar_to_target
from src.core.calibration import map_to_param

from src.geometry.gyroid import generate_gyroid
from src.geometry.voronoi import generate_voronoi
from src.geometry.lattice import generate_lattice

from src.geometry.tpms import generate_tpms_from_raw
from src.load_raw import load_raw_to_fieldlat_mesh

tpms_family = ["gyroid", "diamond", "primitive", "lidinoid"]

def generate_sample(raw_path, family, contrast="high"):

    

    if family in tpms_family:
        mesh = load_raw_to_fieldlat_mesh(raw_path)

        tpms = generate_tpms_from_raw(
            raw_mesh=mesh,
            lattice_type=family,
            min_cell_size=6.0,
            max_cell_size=15.0,
            threshold=0.4,
            resolution=100
        )

        return tpms
    
    elif family == "voronoi":
        scalar = load_raw(raw_path)
        scalar = normalize_scalar(scalar)

        k = scalar
        param = map_to_param(k, family)

        mesh = generate_voronoi(param)

    elif family == "lattice":
        scalar = load_raw(raw_path)
        scalar = normalize_scalar(scalar)

        k = scalar
        param = map_to_param(k, family)

        mesh = generate_lattice(param)

    else:
        raise ValueError("Unknown")

    return mesh