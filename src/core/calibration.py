def gyroid_inverse(k):
    # example placeholder
    return 0.3 + k * 0.5

def voronoi_inverse(k):
    return 0.2 + k * 0.6

def lattice_inverse(k):
    return 0.25 + k * 0.55


def map_to_param(k_field, family):
    if family == "gyroid":
        return gyroid_inverse(k_field)
    elif family == "voronoi":
        return voronoi_inverse(k_field)
    elif family == "lattice":
        return lattice_inverse(k_field)
    else:
        raise ValueError("Unknown family")