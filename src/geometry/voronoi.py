import numpy as np
import pyvista as pv
from squishicalization.scripts import tesselate, sample_elimination


def squish_sampling(field, n_seed=300, n_final=100):
    coords = np.indices(field.shape).reshape(3, -1).T
    rng = np.random.default_rng(42)
    seeds = coords[rng.choice(len(coords), n_seed, replace=False)]
    seeds = np.ascontiguousarray(seeds)

    if np.std(field) < 1e-6:
        print("uniform data detected")
        weights = np.ones(len(seeds), dtype=np.float32)
    else:
        print("non-uniform data")
        weights = field[tuple(seeds.T)].astype(np.float32)

    sampled = sample_elimination(
        points=seeds,
        target=int(n_final),
        mindist=0,
        maxdist=0,
        weights=weights,
        params=[[0, 1], [10, 1]],
    )
    return np.ascontiguousarray(sampled)


def squish_voronoi(mask, seeds):
    mask = np.ascontiguousarray(mask)
    seeds = np.ascontiguousarray(seeds)
    return tesselate(mask=mask, seeds=seeds)


def squish_mesh(volume, spacing, n_regions):
    data = pv.wrap(volume)
    grid = pv.create_grid(data)
    data = grid.sample(data, categorical=True)
    data = data.pack_labels()
    mesh = data.contour_labeled(
        n_labels=n_regions,
        smoothing=True
    )
    mesh = mesh.connectivity("largest")
    return mesh


def generate_voronoi(field, n_seed, n_final):
    # For your canonical pseudo-data, keep the full occupied domain.
    mask = np.ones_like(field, dtype=np.int32)

    seeds = squish_sampling(
        field=field,
        n_seed=int(n_seed),
        n_final=int(n_final)
    )

    volume = squish_voronoi(mask, seeds)
    mesh = squish_mesh(volume, spacing=(1, 1, 1), n_regions=len(seeds))
    return mesh