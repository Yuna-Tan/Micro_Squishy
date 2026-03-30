import numpy as np
import pyvista as pv

from squishicalization.scripts import tesselate, sample_elimination


# =========================
# 1️⃣ sampling（直接调用）
# =========================
def squish_sampling(field, n_seed=300, n_final=100):

    field = (field - field.min()) / (field.max() - field.min() + 1e-8)

    # coords = np.array(np.nonzero(field)).T
    coords = np.ascontiguousarray(
        np.array(np.nonzero(field)).T
    )

    rng = np.random.default_rng(42)
    seeds = coords[rng.choice(len(coords), n_seed, replace=False)]
    seeds = np.ascontiguousarray(seeds)

    weights = field[tuple(seeds.T)]

    # ⭐关键：完全使用原函数
    sampled = sample_elimination(
        points=seeds,
        target=n_final,
        mindist=0,
        maxdist=0,
        weights=weights,
        params=[[0,1],[10,1]]
    )

    return sampled


# =========================
# 2️⃣ Voronoi（直接调用）
# =========================
def squish_voronoi(mask, seeds):
    mask = np.ascontiguousarray(mask)
    seeds = np.ascontiguousarray(seeds)

    return tesselate(mask=mask, seeds=seeds)


# =========================
# 3️⃣ mesh（轻封装）
# =========================
def squish_mesh(volume, spacing, n_regions):

    data = pv.wrap(volume)
    grid = pv.create_grid(data)

    data = grid.sample(data, categorical=True)
    data = data.pack_labels()

    mesh = data.contour_labeled(
        n_labels=n_regions,
        smoothing=True
    )

    mesh = mesh.connectivity('largest')

    return mesh


# =========================
# 4️⃣ 总入口（你pipeline用这个）
# =========================
def generate_voronoi(field):

    mask = np.zeros_like(field)
    mask[field > field.min()] = 1

    seeds = squish_sampling(field)

    volume = squish_voronoi(mask, seeds)

    mesh = squish_mesh(volume, spacing=(1,1,1), n_regions=len(seeds))

    return mesh