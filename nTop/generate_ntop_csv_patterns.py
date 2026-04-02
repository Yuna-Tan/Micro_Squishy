import numpy as np
import csv
from pathlib import Path


def normalize_to_unit_interval(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def make_grid(n_x: int, n_y: int, n_z: int, spacing: float = 1.0):
    """
    Create a regular 3D grid of point coordinates.

    Returns
    -------
    X, Y, Z : np.ndarray
        Arrays of shape (n_x, n_y, n_z)
    """
    xs = np.arange(n_x, dtype=np.float32) * spacing
    ys = np.arange(n_y, dtype=np.float32) * spacing
    zs = np.arange(n_z, dtype=np.float32) * spacing
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z


def pattern_uniform(n_x: int, n_y: int, n_z: int, value: float = 0.5) -> np.ndarray:
    """Uniform scalar field."""
    return np.full((n_x, n_y, n_z), value, dtype=np.float32)


def pattern_linear_gradient(
    n_x: int,
    n_y: int,
    n_z: int,
    axis: str = "x",
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """
    Linear gradient along one axis.
    axis: 'x', 'y', or 'z'
    """
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    if axis == "x":
        line = np.linspace(low, high, n_x, dtype=np.float32)[:, None, None]
        field = np.broadcast_to(line, (n_x, n_y, n_z)).copy()
    elif axis == "y":
        line = np.linspace(low, high, n_y, dtype=np.float32)[None, :, None]
        field = np.broadcast_to(line, (n_x, n_y, n_z)).copy()
    else:
        line = np.linspace(low, high, n_z, dtype=np.float32)[None, None, :]
        field = np.broadcast_to(line, (n_x, n_y, n_z)).copy()

    return field


def pattern_step_boundary(
    n_x: int,
    n_y: int,
    n_z: int,
    axis: str = "x",
    low: float = 0.2,
    high: float = 0.8,
) -> np.ndarray:
    """
    Two-zone step boundary.
    Half of the volume is low, the other half is high.
    """
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    field = np.full((n_x, n_y, n_z), low, dtype=np.float32)

    if axis == "x":
        mid = n_x // 2
        field[mid:, :, :] = high
    elif axis == "y":
        mid = n_y // 2
        field[:, mid:, :] = high
    else:
        mid = n_z // 2
        field[:, :, mid:] = high

    return field


def pattern_local_peak(
    n_x: int,
    n_y: int,
    n_z: int,
    sigma_ratio: float = 0.18,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """
    Local peak in the center using a 3D Gaussian.
    sigma_ratio controls how spread the peak is.
    """
    X, Y, Z = np.meshgrid(
        np.arange(n_x, dtype=np.float32),
        np.arange(n_y, dtype=np.float32),
        np.arange(n_z, dtype=np.float32),
        indexing="ij",
    )

    cx = (n_x - 1) / 2.0
    cy = (n_y - 1) / 2.0
    cz = (n_z - 1) / 2.0

    sigma = sigma_ratio * min(n_x, n_y, n_z)
    sigma2 = sigma * sigma

    field = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2.0 * sigma2))
    field = normalize_to_unit_interval(field)
    field = low + (high - low) * field
    return field.astype(np.float32)


def write_csv_xyzs(
    filepath: Path,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    S: np.ndarray,
):
    """
    Write CSV in:
    X,Y,Z,S
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "Z", "S"])

        flat_x = X.ravel()
        flat_y = Y.ravel()
        flat_z = Z.ravel()
        flat_s = S.ravel()

        for x, y, z, s in zip(flat_x, flat_y, flat_z, flat_s):
            writer.writerow([float(x), float(y), float(z), float(s)])


def generate_all_patterns(
    out_dir: str = "ntop_csv_patterns",
    n_x: int = 32,
    n_y: int = 32,
    n_z: int = 32,
    spacing: float = 1.0,
):
    out_path = Path(out_dir)
    X, Y, Z = make_grid(n_x, n_y, n_z, spacing=spacing)

    # 1. Uniform
    uniform = pattern_uniform(n_x, n_y, n_z, value=0.5)
    write_csv_xyzs(out_path / "uniform.csv", X, Y, Z, uniform)

    # 2. Linear gradient along x
    gradient = pattern_linear_gradient(n_x, n_y, n_z, axis="x", low=0.0, high=1.0)
    write_csv_xyzs(out_path / "linear_gradient_x.csv", X, Y, Z, gradient)

    # 3. Step boundary along x
    boundary = pattern_step_boundary(n_x, n_y, n_z, axis="x", low=0.2, high=0.8)
    write_csv_xyzs(out_path / "step_boundary_x.csv", X, Y, Z, boundary)

    # 4. Local peak in the middle
    peak = pattern_local_peak(n_x, n_y, n_z, sigma_ratio=0.18, low=0.0, high=1.0)
    write_csv_xyzs(out_path / "local_peak_center.csv", X, Y, Z, peak)

    print(f"Done. CSV files written to: {out_path.resolve()}")


if __name__ == "__main__":
    generate_all_patterns(
        out_dir="ntop_csv_patterns",
        n_x=32,
        n_y=32,
        n_z=32,
        spacing=1.0,
    )