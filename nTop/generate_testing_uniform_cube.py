import numpy as np
import csv
from pathlib import Path


def make_uniform_csv_for_ntop(
    output_csv: str = "uniform_20mm_cube.csv",
    cube_size_mm: float = 20.0,
    spacing_mm: float = 1.0,
    scalar_value: float = 0.5,
):
    """
    Generate a CSV in X,Y,Z,S format for a uniform scalar point map.

    Parameters
    ----------
    output_csv : str
        Output CSV filename.
    cube_size_mm : float
        Cube size in mm. 20.0 means a 20 mm x 20 mm x 20 mm cube.
    spacing_mm : float
        Point spacing in mm.
        Example:
            1.0 -> 21 points per axis from 0 to 20
            0.5 -> 41 points per axis from 0 to 20
    scalar_value : float
        Uniform scalar value assigned to every point.
    """
    # Include both ends: 0 and cube_size_mm
    coords = np.arange(0.0, cube_size_mm + 1e-6, spacing_mm, dtype=np.float32)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "Z", "S"])

        for x in coords:
            for y in coords:
                for z in coords:
                    writer.writerow([float(x), float(y), float(z), float(scalar_value)])

    n = len(coords)
    print(f"Saved: {output_path.resolve()}")
    print(f"Cube size: {cube_size_mm} mm")
    print(f"Spacing: {spacing_mm} mm")
    print(f"Grid points per axis: {n}")
    print(f"Total points: {n**3}")
    print(f"Uniform scalar value: {scalar_value}")


if __name__ == "__main__":
    make_uniform_csv_for_ntop(
        output_csv="uniform_20mm_cube.csv",
        cube_size_mm=20.0,
        spacing_mm=1.0,
        scalar_value=0.5,
    )