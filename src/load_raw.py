from pathlib import Path
import numpy as np
import pyvista as pv

def load_raw(
    path,
    shape=(100, 100, 100),
    dtype=np.uint16
):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)

    if data.size != shape[0]*shape[1]*shape[2]:
        raise ValueError("Size mismatch")

    return data.reshape(shape, order="F").astype(np.float32) / 65535.0

def load_raw_to_fieldlat_mesh(
    path: str | Path,
    shape=(100, 100, 100),
    spacing=(0.2, 0.2, 0.2),
    normalize=True
) -> pv.UnstructuredGrid:
    
    path = Path(path)

    # read raw
    data = np.fromfile(path, dtype=np.uint16)

    expected = np.prod(shape)
    if data.size != expected:
        raise ValueError(f"Size mismatch: expected {expected}, got {data.size}")

    # reshape
    field = data.reshape(shape, order="F").astype(np.float32)

    # normalize
    if normalize:
        field = field / 65535.0

    # PyVista grid
    grid = pv.ImageData(
        dimensions=shape,
        spacing=spacing
    )

    mesh = grid.cast_to_unstructured_grid()

    mesh.point_data["stress"] = field.ravel(order="F")

    print("------stress------")
    print(field.ravel(order="F")[:20])
    print(field.ravel(order="F")[60:80])
    print(field.ravel(order="F")[-20:])
    return mesh