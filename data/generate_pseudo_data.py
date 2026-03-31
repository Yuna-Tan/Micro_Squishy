from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal
import json
import numpy as np


# =========================================================
# Type aliases
# =========================================================
Array3D = np.ndarray
DirectionName = Literal[
    "x+", "x-", "y+", "y-", "z+", "z-",
    "diag_xy+", "diag_xy-"
]
ContrastName = Literal["low", "high"]


# =========================================================
# Squishicalization compatibility settings
# Based on your current loader:
#   DEFAULT_SHAPE = (100, 100, 100)
#   DEFAULT_DTYPE = numpy.uint16
#   reshape(..., order='F')
# =========================================================
SQUISH_DEFAULT_SHAPE = (100, 100, 100)
SQUISH_DEFAULT_DTYPE = np.uint16
SQUISH_UINT16_MAX = np.iinfo(np.uint16).max


@dataclass
class FieldMeta:
    name: str
    contrast: str
    shape: Tuple[int, int, int]
    value_min: float
    value_max: float
    description: str
    params: Dict


# =========================================================
# Grid helpers
# =========================================================
def make_normalized_grid(
    shape: Tuple[int, int, int],
    indexing: str = "ij"
) -> Tuple[Array3D, Array3D, Array3D]:
    """
    Create coordinate grids in [-1, 1].
    """
    nx, ny, nz = shape
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
    return np.meshgrid(x, y, z, indexing=indexing)


def normalize_to_range(
    arr: Array3D,
    out_min: float,
    out_max: float,
    eps: float = 1e-8
) -> Array3D:
    amin = float(arr.min())
    amax = float(arr.max())
    scaled = (arr - amin) / (amax - amin + eps)
    return (out_min + scaled * (out_max - out_min)).astype(np.float32)


# =========================================================
# Optional domain masks
# Recommended default for your main study: full_cube
# so pattern semantics are not confounded by occupied volume.
# =========================================================
def make_mask(
    shape: Tuple[int, int, int],
    mask_type: str = "full_cube",
    radius: float = 0.95,
    ellipsoid_radii: Tuple[float, float, float] = (0.95, 0.95, 0.95)
) -> Array3D:
    X, Y, Z = make_normalized_grid(shape)

    if mask_type == "full_cube":
        mask = np.ones(shape, dtype=bool)

    elif mask_type == "sphere":
        R = np.sqrt(X**2 + Y**2 + Z**2)
        mask = R <= radius

    elif mask_type == "ellipsoid":
        rx, ry, rz = ellipsoid_radii
        R = np.sqrt((X / rx) ** 2 + (Y / ry) ** 2 + (Z / rz) ** 2)
        mask = R <= 1.0

    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return mask


def apply_mask(
    field: Array3D,
    mask: Optional[Array3D],
    outside_value: float = 0.0
) -> Array3D:
    if mask is None:
        return field.astype(np.float32)

    out = np.full_like(field, fill_value=outside_value, dtype=np.float32)
    out[mask] = field[mask]
    return out


# =========================================================
# Contrast presets
# =========================================================
def contrast_preset(level: ContrastName) -> Dict[str, float]:
    """
    low contrast:
        subtle but still distinguishable
    high contrast:
        clearly distinguishable
    """
    if level == "low":
        return {
            "low": 0.35,
            "high": 0.65,
            "baseline": 0.35,
            "peak": 0.65,
        }
    elif level == "high":
        return {
            "low": 0.20,
            "high": 0.80,
            "baseline": 0.20,
            "peak": 0.80,
        }
    else:
        raise ValueError(f"Unknown contrast level: {level}")


# =========================================================
# Pattern generators
# =========================================================
def generate_uniform_field(
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    value: float = 0.5,
    mask: Optional[Array3D] = None,
    outside_value: float = 0.0
) -> Array3D:
    """
    P1 Uniform Field
    No spatial variation.
    """
    field = np.full(shape, fill_value=value, dtype=np.float32)
    return apply_mask(field, mask, outside_value=outside_value)


def generate_linear_gradient(
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    direction: DirectionName = "x+",
    low: float = 0.2,
    high: float = 0.8,
    gamma: float = 1.0,
    mask: Optional[Array3D] = None,
    outside_value: float = 0.0
) -> Array3D:
    """
    P2 Linear Gradient
    Monotonic scalar gradient.

    gamma:
        1.0 = linear
        >1  = flatter middle, steeper ends
        <1  = steeper middle, flatter ends

    Recommended for first paper:
        gamma = 1.0
    """
    X, Y, Z = make_normalized_grid(shape)

    if direction == "x+":
        t = (X + 1.0) / 2.0
    elif direction == "x-":
        t = 1.0 - (X + 1.0) / 2.0
    elif direction == "y+":
        t = (Y + 1.0) / 2.0
    elif direction == "y-":
        t = 1.0 - (Y + 1.0) / 2.0
    elif direction == "z+":
        t = (Z + 1.0) / 2.0
    elif direction == "z-":
        t = 1.0 - (Z + 1.0) / 2.0
    elif direction == "diag_xy+":
        t = (X + Y + 2.0) / 4.0
    elif direction == "diag_xy-":
        t = 1.0 - (X + Y + 2.0) / 4.0
    else:
        raise ValueError(f"Unknown direction: {direction}")

    t = np.clip(t, 0.0, 1.0)

    if gamma != 1.0:
        t = np.power(t, gamma)

    field = low + (high - low) * t
    return apply_mask(field.astype(np.float32), mask, outside_value=outside_value)


def generate_step_boundary(
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    orientation: DirectionName = "x+",
    low: float = 0.2,
    high: float = 0.8,
    boundary_offset: float = 0.0,
    transition_width: float = 0.0,
    mask: Optional[Array3D] = None,
    outside_value: float = 0.0
) -> Array3D:
    """
    P3 Step Boundary
    Two scalar regions separated by a hard or soft boundary.

    IMPORTANT:
    This keeps the full occupied volume and only changes scalar values.
    It does NOT make half the object empty.
    """
    X, Y, Z = make_normalized_grid(shape)

    if orientation == "x+":
        coord = X - boundary_offset
    elif orientation == "x-":
        coord = -X - boundary_offset
    elif orientation == "y+":
        coord = Y - boundary_offset
    elif orientation == "y-":
        coord = -Y - boundary_offset
    elif orientation == "z+":
        coord = Z - boundary_offset
    elif orientation == "z-":
        coord = -Z - boundary_offset
    elif orientation == "diag_xy+":
        coord = (X + Y) / np.sqrt(2.0) - boundary_offset
    elif orientation == "diag_xy-":
        coord = (X - Y) / np.sqrt(2.0) - boundary_offset
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    if transition_width <= 0.0:
        field = np.where(coord < 0.0, low, high).astype(np.float32)
    else:
        s = 1.0 / (1.0 + np.exp(-coord / transition_width))
        field = (low + (high - low) * s).astype(np.float32)

    return apply_mask(field, mask, outside_value=outside_value)


def generate_localized_peak(
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    baseline: float = 0.2,
    peak: float = 0.8,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    sigma: Tuple[float, float, float] = (0.35, 0.35, 0.35),
    normalize_peak: bool = True,
    mask: Optional[Array3D] = None,
    outside_value: float = 0.0
) -> Array3D:
    """
    P4 Localized Peak
    Gaussian-like hotspot on a baseline field.

    sigma:
        Smaller -> sharper peak
        Larger  -> broader peak

    Recommended first value:
        sigma = (0.35, 0.35, 0.35)
    """
    X, Y, Z = make_normalized_grid(shape)
    cx, cy, cz = center
    sx, sy, sz = sigma

    gauss = np.exp(
        -(
            ((X - cx) ** 2) / (2 * sx**2) +
            ((Y - cy) ** 2) / (2 * sy**2) +
            ((Z - cz) ** 2) / (2 * sz**2)
        )
    ).astype(np.float32)

    if normalize_peak:
        gauss = normalize_to_range(gauss, 0.0, 1.0)

    field = baseline + (peak - baseline) * gauss
    return apply_mask(field.astype(np.float32), mask, outside_value=outside_value)


# =========================================================
# Raw export helpers for Squishicalization
# =========================================================
def float_field_to_uint16_raw_range(
    field: Array3D,
    in_range: Tuple[float, float] = (0.0, 1.0),
    out_range: Tuple[int, int] = (0, SQUISH_UINT16_MAX),
    clip: bool = True
) -> np.ndarray:
    """
    Map float field from [0,1] to uint16 [0,65535].
    """
    in_min, in_max = in_range
    out_min, out_max = out_range

    arr = field.astype(np.float32)

    if clip:
        arr = np.clip(arr, in_min, in_max)

    arr = (arr - in_min) / (in_max - in_min + 1e-8)
    arr = out_min + arr * (out_max - out_min)
    arr = np.rint(arr).astype(np.uint16)
    return arr


def save_raw_for_squishicalization(
    field_uint16: np.ndarray,
    out_path: str | Path
) -> None:
    """
    Save raw exactly in the format expected by your Squishicalization loader:
        dtype = uint16
        reshape(..., order='F')

    Therefore we flatten with order='F' before writing.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(field_uint16, dtype=np.uint16)
    arr.ravel(order="F").tofile(out_path)


def load_raw_like_squishicalization(
    path: str | Path,
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    dtype: np.dtype = SQUISH_DEFAULT_DTYPE
) -> np.ndarray:
    """
    Optional integrity check:
    load raw exactly as your Squishicalization code does.
    """
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)

    expected_size = np.prod(shape)
    if data.size != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size} values, got {data.size}"
        )

    return data.reshape(shape, order="F")


# =========================================================
# Suite generation
# =========================================================
def generate_four_pattern_suite(
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    contrast: ContrastName = "high",
    gradient_direction: DirectionName = "x+",
    boundary_orientation: DirectionName = "x+",
    peak_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    peak_sigma: Tuple[float, float, float] = (0.35, 0.35, 0.35),
    mask_type: str = "full_cube"
) -> Dict[str, Tuple[Array3D, FieldMeta]]:
    vals = contrast_preset(contrast)
    mask = make_mask(shape, mask_type=mask_type)
    uniform_value = 0.5 * (vals["low"] + vals["high"])

    suite: Dict[str, Tuple[Array3D, FieldMeta]] = {}

    # P1 Uniform
    f1 = generate_uniform_field(
        shape=shape,
        value=uniform_value,
        mask=mask
    )
    m1 = FieldMeta(
        name="P1_Uniform_Field",
        contrast=contrast,
        shape=shape,
        value_min=float(f1[mask].min()),
        value_max=float(f1[mask].max()),
        description="No spatial variation; baseline/control condition.",
        params={
            "value": uniform_value,
            "contrast": contrast,
            "mask_type": mask_type,
        }
    )
    suite[m1.name] = (f1, m1)

    # P2 Gradient
    f2 = generate_linear_gradient(
        shape=shape,
        direction=gradient_direction,
        low=vals["low"],
        high=vals["high"],
        gamma=1.0,
        mask=mask
    )
    m2 = FieldMeta(
        name="P2_Linear_Gradient",
        contrast=contrast,
        shape=shape,
        value_min=float(f2[mask].min()),
        value_max=float(f2[mask].max()),
        description="Continuous scalar gradient.",
        params={
            "direction": gradient_direction,
            "low": vals["low"],
            "high": vals["high"],
            "gamma": 1.0,
            "contrast": contrast,
            "mask_type": mask_type,
        }
    )
    suite[m2.name] = (f2, m2)

    # P3 Step boundary
    f3 = generate_step_boundary(
        shape=shape,
        orientation=boundary_orientation,
        low=vals["low"],
        high=vals["high"],
        boundary_offset=0.0,
        transition_width=0.0,
        mask=mask
    )
    m3 = FieldMeta(
        name="P3_Step_Boundary",
        contrast=contrast,
        shape=shape,
        value_min=float(f3[mask].min()),
        value_max=float(f3[mask].max()),
        description="Discrete two-region field with abrupt boundary.",
        params={
            "orientation": boundary_orientation,
            "low": vals["low"],
            "high": vals["high"],
            "boundary_offset": 0.0,
            "transition_width": 0.0,
            "contrast": contrast,
            "mask_type": mask_type,
        }
    )
    suite[m3.name] = (f3, m3)

    # P4 Localized peak
    f4 = generate_localized_peak(
        shape=shape,
        baseline=vals["baseline"],
        peak=vals["peak"],
        center=peak_center,
        sigma=peak_sigma,
        normalize_peak=True,
        mask=mask
    )
    m4 = FieldMeta(
        name="P4_Localized_Peak",
        contrast=contrast,
        shape=shape,
        value_min=float(f4[mask].min()),
        value_max=float(f4[mask].max()),
        description="Localized Gaussian-like hotspot on a baseline field.",
        params={
            "baseline": vals["baseline"],
            "peak": vals["peak"],
            "center": peak_center,
            "sigma": peak_sigma,
            "contrast": contrast,
            "mask_type": mask_type,
        }
    )
    suite[m4.name] = (f4, m4)

    return suite


# =========================================================
# Save helpers
# =========================================================
def save_suite(
    suite: Dict[str, Tuple[Array3D, FieldMeta]],
    out_dir: str | Path,
    save_npy: bool = True,
    save_json: bool = True,
    save_raw: bool = True,
    verify_raw_reload: bool = True
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for base_name, (field, meta) in suite.items():
        tagged_name = f"{base_name}_{meta.contrast}"

        if save_npy:
            np.save(out_dir / f"{tagged_name}.npy", field)

        if save_json:
            with open(out_dir / f"{tagged_name}.json", "w", encoding="utf-8") as f:
                json.dump(asdict(meta), f, indent=2, ensure_ascii=False)

        if save_raw:
            raw_uint16 = float_field_to_uint16_raw_range(field)
            raw_path = out_dir / f"{tagged_name}.raw"
            save_raw_for_squishicalization(raw_uint16, raw_path)

            if verify_raw_reload:
                reloaded = load_raw_like_squishicalization(
                    raw_path,
                    shape=meta.shape,
                    dtype=np.uint16
                )
                if reloaded.shape != field.shape:
                    raise RuntimeError(
                        f"Raw reload shape mismatch for {tagged_name}: "
                        f"{reloaded.shape} vs {field.shape}"
                    )


# =========================================================
# Main export wrappers
# =========================================================
def export_one_contrast_dataset(
    out_dir: str | Path,
    contrast: ContrastName,
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    gradient_direction: DirectionName = "x+",
    boundary_orientation: DirectionName = "x+",
    peak_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    peak_sigma: Tuple[float, float, float] = (0.35, 0.35, 0.35),
    mask_type: str = "full_cube"
) -> Dict[str, Tuple[Array3D, FieldMeta]]:
    suite = generate_four_pattern_suite(
        shape=shape,
        contrast=contrast,
        gradient_direction=gradient_direction,
        boundary_orientation=boundary_orientation,
        peak_center=peak_center,
        peak_sigma=peak_sigma,
        mask_type=mask_type
    )

    save_suite(
        suite=suite,
        out_dir=out_dir,
        save_npy=True,
        save_json=True,
        save_raw=True,
        verify_raw_reload=True
    )
    return suite


def export_both_contrast_datasets(
    out_root: str | Path = "pseudo_data_squish_ready",
    shape: Tuple[int, int, int] = SQUISH_DEFAULT_SHAPE,
    gradient_direction: DirectionName = "x+",
    boundary_orientation: DirectionName = "x+",
    peak_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    peak_sigma: Tuple[float, float, float] = (0.35, 0.35, 0.35),
    mask_type: str = "full_cube"
) -> Dict[str, Dict[str, Tuple[Array3D, FieldMeta]]]:
    out_root = Path(out_root)
    print(f"Path: {out_root}")
    suite_high = export_one_contrast_dataset(
        out_dir=out_root / "high",
        contrast="high",
        shape=shape,
        gradient_direction=gradient_direction,
        boundary_orientation=boundary_orientation,
        peak_center=peak_center,
        peak_sigma=peak_sigma,
        mask_type=mask_type
    )

    suite_low = export_one_contrast_dataset(
        out_dir=out_root / "low",
        contrast="low",
        shape=shape,
        gradient_direction=gradient_direction,
        boundary_orientation=boundary_orientation,
        peak_center=peak_center,
        peak_sigma=peak_sigma,
        mask_type=mask_type
    )

    return {
        "high": suite_high,
        "low": suite_low,
    }


# =========================================================
# Example run
# =========================================================
if __name__ == "__main__":
    suites = export_both_contrast_datasets(
        out_root="/Users/tanleyu/Desktop/Austria",
        shape=(100, 100, 100),
        gradient_direction="x+",
        boundary_orientation="x+",
        peak_center=(0.0, 0.0, 0.0),
        peak_sigma=(0.35, 0.35, 0.35),
        mask_type="full_cube"
    )

    print("Export complete.")
    print("Saved BOTH high and low contrast datasets.")
    print("Folders:")
    print("  pseudo_data_squish_ready/high/")
    print("  pseudo_data_squish_ready/low/")
    print()
    print("Each folder contains:")
    print("  - .npy")
    print("  - .json")
    print("  - .raw (Squishicalization-ready)")