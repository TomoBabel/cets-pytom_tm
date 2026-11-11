import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from cets_data_model.models.models import (
    Axis,
    AxisType,
    AxisUnit,
    CoordinateSystem,
    PointMatrixSet3D,
    ProbabilityMap3D,
    Scale,
)


# The yaml files are named after the annotation id, which already contains the tomogram id
def write_coordinates_yaml(
    coordinates: PointMatrixSet3D, output_directory: Path
) -> None:
    _write_obj_yaml(coordinates, output_directory)


def write_scores_yaml(scores: ProbabilityMap3D, output_directory: Path) -> None:
    _write_obj_yaml(scores, output_directory)


def _write_obj_yaml(
    cets_obj: PointMatrixSet3D | ProbabilityMap3D, output_directory: Path
) -> None:
    yaml_file = output_directory / f"{cets_obj.id}_pytom_tm_to_cets.yaml"
    metadata_dict = cets_obj.model_dump(mode="json")
    with open(yaml_file, "w") as f:
        yaml.dump(metadata_dict, f, sort_keys=False, explicit_start=True)
    print(f"yaml file successfully written! -> {yaml_file}")


def read_star_table(star_file: os.PathLike, block: str) -> Dict[str, List[str]]:
    """Reads a table (loop) of a star file.

    :param block: name of the data block (without the ``data_`` prefix).

    :returns: the values of each column (label without the leading underscore), as strings,
    in the order of the file. Empty lists if the table has no rows.
    """
    labels: List[str] = []
    rows: List[List[str]] = []
    in_block = in_loop = False
    with open(star_file) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                if in_block:
                    break  # next block
                in_block = line == f"data_{block}"
            elif not in_block:
                continue
            elif line == "loop_":
                in_loop = True
            elif in_loop and line.startswith("_"):
                labels.append(line.split()[0][1:])
            elif in_loop:
                values = line.split()
                if len(values) != len(labels):
                    raise ValueError(
                        f"{star_file}: found a row with {len(values)} values, but the "
                        f"table has {len(labels)} columns"
                    )
                rows.append(values)
    if not labels:
        raise ValueError(f"{star_file} does not contain a table in data_{block}")
    return {label: [row[i] for row in rows] for i, label in enumerate(labels)}


def relion_euler_to_matrix(rot: float, tilt: float, psi: float) -> np.ndarray:
    """Rotation matrix that rotates the template into the orientation of the particle in the
    tomogram, from the RELION Euler angles (degrees) of the particles star file.

    RELION angles are intrinsic ZYZ and describe the opposite rotation (the one aligning the
    particle to the reference), so the matrix is the transpose of the intrinsic ZYZ one:
    R = (Rz(rot) @ Ry(tilt) @ Rz(psi)).T = Rz(-psi) @ Ry(-tilt) @ Rz(-rot).

    For the rotation (z1, x, z2) of the pytom-match-pick list of rotations it is
    R = Rz(z2) @ Rx(x) @ Rz(z1), the rotation applied to the template in the template
    matching, and the same convention as the matrices of cets-gapstop and cets-dynamo.
    """
    return _rot_z(-psi) @ _rot_y(-tilt) @ _rot_z(-rot)


def _rot_z(angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _rot_y(angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def gen_coordinate_systems(
    name: str, ndim: int = 2
) -> Tuple[CoordinateSystem, CoordinateSystem]:
    """Builds the (array, physical) coordinate-system pair for an ``ndim`` image/frame.

    Names follow the CETS proposal convention ``{name}_array`` / ``{name}_physical``. The array
    system is pixel/array coords (unit ``pixel``); the physical system is in Å. ``ndim`` is 2 for
    2-D images (x, y) and 3 for volumes (x, y, z).
    """
    axes = ("x", "y", "z")[:ndim]
    array_cs = CoordinateSystem(
        name=f"{name}_array",
        axes=[
            Axis(name=a, axis_type=AxisType.array, axis_unit=AxisUnit.pixel)
            for a in axes
        ],
    )
    physical_cs = CoordinateSystem(
        name=f"{name}_physical",
        axes=[
            Axis(name=a, axis_type=AxisType.space, axis_unit=AxisUnit.angstrom)
            for a in axes
        ],
    )
    return array_cs, physical_cs


def gen_array_to_physical(
    pixel_size: float, array_cs_name: str, physical_cs_name: str, ndim: int = 2
) -> Scale:
    """The single canonical ``array_to_physical`` transformation for an image: a Scale mapping
    pixel/array coordinates to physical (Å) coordinates by the (isotropic) pixel/voxel size.
    The spec requires exactly one such transformation per image."""
    return Scale(
        scale=[pixel_size] * ndim,
        name="array_to_physical",
        input=array_cs_name,
        output=physical_cs_name,
    )


def validate_file(filename: os.PathLike, expected_ext: str) -> Path:
    if filename is None:
        raise ValueError("The introduced file cannot be None")
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{filename} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {filename}: {p}")

    ext = p.suffix
    if ext != expected_ext:
        raise ValueError(f"Invalid file extension '{ext}'. Expected: {expected_ext}")
    return p
