import os
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import mrcfile
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


def write_star_table(
    star_file: os.PathLike, block: str, table: Mapping[str, Sequence[str]]
) -> Path:
    """Writes a table (loop) of a star file, with the layout of the starfile Python package
    used by pytom-match-pick: one tab-separated row per element of the columns.

    :param block: name of the data block (without the ``data_`` prefix).
    :param table: the values of each column (label without the leading underscore), already
    formatted as strings.
    """
    labels = list(table)
    n_rows = {len(values) for values in table.values()}
    if len(n_rows) != 1:
        raise ValueError(
            f"The columns of a star table must have the same number of values: "
            f"{ {label: len(values) for label, values in table.items()} }"
        )
    lines = ["# Created by cets-pytom_tm", "", "", f"data_{block}", "", "loop_"]
    lines += [f"_{label} #{i + 1}" for i, label in enumerate(labels)]
    lines += ["\t".join(row) for row in zip(*(table[label] for label in labels))]
    star_path = Path(star_file)
    star_path.write_text("\n".join(lines) + "\n")
    return star_path


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


def matrix_to_relion_euler(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of ``relion_euler_to_matrix``: the RELION Euler angles (rot, tilt, psi), in
    degrees, of a rotation matrix, with tilt in [0, 180] and rot and psi in (-180, 180].

    When tilt is 0 or 180 only rot + psi (or rot - psi) is defined, so rot is set to 0: the
    angles may differ from the ones the matrix was built from, but not the rotation.
    """
    # Intrinsic ZYZ matrix of the RELION angles: Rz(rot) @ Ry(tilt) @ Rz(psi)
    zyz = np.asarray(matrix).T
    tilt = np.arccos(np.clip(zyz[2, 2], -1.0, 1.0))
    if abs(np.sin(tilt)) < 1e-8:
        rot = 0.0
        if zyz[2, 2] > 0:
            psi = np.arctan2(zyz[1, 0], zyz[0, 0])
        else:
            psi = np.arctan2(zyz[0, 1], zyz[1, 1])
    else:
        rot = np.arctan2(zyz[1, 2], zyz[0, 2])
        psi = np.arctan2(zyz[2, 1], -zyz[2, 0])
    return float(np.rad2deg(rot)), float(np.rad2deg(tilt)), float(np.rad2deg(psi))


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


def get_voxel_size(
    cets_obj: PointMatrixSet3D | ProbabilityMap3D,
) -> Optional[float]:
    """Voxel size (Å) of a CETS annotation: the scale of its ``array_to_physical``
    transformation. None if it has none."""
    for transformation in cets_obj.coordinate_transformations or []:
        if isinstance(transformation, Scale) and transformation.scale:
            return float(transformation.scale[0])
    return None


def export_mrc(
    mrc_file: os.PathLike,
    out_file: os.PathLike,
    voxel_size: Optional[float] = None,
    copy: bool = False,
) -> Path:
    """Makes an mrc file available with another name: as a symbolic link, unless a copy is
    requested or needed.

    :param voxel_size: voxel size (Å) the file must have in its header. If it is not the one
    of ``mrc_file``, the file is copied and its header updated, instead of linked.
    """
    src = validate_file(mrc_file, expected_ext=Path(mrc_file).suffix)
    dst = Path(out_file).expanduser()
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src:
            return dst
        raise FileExistsError(f"{dst} already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.open(src, header_only=True, permissive=True) as mrc:
        header_voxel_size = float(mrc.voxel_size.x)
    fix_header = voxel_size is not None and not np.isclose(
        header_voxel_size, voxel_size, rtol=1e-4
    )
    if not (copy or fix_header):
        dst.symlink_to(src)
        return dst
    shutil.copyfile(src, dst)
    if fix_header:
        # Memory-mapped, so only the header is rewritten: a file opened with
        # header_only=True in "r+" mode is truncated to its header when closed.
        with mrcfile.mmap(dst, mode="r+", permissive=True) as mrc:
            mrc.voxel_size = voxel_size
    return dst
