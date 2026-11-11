import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from pytom_tm.constants import (
    JOB_BINNING,
    JOB_CLASS,
    JOB_OUTPUT_DIR,
    JOB_SUFFIX,
    JOB_TEMPLATE,
    JOB_TOMO_ID,
    JOB_TOMO_SHAPE,
    JOB_TOMOGRAM,
    JOB_TS_METADATA,
    JOB_TS_PIXEL_SIZE,
    JOB_VOXEL_SIZE,
    PARTICLES_SUFFIX,
    RELION_TS_METADATA_CLASS,
    SCORES_SUFFIX,
)


@dataclass(frozen=True)
class PytomJob:
    """Parameters of a pytom-match-pick template matching job (<tomo_id>_job.json). Only the
    ones relevant for CETS are kept."""

    job_file: Path
    tomo_name: str
    """Name of the tomogram file without extension (tomo_id field), which prefixes the names
    of all the files of the job."""
    voxel_size: float
    """Voxel size (Å) of the tomogram."""
    tomo_shape: Tuple[int, int, int]
    """Size (x, y, z), in voxels, of the tomogram."""
    scores_map: Optional[Path]
    """Scores map of the job (<tomo_id>_scores.mrc). None if it cannot be found."""
    tomogram: Optional[Path]
    """Tomogram that was template-matched. None if the recorded path cannot be found."""
    template: Optional[Path]
    """Template used. None if the recorded path cannot be found."""
    relion_binning: Optional[float] = None
    """Binning of the tomogram, only for the jobs created from RELION5 tomograms."""
    relion_ts_pixel_size: Optional[float] = None
    """Pixel size (Å) of the unbinned tilt-series, only for the jobs created from RELION5
    tomograms."""


def find_job(star_file: os.PathLike) -> Optional[PytomJob]:
    """Finds the template matching job that produced a particles star file.

    pytom-match-pick names all the files of a tomogram after it, and extracts the particles
    in the directory of the job file, so the job of <tomo_id>_particles.star is the file
    <tomo_id>_job.json next to it.

    :returns: the job, or None if there is no such file next to the star file or it cannot be
    read as a job file.
    """
    star_path = Path(star_file)
    if not star_path.name.endswith(PARTICLES_SUFFIX):
        return None
    job_file = star_path.with_name(
        star_path.name[: -len(PARTICLES_SUFFIX)] + JOB_SUFFIX
    )
    if not job_file.is_file():
        return None
    try:
        return read_job(job_file)
    except (OSError, ValueError):
        return None


def read_job(job_file: os.PathLike) -> PytomJob:
    """Reads a pytom-match-pick job file (<tomo_id>_job.json).

    The scores map is looked for next to the job file and, if it is not there, in the output
    directory recorded in it. The recorded paths may be relative to the directory the program
    was run in, so they are resolved against the ancestors of the job file directory.

    :raises ValueError: if the file is not a job file (it is not a json file or it lacks the
    tomogram name, voxel size or shape).
    """
    job_path = Path(job_file)
    try:
        with open(job_path) as f:
            data = json.load(f)
        tomo_name = str(data[JOB_TOMO_ID])
        voxel_size = float(data[JOB_VOXEL_SIZE])
        x, y, z = (int(n) for n in data[JOB_TOMO_SHAPE])
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as e:
        raise ValueError(
            f"{job_path} is not a pytom-match-pick job file: {e!r}"
        ) from None

    scores_name = tomo_name + SCORES_SUFFIX
    scores_map = _existing_file(job_path.parent / scores_name)
    if scores_map is None:
        output_dir = _resolve(data.get(JOB_OUTPUT_DIR), job_path.parent, is_dir=True)
        if output_dir is not None:
            scores_map = _existing_file(output_dir / scores_name)

    ts_metadata = data.get(JOB_TS_METADATA)
    is_relion = (
        isinstance(ts_metadata, dict)
        and ts_metadata.get(JOB_CLASS) == RELION_TS_METADATA_CLASS
    )
    return PytomJob(
        job_file=job_path,
        tomo_name=tomo_name,
        voxel_size=voxel_size,
        tomo_shape=(x, y, z),
        scores_map=scores_map,
        tomogram=_resolve(data.get(JOB_TOMOGRAM), job_path.parent),
        template=_resolve(data.get(JOB_TEMPLATE), job_path.parent),
        relion_binning=_as_float(ts_metadata.get(JOB_BINNING)) if is_relion else None,
        relion_ts_pixel_size=(
            _as_float(ts_metadata.get(JOB_TS_PIXEL_SIZE)) if is_relion else None
        ),
    )


def _existing_file(path: Path) -> Optional[Path]:
    return path if path.is_file() else None


def _resolve(path: Any, start_dir: Path, is_dir: bool = False) -> Optional[Path]:
    """An absolute path if it exists, or the first existing one relative to ``start_dir`` or
    any of its ancestors."""
    if not isinstance(path, str) or not path:
        return None

    def exists(p: Path) -> bool:
        return p.is_dir() if is_dir else p.is_file()

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if exists(candidate) else None
    for directory in (start_dir, *start_dir.parents):
        if exists(directory / candidate):
            return directory / candidate
    return None


def _as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None
