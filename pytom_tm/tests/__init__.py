import json
from pathlib import Path
from typing import Optional, Sequence

import mrcfile
import numpy as np

TEST_DATA_DIR = Path(__file__).parent / "test_data"
# First particles of a real star file written by pytom_extract_candidates.py with
# --relion5-compat (scipion-em-pytom_tm, tomogram TS_03), and its job file with the
# tilt-series metadata cut to two tilts and the paths made relative to the Scipion project.
REAL_STAR = TEST_DATA_DIR / "TS_03_tomo_particles.star"
REAL_JOB = TEST_DATA_DIR / "TS_03_tomo_job.json"
REAL_TOMO_NAME = "TS_03_tomo"
REAL_GRID = (480, 464, 140)
REAL_PIXEL_SIZE = 5.4
REAL_CUT_OFF = 0.145522
# Voxel (0-based) and score of the first real particles: the scores map has these values at
# these voxels
REAL_FIRST_PARTICLES = [
    ((76, 399, 90), 0.282938),
    ((139, 241, 84), 0.280602),
    ((415, 403, 85), 0.260337),
]

TOMO_ID = "TS_01"
TOMO_NAME = "TS_01_tomo"
PIXEL_SIZE = 5.4
GRID = (40, 30, 20)

STAR_COLUMNS = (
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnLCCmax",
    "rlnCutOff",
    "rlnSearchStd",
    "rlnDetectorPixelSize",
    "rlnMicrographName",
)
RELION5_COLUMNS = {
    "rlnCoordinateX": "rlnCenteredCoordinateXAngst",
    "rlnCoordinateY": "rlnCenteredCoordinateYAngst",
    "rlnCoordinateZ": "rlnCenteredCoordinateZAngst",
    "rlnDetectorPixelSize": "rlnTomoTiltSeriesPixelSize",
    "rlnMicrographName": "rlnTomoName",
}


def particle(
    position: Sequence[float],
    angles: Sequence[float] = (0.0, 0.0, 0.0),
    score: float = 0.5,
    tomo_name: str = TOMO_NAME,
    cut_off: float = 0.1,
    pixel_size: float = PIXEL_SIZE,
) -> list[object]:
    """A row of the particles star file: position (x, y, z), RELION angles (rot, tilt, psi),
    score, cut-off, search std, voxel size and tomogram name."""
    return [*position, *angles, score, cut_off, 0.02, pixel_size, tomo_name]


def write_star(
    star_file: Path, rows: Sequence[Sequence[object]], relion5: bool = False
) -> Path:
    """Writes a particles star file as pytom_extract_candidates.py does: by default with the
    positions as given (voxels), with relion5 with the labels of --relion5-compat (the
    positions must then be given in Å relative to the centre of the tomogram)."""
    columns = [RELION5_COLUMNS.get(c, c) if relion5 else c for c in STAR_COLUMNS]
    lines = ["# Created by the starfile Python package", "", "data_particles", ""]
    lines.append("loop_")
    lines += [f"_{column} #{i + 1}" for i, column in enumerate(columns)]
    lines += ["\t".join(str(value) for value in row) for row in rows]
    star_file.write_text("\n".join(lines) + "\n")
    return star_file


def write_job(
    job_file: Path,
    tomo_name: str = TOMO_NAME,
    tomo_shape: Sequence[int] = GRID,
    voxel_size: float = PIXEL_SIZE,
    output_dir: Optional[str] = None,
    relion_binning: Optional[float] = None,
    relion_ts_pixel_size: Optional[float] = None,
) -> Path:
    """Writes the fields of a pytom-match-pick job file that are relevant for CETS."""
    ts_metadata: dict[str, object] = {"__class__": "TiltSeriesMetaData"}
    if relion_binning is not None:
        ts_metadata = {
            "binning": relion_binning,
            "tilt_series_pixel_size": relion_ts_pixel_size,
            "__class__": "RelionTiltSeriesMetaData",
        }
    job = {
        "tomo_id": tomo_name,
        "tomogram": f"{tomo_name}.mrc",
        "template": "template.mrc",
        "output_dir": output_dir or str(job_file.parent),
        "voxel_size": voxel_size,
        "tomo_shape": list(tomo_shape),
        "ts_metadata": ts_metadata,
    }
    job_file.write_text(json.dumps(job, indent=4))
    return job_file


def write_mrc(
    mrc_file: Path, dims_xyz: Sequence[int] = GRID, voxel_size: float = PIXEL_SIZE
) -> Path:
    x, y, z = dims_xyz
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(np.zeros((z, y, x), dtype=np.int8))
        mrc.voxel_size = voxel_size
    return mrc_file
