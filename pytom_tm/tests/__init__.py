"""Test data: a real pytom-match-pick template matching of the tomogram TS_03.

- ``data/inputs``: the tomogram (``TS_03.mrc``), the template (``Reference.mrc``) and its
  mask (``Mask.mrc``), and the tilt angles (``TS_03.tlt``), accumulated dose
  (``TS_03_dose.txt``) and defocus (``TS_03.defocus``, IMOD format) of its tilt series.
- ``data/outputs``: what pytom-match-pick wrote for it: the job file (``TS_03_job.json``), the
  scores and angles maps (``TS_03_scores.mrc``, ``TS_03_angles.mrc``) and the particles
  extracted with ``--relion5-compat`` (``TS_03_particles.star``).

The other variants the tests need (the default flavour of the star file, a job of a RELION5
tomogram, several tomograms, ...) are derived from these files in a temporary directory.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pytom_tm.utils.utils import read_star_table, write_star_table

DATA_DIR = Path(__file__).parent / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

TOMOGRAM = INPUTS_DIR / "TS_03.mrc"
TEMPLATE = INPUTS_DIR / "Reference.mrc"
MASK = INPUTS_DIR / "Mask.mrc"
TLT = INPUTS_DIR / "TS_03.tlt"
DOSE = INPUTS_DIR / "TS_03_dose.txt"
DEFOCUS = INPUTS_DIR / "TS_03.defocus"
JOB = OUTPUTS_DIR / "TS_03_job.json"
SCORES_MAP = OUTPUTS_DIR / "TS_03_scores.mrc"
ANGLES_MAP = OUTPUTS_DIR / "TS_03_angles.mrc"
STAR = OUTPUTS_DIR / "TS_03_particles.star"

TOMO_NAME = "TS_03"  # Name of the tomogram file, prefix of the files of the job
TOMO_ID = "TS_03"  # Id of the CETS tomogram
GRID = (240, 232, 70)  # x, y, z
PIXEL_SIZE = 21.6
TEMPLATE_GRID = (24, 24, 24)
N_PARTICLES = 401
N_ROTATIONS = 108
CUT_OFF = 0.212666
SEARCH_STD = 0.036318
# Voxel (0-based), RELION angles (rot, tilt, psi), score and index in the angles map of the
# first particles of the star file, read from the files
FIRST_PARTICLES = [
    ((208, 202, 45), (45.0, 109.471221, -180.0), 0.338375, 87),
    ((207, 202, 42), (45.0, 109.471221, -180.0), 0.329429, 87),
    ((36, 201, 47), (67.5, 90.0, -60.0), 0.328555, 47),
]

PARTICLES_BLOCK = "particles"
CENTERED_COLUMNS = [f"rlnCenteredCoordinate{axis}Angst" for axis in "XYZ"]
VOXEL_COLUMNS = [f"rlnCoordinate{axis}" for axis in "XYZ"]
ANGLE_COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]

StarTable = Dict[str, List[str]]


def star_table() -> StarTable:
    """The table of the real particles star file."""
    return read_star_table(STAR, PARTICLES_BLOCK)


def columns(table: StarTable, labels: Sequence[str]) -> np.ndarray:
    """The values of some columns of a star table, as an (N, len(labels)) array."""
    return np.array([table[label] for label in labels], dtype=float).T


def write_table(star_file: Path, table: StarTable) -> Path:
    return write_star_table(star_file, PARTICLES_BLOCK, table)


def to_voxel_flavour(table: StarTable) -> StarTable:
    """The table of a --relion5-compat star file as pytom-match-pick writes it without that
    option: the positions are the 0-based voxels of the tomogram (centred Å / voxel size +
    dims / 2), with the voxel size and the tomogram name in rlnDetectorPixelSize and
    rlnMicrographName."""
    voxels = columns(table, CENTERED_COLUMNS) / PIXEL_SIZE + np.array(GRID) / 2
    voxel_table: StarTable = {
        label: [f"{v:.6f}" for v in voxels[:, i]]
        for i, label in enumerate(VOXEL_COLUMNS)
    }
    renamed = {
        "rlnTomoTiltSeriesPixelSize": "rlnDetectorPixelSize",
        "rlnTomoName": "rlnMicrographName",
    }
    for label, values in table.items():
        if label not in CENTERED_COLUMNS:
            voxel_table[renamed.get(label, label)] = list(values)
    return voxel_table


def select_rows(table: StarTable, rows: Sequence[int]) -> StarTable:
    return {label: [values[i] for i in rows] for label, values in table.items()}


def job_dict() -> Dict[str, Any]:
    with open(JOB) as f:
        return json.load(f)


def write_job_copy(
    job_file: Path, changes: Optional[Dict[str, Any]] = None, remove: Sequence[str] = ()
) -> Path:
    """A copy of the real job file with some fields changed or removed. Its paths are
    absolute paths of the machine where the job ran, so the copies of the tests usually
    replace them."""
    job = job_dict()
    job.update(changes or {})
    for field in remove:
        job.pop(field)
    job_file.write_text(json.dumps(job, indent=4))
    return job_file
