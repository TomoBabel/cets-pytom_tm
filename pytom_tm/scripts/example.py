"""Converts the particles picked by pytom-match-pick (particles star files) found in a
directory into CETS metadata, together with the scores maps they were extracted from, when
they can be found."""

import argparse
from pathlib import Path
from typing import Dict, Optional

from pytom_tm.constants import PARTICLES_SUFFIX
from pytom_tm.converters.coordinates3d import PytomCoordinates3D
from pytom_tm.converters.score_tomogram import PytomScoreTomogram
from pytom_tm.utils.job import find_job
from pytom_tm.utils.utils import write_coordinates_yaml, write_scores_yaml

### PYTOM-MATCH-PICK TO CETS #######################################################
scratch_dir = Path("/home/jjimenez/CZII/cets_scratch_dir")
# Any directory containing particles star files written by pytom_extract_candidates.py
# (<tomo_name>_particles.star), searched recursively.
pytom_dir = Path(
    "/home/jjimenez/ScipionUserData/projects/czii_re5_extract_subtomos/Runs/004736_ProtPytomExtractCoordinates/"
)
# CETS tomogram id of each tomogram name of the star files (rlnMicrographName / rlnTomoName
# column, the name of the tomogram file without extension). When a name is not in this dict,
# it is used as the CETS tomogram id.
tomo_ids: Dict[str, str] = {"TS_03_tomo": "TS_03", "TS_54_tomo": "TS_54"}


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return number


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--sampling-rate",
    type=positive_float,
    default=None,
    help="Sampling rate (Å/px) of the tomograms the particles were picked in. If not given, "
    "it is taken from the job file, the star file or the header of the scores map.",
)
sampling_rate: Optional[float] = parser.parse_args().sampling_rate

converter = PytomCoordinates3D()
scores_converter = PytomScoreTomogram()
n_converted = 0
for star_file in sorted(pytom_dir.rglob(f"*{PARTICLES_SUFFIX}")):
    # Template matching job that produced the particles and its scores map: found by name,
    # next to the star file (<tomo_name>_job.json and <tomo_name>_scores.mrc). Both optional.
    job = find_job(star_file)
    if job is None:
        print(f"NOTE: no job file found for {star_file}.")
    for tomo_name in converter.get_tomo_names(star_file):
        tomo_id = tomo_ids.get(tomo_name, tomo_name)
        score_annotation_id = None
        scores_map = None
        if job is not None and job.tomo_name == tomo_name:
            scores_map = job.scores_map
        if scores_map is None:
            print(
                f"NOTE: no scores map found for {tomo_name} in {star_file}; converting "
                f"the coordinates alone."
            )
        else:
            # Scores map -> ProbabilityMap3D (Region.annotations)
            scores = scores_converter.pytom_tm_to_cets(
                scores_map=scores_map,
                tomo_id=tomo_id,
                pixel_size=sampling_rate,
                threshold=converter.get_cut_off(star_file, tomo_name),
            )
            write_scores_yaml(scores, scratch_dir)
            score_annotation_id = scores.id
        # Picked coordinates -> PointMatrixSet3D (Region.annotations), derived from the scores
        coordinates = converter.pytom_tm_to_cets(
            star_file=star_file,
            tomo_id=tomo_id,
            tomo_name=tomo_name,
            pixel_size=sampling_rate,
            job=job if job is not None and job.tomo_name == tomo_name else None,
            reference_map=scores_map,
            score_annotation_id=score_annotation_id,
        )
        if coordinates is not None:
            write_coordinates_yaml(coordinates, scratch_dir)
            n_converted += 1

if not n_converted:
    raise SystemExit(
        f"No pytom-match-pick particles star files were found in {pytom_dir}"
    )
