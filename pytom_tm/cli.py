"""Command line of the pytom-match-pick <-> CETS converters (``cets-pytom_tm``).

Examples::

    cets-pytom_tm coordinates to-cets --star TS_03_tomo_particles.star --tomo-id TS_03 \\
        -o TS_03_coords.yaml
    cets-pytom_tm coordinates from-cets -i TS_03_coords.yaml \\
        -o out/TS_03_tomo_particles.star --tomo-name TS_03_tomo
    cets-pytom_tm scores to-cets --scores-map TS_03_tomo_scores.mrc --tomo-id TS_03 \\
        -o TS_03_scores.yaml
    cets-pytom_tm scores from-cets -i TS_03_scores.yaml --output-dir out \\
        --tomo-name TS_03_tomo

``to-cets`` writes a CETS yaml file with one entry per kind of object (``coordinates``,
``scores``), which is what ``from-cets`` reads.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from cets_data_model.models.models import (
    AnnotatedObject,
    PointMatrixSet3D,
    ProbabilityMap3D,
    SegmentedFeatureType,
)
from cets_data_model.utils.cli import (
    FROM_CETS,
    TO_CETS,
    add_data_type,
    add_input_yaml,
    add_output_yaml,
    check_outputs,
    prepare_outputs,
    run_cli,
)
from cets_data_model.utils.yaml_io import load_cets_object, write_cets_yaml
from pytom_tm.converters.coordinates3d import PytomCoordinates3D
from pytom_tm.converters.score_tomogram import PytomScoreTomogram
from pytom_tm.utils.job import PytomJob, find_job, read_job

COORDINATES_KEY = "coordinates"
SCORES_KEY = "scores"


def _annotated_object(name: Optional[str]) -> Optional[AnnotatedObject]:
    if name is None:
        return None
    return AnnotatedObject(feature_type=SegmentedFeatureType.other, name=name)


def _add_annotation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tomo-id",
        required=True,
        help="Id of the CETS tomogram (source_tomogram_id). It names the coordinate "
        "systems, so it must match the id of the CETS tomogram.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Sampling rate (Å/px) of the tomogram.",
    )
    parser.add_argument(
        "--annotated-object",
        help="Name of what the template represents (e.g. ribosome). Template matching "
        "does not record it.",
    )
    parser.add_argument(
        "--annotation-id",
        help="Id of the annotation, unique within its region. Default: "
        "pytom_tm_coords_<tomo id> / pytom_tm_scores_<tomo id>.",
    )


def _tomo_name(
    tomo_name: Optional[str], job: Optional[PytomJob], tomo_id: Optional[str]
) -> str:
    """Name of the tomogram for pytom-match-pick: the given one, else the one of the job,
    else the CETS tomogram id."""
    name = tomo_name or (job.tomo_name if job is not None else None) or tomo_id
    if not name:
        raise ValueError(
            "The annotation has no source tomogram id: provide --tomo-name."
        )
    return name


# ------------------------------------------------------------------------- coordinates
def coordinates_to_cets(args: argparse.Namespace) -> None:
    job: Optional[PytomJob] = None
    if args.job is not None:
        job = read_job(args.job)
    elif not args.no_job:
        # pytom-match-pick writes <tomo name>_job.json next to <tomo name>_particles.star
        job = find_job(args.star)
        if job is not None and args.tomo_name not in (None, job.tomo_name):
            job = None
        if job is not None:
            print(f"Using the job file found next to the star file: {job.job_file}")
    coordinates = PytomCoordinates3D().pytom_tm_to_cets(
        star_file=args.star,
        tomo_id=args.tomo_id,
        tomo_name=args.tomo_name,
        pixel_size=args.pixel_size,
        job=job,
        reference_map=args.reference_map,
        annotated_object=_annotated_object(args.annotated_object),
        score_annotation_id=args.score_annotation_id,
        annotation_id=args.annotation_id,
    )
    if coordinates is None:
        which = f" of tomogram {args.tomo_name}" if args.tomo_name else ""
        raise ValueError(f"{args.star} has no particles{which}.")
    out = write_cets_yaml(args.output, {COORDINATES_KEY: coordinates})
    print(f"CETS coordinates written -> {out}")


def coordinates_from_cets(args: argparse.Namespace) -> None:
    coordinates = load_cets_object(args.input, COORDINATES_KEY, PointMatrixSet3D)
    job = read_job(args.job) if args.job is not None else None
    tomo_name = _tomo_name(args.tomo_name, job, coordinates.source_tomogram_id)
    outputs = prepare_outputs(args.output)
    PytomCoordinates3D().cets_to_pytom_tm(
        coordinates,
        args.output,
        tomo_name=tomo_name,
        pixel_size=args.pixel_size,
        cut_off=args.cut_off,
        search_std=args.search_std,
        relion5_compat=args.relion5_compat,
        tomo_shape=tuple(args.tomo_shape) if args.tomo_shape else None,
        job=job,
    )
    check_outputs(outputs)


def _add_coordinates(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types,
        "coordinates",
        "Picked particles: particles star file <-> PointMatrixSet3D.",
    )

    to_cets = directions.add_parser(
        TO_CETS, help="pytom-match-pick particles star file -> CETS yaml."
    )
    to_cets.add_argument(
        "--star",
        required=True,
        type=Path,
        help="Particles star file written by pytom_extract_candidates.py "
        "(<tomo name>_particles.star).",
    )
    _add_annotation_options(to_cets)
    to_cets.add_argument(
        "--tomo-name",
        help="Tomogram name (rlnMicrographName / rlnTomoName column) of the particles to "
        "convert. Needed if the star file has the particles of several tomograms.",
    )
    to_cets.add_argument(
        "--job",
        type=Path,
        help="Template matching job file (<tomo name>_job.json): gives the size and voxel "
        "size of the tomogram, needed for the --relion5-compat star files. Default: the "
        "job file next to the star file, if found.",
    )
    to_cets.add_argument(
        "--no-job",
        action="store_true",
        help="Do not look for the job file next to the star file.",
    )
    to_cets.add_argument(
        "--reference-map",
        type=Path,
        help="mrc file on the voxel grid the particles were picked in (tomogram or scores "
        "map): gives the size of the tomogram without a job, and checks that the "
        "particles are inside the grid.",
    )
    to_cets.add_argument(
        "--score-annotation-id",
        help="Id of the annotation of the scores map the particles were extracted from "
        "(derived_from_annotation_id).",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=coordinates_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS, help="CETS yaml -> pytom-match-pick particles star file."
    )
    add_input_yaml(from_cets, "CETS yaml with the coordinates.")
    from_cets.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Star file to write (<tomo name>_particles.star).",
    )
    from_cets.add_argument(
        "--tomo-name",
        help="Tomogram name written in the star file (tomogram file name without "
        "extension). Default: the one of --job, else the source tomogram id.",
    )
    from_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Sampling rate (Å/px) of the tomogram. Default: the one of --job, else the "
        "one of the annotation.",
    )
    from_cets.add_argument(
        "--cut-off", type=float, help="Score cut-off of the extraction (rlnCutOff)."
    )
    from_cets.add_argument(
        "--search-std",
        type=float,
        help="Standard deviation of the template matching search (rlnSearchStd).",
    )
    from_cets.add_argument(
        "--relion5-compat",
        action="store_true",
        help="Write the positions in Å relative to the centre of the tomogram, as "
        "pytom-match-pick --relion5-compat. Needs --job, or --tomo-shape and a pixel size.",
    )
    from_cets.add_argument(
        "--tomo-shape",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Size of the tomogram (voxels), for --relion5-compat without --job.",
    )
    from_cets.add_argument(
        "--job",
        type=Path,
        help="Template matching job file of the tomogram (<tomo name>_job.json).",
    )
    from_cets.set_defaults(func=coordinates_from_cets)


# ------------------------------------------------------------------------- scores
def scores_to_cets(args: argparse.Namespace) -> None:
    scores = PytomScoreTomogram().pytom_tm_to_cets(
        scores_map=args.scores_map,
        tomo_id=args.tomo_id,
        pixel_size=args.pixel_size,
        threshold=args.threshold,
        annotated_object=_annotated_object(args.annotated_object),
        annotation_id=args.annotation_id,
    )
    out = write_cets_yaml(args.output, {SCORES_KEY: scores})
    print(f"CETS scores map written -> {out}")


def scores_from_cets(args: argparse.Namespace) -> None:
    scores = load_cets_object(args.input, SCORES_KEY, ProbabilityMap3D)
    out = PytomScoreTomogram().cets_to_pytom_tm(
        scores,
        args.output_dir,
        tomo_name=_tomo_name(args.tomo_name, None, scores.source_tomogram_id),
        copy=args.copy,
    )
    print(f"pytom-match-pick scores map written -> {out}")


def _add_scores(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types,
        "scores",
        "Template matching scores map (<tomo name>_scores.mrc) <-> ProbabilityMap3D.",
    )

    to_cets = directions.add_parser(
        TO_CETS, help="pytom-match-pick scores map (.mrc) -> CETS yaml."
    )
    to_cets.add_argument(
        "--scores-map",
        required=True,
        type=Path,
        help="Scores map (<tomo name>_scores.mrc).",
    )
    _add_annotation_options(to_cets)
    to_cets.add_argument(
        "--threshold",
        type=float,
        help="Score above which the voxels were particle candidates in the particle "
        "extraction (rlnCutOff of the particles star file).",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=scores_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS,
        help="CETS yaml -> pytom-match-pick scores map (<tomo name>_scores.mrc).",
    )
    add_input_yaml(from_cets, "CETS yaml with the scores map.")
    from_cets.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the scores map in.",
    )
    from_cets.add_argument(
        "--tomo-name",
        help="Tomogram name used in the name of the map (tomogram file name without "
        "extension). Default: the source tomogram id.",
    )
    from_cets.add_argument(
        "--copy",
        action="store_true",
        help="Copy the map instead of linking it.",
    )
    from_cets.set_defaults(func=scores_from_cets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cets-pytom_tm",
        description="Converts pytom-match-pick files to and from CETS metadata (yaml).",
    )
    data_types = parser.add_subparsers(
        dest="data_type", metavar="{coordinates,scores}", required=True
    )
    _add_coordinates(data_types)
    _add_scores(data_types)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
