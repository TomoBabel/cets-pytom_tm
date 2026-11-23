# -*- coding: utf-8 -*-
"""pytom-match-pick -> CETS, on the real template matching of TS_03 (see
``pytom_tm.tests``)."""

import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple
from unittest import TestCase

import mrcfile
import numpy as np
import yaml

from cets_data_model.models.models import (
    AnnotatedObject,
    AnnotationType,
    PointMatrixSet3D,
    ProbabilityMap3D,
    ProbabilityMapValueType,
    Region,
)
from pytom_tm.converters.coordinates3d import PytomCoordinates3D
from pytom_tm.converters.score_tomogram import PytomScoreTomogram
from pytom_tm.tests import (
    ANGLE_COLUMNS,
    ANGLES_MAP,
    CENTERED_COLUMNS,
    CUT_OFF,
    DEFOCUS,
    DOSE,
    FIRST_PARTICLES,
    GRID,
    JOB,
    MASK,
    N_PARTICLES,
    N_ROTATIONS,
    PIXEL_SIZE,
    SCORES_MAP,
    STAR,
    TEMPLATE,
    TEMPLATE_GRID,
    TLT,
    TOMO_ID,
    TOMO_NAME,
    TOMOGRAM,
    columns,
    job_dict,
    select_rows,
    star_table,
    to_voxel_flavour,
    write_job_copy,
    write_table,
)
from pytom_tm.utils.job import find_job, read_job
from pytom_tm.utils.utils import (
    read_star_table,
    relion_euler_to_matrix,
    write_coordinates_yaml,
)


def rot_z(angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rot_x(angle: float) -> np.ndarray:
    a = np.deg2rad(angle)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def grid(path: Path) -> Tuple[Tuple[int, int, int], float]:
    """Dimensions (x, y, z) and voxel size of an mrc file."""
    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        header = mrc.header
        return (int(header.nx), int(header.ny), int(header.nz)), float(mrc.voxel_size.x)


def first_voxels() -> np.ndarray:
    return np.array([voxel for voxel, _, _, _ in FIRST_PARTICLES], dtype=float)


class CetsPytomBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_pytom_tm"))
        self.star = self.test_dir / f"{TOMO_NAME}_particles.star"
        self.job_file = self.test_dir / f"{TOMO_NAME}_job.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class CetsPytomDataTest(CetsPytomBaseTest):
    """The files of the job are consistent with each other and with its inputs."""

    def test_grids(self) -> None:
        for mrc_file in (TOMOGRAM, SCORES_MAP, ANGLES_MAP):
            with self.subTest(mrc_file=mrc_file.name):
                self.assertEqual(grid(mrc_file)[0], GRID)
                self.assertAlmostEqual(grid(mrc_file)[1], PIXEL_SIZE, places=4)
        for mrc_file in (TEMPLATE, MASK):
            with self.subTest(mrc_file=mrc_file.name):
                self.assertEqual(grid(mrc_file), (TEMPLATE_GRID, grid(TOMOGRAM)[1]))

    def test_job_matches_its_inputs(self) -> None:
        """The tilt-series metadata of the job are the ones of the input files"""
        job = job_dict()
        self.assertEqual(job["tomo_id"], TOMO_NAME)
        self.assertEqual(Path(job["tomogram"]).name, TOMOGRAM.name)
        self.assertEqual(Path(job["template"]).name, TEMPLATE.name)
        self.assertEqual(Path(job["mask"]).name, MASK.name)
        self.assertEqual(job["template_shape"], list(TEMPLATE_GRID))
        self.assertEqual(job["n_rotations"], N_ROTATIONS)
        ts = job["ts_metadata"]
        tilt_angles = np.loadtxt(TLT)
        np.testing.assert_allclose(ts["tilt_angles"], tilt_angles)
        np.testing.assert_allclose(ts["dose_accumulation"], np.loadtxt(DOSE))
        # IMOD defocus file: a header line, then defocus 1 and 2 (nm) of each tilt, in the
        # order of the tilt angles (its angle column has the same angles, but in the
        # order of acquisition)
        defocus = np.loadtxt(DEFOCUS, skiprows=1)
        np.testing.assert_allclose(np.sort(defocus[:, 2]), tilt_angles)
        np.testing.assert_allclose(
            [ctf["defocus"] for ctf in ts["ctf_data"]],
            defocus[:, 4:6].mean(axis=1) * 1e-9,
            rtol=1e-4,
        )

    def test_particles_match_the_maps(self) -> None:
        """Each particle is at a voxel whose score is its rlnLCCmax, above the cut-off, and
        the index of the angles map identifies its orientation"""
        table = star_table()
        self.assertEqual(len(table["rlnLCCmax"]), N_PARTICLES)
        voxels = columns(table, CENTERED_COLUMNS) / PIXEL_SIZE + np.array(GRID) / 2
        np.testing.assert_allclose(voxels, np.round(voxels), atol=1e-6)
        x, y, z = np.round(voxels).astype(int).T
        scores = columns(table, ["rlnLCCmax"])[:, 0]
        np.testing.assert_allclose(mrcfile.read(SCORES_MAP)[z, y, x], scores, atol=1e-6)
        self.assertTrue(np.all(scores >= CUT_OFF))
        indices = mrcfile.read(ANGLES_MAP)[z, y, x].astype(int)
        self.assertTrue(np.all((indices >= 0) & (indices < N_ROTATIONS)))
        orientations: Dict[int, Set[Tuple[float, ...]]] = defaultdict(set)
        for index, angles in zip(indices, columns(table, ANGLE_COLUMNS)):
            orientations[index].add(tuple(np.round(angles, 4)))
        self.assertTrue(all(len(angles) == 1 for angles in orientations.values()))
        self.assertEqual(
            len({angles for group in orientations.values() for angles in group}),
            len(orientations),
        )
        for i, (voxel, angles, score, index) in enumerate(FIRST_PARTICLES):
            self.assertEqual(tuple(np.round(voxels[i]).astype(int)), voxel)
            self.assertEqual(orientations[index], {tuple(np.round(angles, 4))})
            self.assertEqual(scores[i], score)


class CetsPytomUtilsTest(CetsPytomBaseTest):
    def test_read_star(self) -> None:
        table = read_star_table(STAR, "particles")
        self.assertEqual(len(table), 11)
        self.assertEqual(table["rlnTomoName"], [TOMO_NAME] * N_PARTICLES)
        self.assertEqual(table["rlnLCCmax"][0], f"{FIRST_PARTICLES[0][2]:.6f}")

    def test_reject_star_without_particles_table(self) -> None:
        self.star.write_text("data_optics\n\nloop_\n_rlnVoltage #1\n300\n")
        with self.assertRaises(ValueError):
            read_star_table(self.star, "particles")
        self.assertEqual(read_star_table(self.star, "optics"), {"rlnVoltage": ["300"]})

    def test_relion_euler(self) -> None:
        """The matrix rotates the template into the particle: the transpose of the RELION
        (intrinsic ZYZ, particle to reference) one"""
        np.testing.assert_allclose(relion_euler_to_matrix(0, 0, 0), np.eye(3))
        np.testing.assert_allclose(
            relion_euler_to_matrix(30, 0, 45), rot_z(-75), atol=1e-12
        )
        # A rotation of the reference of 90 degrees around y is tilt = -90 in RELION
        np.testing.assert_allclose(
            relion_euler_to_matrix(0, -90, 0) @ [0, 0, 1], [1, 0, 0], atol=1e-12
        )
        for angles in columns(star_table(), ANGLE_COLUMNS):
            matrix = relion_euler_to_matrix(*angles)
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix), 1.0)

    def test_angles_match_pytom_rotation(self) -> None:
        """Third particle: its index in the angles map (47) is the rotation (z1, x, z2) =
        (22.5, 90, 330) degrees of the pytom-match-pick list of rotations, the one applied to
        the template: R = Rz(z2) @ Rx(x) @ Rz(z1)"""
        _, (rot, tilt, psi), _, index = FIRST_PARTICLES[2]
        self.assertEqual(index, 47)
        z1, x, z2 = 22.5, 90.0, 330.0
        np.testing.assert_allclose(
            relion_euler_to_matrix(rot, tilt, psi),
            rot_z(z2) @ rot_x(x) @ rot_z(z1),
            atol=1e-12,
        )


class CetsPytomJobTest(CetsPytomBaseTest):
    def test_job(self) -> None:
        job = read_job(JOB)
        self.assertEqual(job.tomo_name, TOMO_NAME)
        self.assertEqual(job.voxel_size, PIXEL_SIZE)
        self.assertEqual(job.tomo_shape, GRID)
        self.assertIsNone(job.relion_binning)
        self.assertIsNone(job.relion_ts_pixel_size)
        # Next to the job file
        self.assertEqual(job.scores_map, SCORES_MAP)

    def test_find_job_and_scores_map_by_name(self) -> None:
        """The job and the scores map of <tomo>_particles.star are <tomo>_job.json and
        <tomo>_scores.mrc next to it"""
        job = find_job(STAR)
        assert job is not None
        self.assertEqual(job.job_file, JOB)
        self.assertEqual(job.scores_map, SCORES_MAP)
        # Elsewhere, without them
        shutil.copy(STAR, self.star)
        self.assertIsNone(find_job(self.star))
        write_job_copy(self.job_file, {"output_dir": str(self.test_dir / "none")})
        job = find_job(self.star)
        assert job is not None
        self.assertIsNone(job.scores_map)
        # Not named as a particles star file
        other = shutil.copy(STAR, self.test_dir / "particles.star")
        self.assertIsNone(find_job(other))

    def test_paths_relative_to_the_project(self) -> None:
        """The paths recorded in the job (absolute on the machine where it ran) may be
        relative to the directory pytom-match-pick was run in: they are resolved against
        the ancestors of the job file"""
        project = self.test_dir / "project"
        (project / "Runs/tm").mkdir(parents=True)
        (project / "inputs").symlink_to(TOMOGRAM.parent)
        (project / "Runs/tm" / SCORES_MAP.name).symlink_to(SCORES_MAP)
        extract_dir = project / "Runs/extract/extra"
        extract_dir.mkdir(parents=True)
        job = read_job(
            write_job_copy(
                extract_dir / JOB.name,
                {
                    "output_dir": "Runs/tm",
                    "tomogram": f"inputs/{TOMOGRAM.name}",
                    "template": f"inputs/{TEMPLATE.name}",
                },
            )
        )
        self.assertEqual(job.scores_map, project / "Runs/tm" / SCORES_MAP.name)
        self.assertEqual(job.tomogram, project / "inputs" / TOMOGRAM.name)
        self.assertEqual(job.template, project / "inputs" / TEMPLATE.name)
        # Not found
        job = read_job(
            write_job_copy(self.job_file, {"output_dir": "x", "tomogram": "y.mrc"})
        )
        self.assertIsNone(job.scores_map)
        self.assertIsNone(job.tomogram)

    def test_relion_job(self) -> None:
        job = read_job(
            write_job_copy(
                self.job_file,
                {
                    "ts_metadata": {
                        "binning": 16.0,
                        "tilt_series_pixel_size": 1.35,
                        "__class__": "RelionTiltSeriesMetaData",
                    }
                },
            )
        )
        self.assertEqual(job.relion_binning, 16.0)
        self.assertEqual(job.relion_ts_pixel_size, 1.35)

    def test_not_a_job_file(self) -> None:
        for field in ("tomo_id", "voxel_size", "tomo_shape"):
            with self.subTest(missing=field):
                with self.assertRaises(ValueError):
                    read_job(write_job_copy(self.job_file, remove=[field]))
        self.job_file.write_text("not json")
        with self.assertRaises(ValueError):
            read_job(self.job_file)
        shutil.copy(STAR, self.star)
        self.assertIsNone(find_job(self.star))


class CetsPytomCoordinatesTest(CetsPytomBaseTest):
    def test_coordinates(self) -> None:
        """--relion5-compat star file: the centred Å positions are converted back to the
        0-based voxels of the tomogram with the size and voxel size of the job"""
        coords = PytomCoordinates3D().pytom_tm_to_cets(STAR, TOMO_ID, job=read_job(JOB))
        assert coords is not None and coords.scores is not None
        assert coords.matrix3D is not None
        self.assertEqual(coords.annotation_type, AnnotationType.point_matrix_set_3D)
        self.assertEqual(coords.id, f"pytom_tm_coords_{TOMO_ID}")
        self.assertEqual(coords.source_tomogram_id, TOMO_ID)
        self.assertEqual(len(coords.origin3D), N_PARTICLES)
        np.testing.assert_allclose(coords.origin3D[:3], first_voxels(), atol=1e-6)
        for i, (_, angles, score, _) in enumerate(FIRST_PARTICLES):
            np.testing.assert_allclose(
                coords.matrix3D[i], relion_euler_to_matrix(*angles), atol=1e-12
            )
            self.assertEqual(coords.scores[i], score)
        # In the order of the star file: descending score
        self.assertEqual(coords.scores, sorted(coords.scores, reverse=True))
        self.assertEqual(
            [cs.name for cs in coords.coordinate_systems],
            [f"{TOMO_ID}_array", f"{TOMO_ID}_physical"],
        )
        self.assertEqual(coords.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3)

    def test_centered_positions_need_the_tomogram_size(self) -> None:
        """Without the job, the size of the tomogram comes from a reference map"""
        converter = PytomCoordinates3D()
        with self.assertRaises(ValueError):
            converter.pytom_tm_to_cets(STAR, TOMO_ID)
        for reference in (TOMOGRAM, SCORES_MAP):
            with self.subTest(reference=reference.name):
                coords = converter.pytom_tm_to_cets(
                    STAR, TOMO_ID, reference_map=reference
                )
                assert coords is not None
                np.testing.assert_allclose(
                    coords.origin3D[:3], first_voxels(), atol=1e-6
                )
                # Voxel size of the star file
                self.assertEqual(
                    coords.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3
                )

    def test_centered_positions_of_relion_job(self) -> None:
        """Jobs of RELION5 tomograms:
        (voxel * binning - (dims * binning / 2 - 1)) * tilt-series pixel size"""
        binning, ts_pixel_size = 16.0, 1.35
        coords = PytomCoordinates3D().pytom_tm_to_cets(STAR, TOMO_ID, job=read_job(JOB))
        assert coords is not None
        voxels = np.array(coords.origin3D)
        table = star_table()
        center = np.array(GRID) * binning / 2 - 1
        centered = (voxels * binning - center) * ts_pixel_size
        for i, label in enumerate(CENTERED_COLUMNS):
            table[label] = [f"{v:.6f}" for v in centered[:, i]]
        write_table(self.star, table)
        job = read_job(
            write_job_copy(
                self.job_file,
                {
                    "ts_metadata": {
                        "binning": binning,
                        "tilt_series_pixel_size": ts_pixel_size,
                        "__class__": "RelionTiltSeriesMetaData",
                    }
                },
            )
        )
        relion = PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID, job=job)
        assert relion is not None
        np.testing.assert_allclose(relion.origin3D, voxels, atol=1e-5)

    def test_voxel_flavour(self) -> None:
        """Default star file: the positions are the 0-based voxels of the tomogram, and the
        voxel size of the star file is used, unless one is given"""
        write_table(self.star, to_voxel_flavour(star_table()))
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        reference = converter.pytom_tm_to_cets(STAR, TOMO_ID, job=read_job(JOB))
        assert reference is not None
        np.testing.assert_allclose(coords.origin3D, reference.origin3D, atol=1e-6)
        self.assertEqual(coords.matrix3D, reference.matrix3D)
        self.assertEqual(coords.scores, reference.scores)
        self.assertEqual(coords.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3)
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID, pixel_size=10.8)
        assert coords is not None
        self.assertEqual(coords.coordinate_transformations[0].scale, [10.8] * 3)
        # The one of the job has priority over the one of the star file
        job = read_job(write_job_copy(self.job_file, {"voxel_size": 10.8}))
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID, job=job)
        assert coords is not None
        self.assertEqual(coords.coordinate_transformations[0].scale, [10.8] * 3)
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID, pixel_size=5.4, job=job)
        assert coords is not None
        self.assertEqual(coords.coordinate_transformations[0].scale, [5.4] * 3)
        # Without any voxel size, only the array coordinate system
        table = to_voxel_flavour(star_table())
        del table["rlnDetectorPixelSize"]
        write_table(self.star, table)
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        self.assertEqual(
            [cs.name for cs in coords.coordinate_systems], [f"{TOMO_ID}_array"]
        )
        self.assertEqual(coords.coordinate_transformations, [])
        with self.assertRaises(ValueError):
            converter.pytom_tm_to_cets(self.star, TOMO_ID, pixel_size=0)

    def test_grid_check(self) -> None:
        """The particles must be inside the grid of the reference map, which must be the
        one of the job"""
        converter = PytomCoordinates3D()
        write_table(self.star, to_voxel_flavour(star_table()))
        with self.assertRaises(ValueError):  # the 24^3 grid of the template
            converter.pytom_tm_to_cets(self.star, TOMO_ID, reference_map=TEMPLATE)
        with self.assertRaises(ValueError):
            converter.pytom_tm_to_cets(
                STAR, TOMO_ID, job=read_job(JOB), reference_map=MASK
            )
        coords = converter.pytom_tm_to_cets(
            STAR, TOMO_ID, job=read_job(JOB), reference_map=SCORES_MAP
        )
        assert coords is not None
        self.assertEqual(len(coords.origin3D), N_PARTICLES)

    def test_several_tomograms(self) -> None:
        """A star file with the particles of TS_03 and of a copy of them named TS_04, as
        pytom_merge_stars.py writes it"""
        table = star_table()
        merged = {label: values + values[:10] for label, values in table.items()}
        merged["rlnTomoName"][N_PARTICLES:] = ["TS_04"] * 10
        merged["rlnCutOff"][N_PARTICLES:] = ["0.3"] * 10
        write_table(self.star, merged)
        converter = PytomCoordinates3D()
        job = read_job(JOB)
        self.assertEqual(converter.get_tomo_names(self.star), [TOMO_NAME, "TS_04"])
        self.assertEqual(converter.get_cut_off(self.star, TOMO_NAME), CUT_OFF)
        self.assertEqual(converter.get_cut_off(self.star, "TS_04"), 0.3)
        with self.assertRaises(ValueError):
            converter.pytom_tm_to_cets(self.star, TOMO_ID, job=job)
        coords = converter.pytom_tm_to_cets(
            self.star, "TS_04", tomo_name="TS_04", job=job
        )
        assert coords is not None
        self.assertEqual(len(coords.origin3D), 10)
        np.testing.assert_allclose(coords.origin3D[:3], first_voxels(), atol=1e-6)
        self.assertIsNone(
            converter.pytom_tm_to_cets(self.star, TOMO_ID, tomo_name="TS_05", job=job)
        )

    def test_no_particles(self) -> None:
        write_table(self.star, select_rows(star_table(), []))
        converter = PytomCoordinates3D()
        self.assertIsNone(converter.pytom_tm_to_cets(self.star, TOMO_ID))
        self.assertIsNone(converter.get_cut_off(self.star))

    def test_reject_other_star_files(self) -> None:
        self.star.write_text("data_particles\n\nloop_\n_rlnVoltage #1\n300\n")
        with self.assertRaises(ValueError):
            PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID)
        table = star_table()
        del table["rlnLCCmax"]
        write_table(self.star, table)
        with self.assertRaises(ValueError):
            PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID, job=read_job(JOB))

    def test_cut_off_and_tomo_names(self) -> None:
        converter = PytomCoordinates3D()
        self.assertEqual(converter.get_cut_off(STAR), CUT_OFF)
        self.assertEqual(converter.get_tomo_names(STAR), [TOMO_NAME])


class CetsPytomScoreTomogramTest(CetsPytomBaseTest):
    def test_scores_map(self) -> None:
        scores = PytomScoreTomogram().pytom_tm_to_cets(
            SCORES_MAP, TOMO_ID, threshold=CUT_OFF
        )
        self.assertEqual(scores.annotation_type, AnnotationType.probability_map_3D)
        self.assertEqual(scores.value_type, ProbabilityMapValueType.score)
        self.assertEqual(scores.threshold, CUT_OFF)
        self.assertEqual((scores.width, scores.height, scores.depth), GRID)
        self.assertEqual(scores.source_tomogram_id, TOMO_ID)
        self.assertEqual(scores.id, f"pytom_tm_scores_{TOMO_ID}")
        self.assertEqual(scores.path, str(SCORES_MAP.resolve()))
        self.assertAlmostEqual(
            scores.coordinate_transformations[0].scale[0], PIXEL_SIZE, places=5
        )
        scores = PytomScoreTomogram().pytom_tm_to_cets(
            SCORES_MAP, TOMO_ID, pixel_size=10.8
        )
        self.assertEqual(scores.coordinate_transformations[0].scale, [10.8] * 3)
        with self.assertRaises(ValueError):
            PytomScoreTomogram().pytom_tm_to_cets(SCORES_MAP, TOMO_ID, pixel_size=-1)

    def test_linked_coordinates_in_region(self) -> None:
        """The coordinates refer to the scores map they were extracted from, and both fit
        in a Region sharing the frame of the tomogram"""
        job = find_job(STAR)
        assert job is not None and job.scores_map is not None
        converter = PytomCoordinates3D()
        scores = PytomScoreTomogram().pytom_tm_to_cets(
            job.scores_map, TOMO_ID, threshold=converter.get_cut_off(STAR)
        )
        coords = converter.pytom_tm_to_cets(
            STAR,
            TOMO_ID,
            job=job,
            reference_map=job.scores_map,
            annotated_object=AnnotatedObject(feature_type="other", name="ribosome"),
            score_annotation_id=scores.id,
        )
        assert coords is not None
        self.assertEqual(coords.derived_from_annotation_id, scores.id)
        self.assertEqual(scores.threshold, CUT_OFF)
        region = Region(id="region", annotations=[scores, coords])
        reconstructed = Region.model_validate_json(region.model_dump_json())
        self.assertIsInstance(reconstructed.annotations[0], ProbabilityMap3D)
        self.assertIsInstance(reconstructed.annotations[1], PointMatrixSet3D)
        self.assertEqual(reconstructed.annotations[1], coords)
        self.assertEqual(
            [cs.name for cs in scores.coordinate_systems],
            [cs.name for cs in coords.coordinate_systems],
        )
        self.assertEqual(
            scores.coordinate_transformations, coords.coordinate_transformations
        )

    def test_yaml_roundtrip(self) -> None:
        coords = PytomCoordinates3D().pytom_tm_to_cets(STAR, TOMO_ID, job=read_job(JOB))
        assert coords is not None
        write_coordinates_yaml(coords, self.test_dir)
        with open(self.test_dir / f"{coords.id}_pytom_tm_to_cets.yaml") as f:
            self.assertEqual(PointMatrixSet3D.model_validate(yaml.safe_load(f)), coords)
