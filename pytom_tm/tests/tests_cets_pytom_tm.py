# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

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
    GRID,
    PIXEL_SIZE,
    REAL_CUT_OFF,
    REAL_FIRST_PARTICLES,
    REAL_GRID,
    REAL_JOB,
    REAL_PIXEL_SIZE,
    REAL_STAR,
    REAL_TOMO_NAME,
    TOMO_ID,
    TOMO_NAME,
    particle,
    write_job,
    write_mrc,
    write_star,
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


class CetsPytomBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_pytom_tm"))
        self.star = self.test_dir / f"{TOMO_NAME}_particles.star"
        self.job_file = self.test_dir / f"{TOMO_NAME}_job.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class CetsPytomUtilsTest(CetsPytomBaseTest):
    def test_read_real_star(self) -> None:
        table = read_star_table(REAL_STAR, "particles")
        self.assertEqual(len(table), 11)
        self.assertEqual(table["rlnTomoName"], [REAL_TOMO_NAME] * 5)
        self.assertEqual(table["rlnLCCmax"][0], "0.282938")

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

    def test_real_angles_match_pytom_rotation(self) -> None:
        """First real particle: pytom-match-pick matched it with the rotation (z1, x, z2) =
        (22.5, 90, 330) degrees of its list (read from its angles map), i.e. R = Rz(z2) @ Rx(x) @ Rz(z1)"""
        rot, tilt, psi = 67.5, 90.0, -60.0
        z1, x, z2 = 22.5, 90.0, 330.0
        matrix = relion_euler_to_matrix(rot, tilt, psi)
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(matrix, rot_z(z2) @ rot_x(x) @ rot_z(z1), atol=1e-12)


class CetsPytomJobTest(CetsPytomBaseTest):
    def test_real_job(self) -> None:
        job = read_job(REAL_JOB)
        self.assertEqual(job.tomo_name, REAL_TOMO_NAME)
        self.assertEqual(job.voxel_size, REAL_PIXEL_SIZE)
        self.assertEqual(job.tomo_shape, REAL_GRID)
        self.assertIsNone(job.relion_binning)
        # Its paths are relative to its Scipion project, so they are not found here
        self.assertIsNone(job.scores_map)
        self.assertIsNone(job.tomogram)

    def test_find_job_and_scores_map_by_name(self) -> None:
        """The job and the scores map of <tomo>_particles.star are <tomo>_job.json and
        <tomo>_scores.mrc next to it"""
        write_star(self.star, [particle((1, 2, 3))])
        self.assertIsNone(find_job(self.star))  # no job file
        write_job(self.job_file)
        job = find_job(self.star)
        assert job is not None
        self.assertIsNone(job.scores_map)
        scores = write_mrc(self.test_dir / f"{TOMO_NAME}_scores.mrc")
        job = find_job(self.star)
        assert job is not None
        self.assertEqual(job.scores_map, scores)

    def test_scores_map_in_output_dir(self) -> None:
        """Scores map in the (relative) output directory recorded in the job file"""
        tm_dir = self.test_dir / "Runs/tm/extra"
        tm_dir.mkdir(parents=True)
        scores = write_mrc(tm_dir / f"{TOMO_NAME}_scores.mrc")
        extract_dir = self.test_dir / "Runs/extract/extra"
        extract_dir.mkdir(parents=True)
        job = read_job(
            write_job(extract_dir / f"{TOMO_NAME}_job.json", output_dir="Runs/tm/extra")
        )
        self.assertEqual(job.scores_map, scores)

    def test_relion_job(self) -> None:
        job = read_job(
            write_job(self.job_file, relion_binning=4.0, relion_ts_pixel_size=1.35)
        )
        self.assertEqual(job.relion_binning, 4.0)
        self.assertEqual(job.relion_ts_pixel_size, 1.35)

    def test_not_a_job_file(self) -> None:
        self.job_file.write_text('{"tomo_id": "TS_01_tomo"}')
        with self.assertRaises(ValueError):
            read_job(self.job_file)
        write_star(self.star, [particle((1, 2, 3))])
        self.assertIsNone(find_job(self.star))


class CetsPytomCoordinatesTest(CetsPytomBaseTest):
    def test_voxel_positions_scores_and_frames(self) -> None:
        """Default star file: positions are the 0-based voxels of the tomogram"""
        write_star(
            self.star,
            [particle((1, 2, 3), score=0.3), particle((10, 20, 15), score=0.2)],
        )
        coords = PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        self.assertEqual(coords.annotation_type, AnnotationType.point_matrix_set_3D)
        self.assertEqual(coords.origin3D, [[1.0, 2.0, 3.0], [10.0, 20.0, 15.0]])
        self.assertEqual(coords.scores, [0.3, 0.2])
        self.assertEqual(coords.source_tomogram_id, TOMO_ID)
        self.assertEqual(coords.id, f"pytom_tm_coords_{TOMO_ID}")
        # The voxel size written in the star file is used
        self.assertEqual(
            [cs.name for cs in coords.coordinate_systems],
            [f"{TOMO_ID}_array", f"{TOMO_ID}_physical"],
        )
        self.assertEqual(coords.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3)
        # A given sampling rate has priority
        coords = PytomCoordinates3D().pytom_tm_to_cets(
            self.star, TOMO_ID, pixel_size=10.0
        )
        assert coords is not None
        self.assertEqual(coords.coordinate_transformations[0].scale, [10.0] * 3)
        with self.assertRaises(ValueError):
            PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID, pixel_size=0)

    def test_centered_positions(self) -> None:
        """--relion5-compat: (voxel - dims / 2) * voxel size, undone with the tomogram size
        of the job or of a reference map"""
        voxels = np.array([[1.0, 2.0, 3.0], [39.0, 29.0, 19.0]])
        centered = (voxels - np.array(GRID) / 2) * PIXEL_SIZE
        write_star(self.star, [particle(p) for p in centered], relion5=True)
        converter = PytomCoordinates3D()
        with self.assertRaises(ValueError):  # the size of the tomogram is unknown
            converter.pytom_tm_to_cets(self.star, TOMO_ID)
        job = read_job(write_job(self.job_file))
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID, job=job)
        assert coords is not None
        np.testing.assert_allclose(coords.origin3D, voxels, atol=1e-9)
        reference = write_mrc(self.test_dir / "tomogram.mrc")
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID, reference_map=reference)
        assert coords is not None
        np.testing.assert_allclose(coords.origin3D, voxels, atol=1e-9)

    def test_centered_positions_of_relion_job(self) -> None:
        """Jobs of RELION5 tomograms:
        (voxel * binning - (dims * binning / 2 - 1)) * tilt-series pixel size"""
        binning, ts_pixel_size = 4.0, 1.35
        voxels = np.array([[1.0, 2.0, 3.0], [39.0, 29.0, 19.0]])
        center = np.array(GRID) * binning / 2 - 1
        centered = (voxels * binning - center) * ts_pixel_size
        write_star(self.star, [particle(p) for p in centered], relion5=True)
        job = read_job(
            write_job(
                self.job_file,
                relion_binning=binning,
                relion_ts_pixel_size=ts_pixel_size,
            )
        )
        coords = PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID, job=job)
        assert coords is not None
        np.testing.assert_allclose(coords.origin3D, voxels, atol=1e-9)

    def test_orientations(self) -> None:
        write_star(self.star, [particle((1, 1, 1), angles=(30, 0, 45))])
        coords = PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        np.testing.assert_allclose(coords.matrix3D[0], rot_z(-75), atol=1e-6)

    def test_grid_check(self) -> None:
        write_star(self.star, [particle((45, 2, 3))])
        small = write_mrc(self.test_dir / "small.mrc")
        with self.assertRaises(ValueError):
            PytomCoordinates3D().pytom_tm_to_cets(
                self.star, TOMO_ID, reference_map=small
            )
        # A reference map of another size than the tomogram of the job
        write_star(self.star, [particle((1, 2, 3))])
        job = read_job(write_job(self.job_file))
        other = write_mrc(self.test_dir / "other.mrc", (8, 8, 8))
        with self.assertRaises(ValueError):
            PytomCoordinates3D().pytom_tm_to_cets(
                self.star, TOMO_ID, job=job, reference_map=other
            )

    def test_several_tomograms(self) -> None:
        write_star(
            self.star,
            [
                particle((1, 1, 1), tomo_name="tomo_a", cut_off=0.1),
                particle((2, 2, 2), tomo_name="tomo_b", cut_off=0.2),
                particle((3, 3, 3), tomo_name="tomo_a", cut_off=0.1),
            ],
        )
        converter = PytomCoordinates3D()
        self.assertEqual(converter.get_tomo_names(self.star), ["tomo_a", "tomo_b"])
        self.assertEqual(converter.get_cut_off(self.star, "tomo_b"), 0.2)
        with self.assertRaises(ValueError):
            converter.pytom_tm_to_cets(self.star, TOMO_ID)
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID, tomo_name="tomo_a")
        assert coords is not None
        self.assertEqual(coords.origin3D, [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        self.assertIsNone(
            converter.pytom_tm_to_cets(self.star, TOMO_ID, tomo_name="tomo_c")
        )

    def test_no_particles(self) -> None:
        write_star(self.star, [])
        self.assertIsNone(PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID))
        self.assertIsNone(PytomCoordinates3D().get_cut_off(self.star))

    def test_reject_other_star_files(self) -> None:
        self.star.write_text("data_particles\n\nloop_\n_rlnVoltage #1\n300\n")
        with self.assertRaises(ValueError):
            PytomCoordinates3D().pytom_tm_to_cets(self.star, TOMO_ID)

    def test_real_star_excerpt(self) -> None:
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(REAL_STAR, "TS_03", job=read_job(REAL_JOB))
        assert coords is not None
        assert coords.scores is not None
        self.assertEqual(len(coords.origin3D), 5)
        for i, (voxel, score) in enumerate(REAL_FIRST_PARTICLES):
            np.testing.assert_allclose(coords.origin3D[i], voxel, atol=1e-6)
            self.assertEqual(coords.scores[i], score)
        self.assertEqual(coords.scores, sorted(coords.scores, reverse=True))
        self.assertEqual(
            coords.coordinate_transformations[0].scale, [REAL_PIXEL_SIZE] * 3
        )
        self.assertEqual(converter.get_cut_off(REAL_STAR), REAL_CUT_OFF)
        self.assertEqual(converter.get_tomo_names(REAL_STAR), [REAL_TOMO_NAME])


class CetsPytomScoreTomogramTest(CetsPytomBaseTest):
    def test_scores_map(self) -> None:
        scores_map = write_mrc(self.test_dir / f"{TOMO_NAME}_scores.mrc")
        scores = PytomScoreTomogram().pytom_tm_to_cets(
            scores_map, TOMO_ID, threshold=0.15
        )
        self.assertEqual(scores.annotation_type, AnnotationType.probability_map_3D)
        self.assertEqual(scores.value_type, ProbabilityMapValueType.score)
        self.assertEqual(scores.threshold, 0.15)
        self.assertEqual((scores.width, scores.height, scores.depth), GRID)
        self.assertEqual(scores.source_tomogram_id, TOMO_ID)
        self.assertEqual(scores.id, f"pytom_tm_scores_{TOMO_ID}")
        self.assertAlmostEqual(
            scores.coordinate_transformations[0].scale[0], PIXEL_SIZE, places=5
        )

    def test_linked_coordinates_in_region(self) -> None:
        """The coordinates refer to the scores map they were extracted from, and both fit
        in a Region sharing the frame of the tomogram"""
        write_star(self.star, [particle((1, 2, 3), cut_off=0.15)])
        write_job(self.job_file)
        write_mrc(self.test_dir / f"{TOMO_NAME}_scores.mrc")
        job = find_job(self.star)
        assert job is not None and job.scores_map is not None
        converter = PytomCoordinates3D()
        scores = PytomScoreTomogram().pytom_tm_to_cets(
            job.scores_map, TOMO_ID, threshold=converter.get_cut_off(self.star)
        )
        coords = converter.pytom_tm_to_cets(
            self.star,
            TOMO_ID,
            job=job,
            reference_map=job.scores_map,
            annotated_object=AnnotatedObject(feature_type="other", name="ribosome"),
            score_annotation_id=scores.id,
        )
        assert coords is not None
        self.assertEqual(coords.derived_from_annotation_id, scores.id)
        region = Region(id="region", annotations=[scores, coords])
        reconstructed = Region.model_validate_json(region.model_dump_json())
        self.assertIsInstance(reconstructed.annotations[0], ProbabilityMap3D)
        self.assertIsInstance(reconstructed.annotations[1], PointMatrixSet3D)
        self.assertEqual(reconstructed.annotations[1], coords)
        self.assertEqual(
            [cs.name for cs in scores.coordinate_systems],
            [cs.name for cs in coords.coordinate_systems],
        )

    def test_yaml_roundtrip(self) -> None:
        coords = PytomCoordinates3D().pytom_tm_to_cets(
            REAL_STAR, "TS_03", job=read_job(REAL_JOB)
        )
        assert coords is not None
        write_coordinates_yaml(coords, self.test_dir)
        with open(self.test_dir / f"{coords.id}_pytom_tm_to_cets.yaml") as f:
            self.assertEqual(PointMatrixSet3D.model_validate(yaml.safe_load(f)), coords)
