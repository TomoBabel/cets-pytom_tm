# -*- coding: utf-8 -*-
"""CETS -> pytom-match-pick, on the real template matching of TS_03 (see
``pytom_tm.tests``)."""

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import (
    PointMatrixSet3D,
    ProbabilityMap3D,
    ProbabilityMapValueType,
)
from cets_data_model.utils.yaml_io import load_cets_object
from pytom_tm.converters.coordinates3d import PytomCoordinates3D
from pytom_tm.converters.score_tomogram import PytomScoreTomogram
from pytom_tm.tests import (
    ANGLE_COLUMNS,
    CENTERED_COLUMNS,
    COORDS_YAML,
    CUT_OFF,
    DATA_DIR,
    GRID,
    JOB,
    N_PARTICLES,
    PIXEL_SIZE,
    SCORES_MAP,
    SCORES_YAML,
    SEARCH_STD,
    STAR,
    TOMO_ID,
    TOMO_NAME,
    VOXEL_COLUMNS,
    columns,
    star_table,
    to_voxel_flavour,
    write_job_copy,
    write_table,
)
from pytom_tm.utils.job import PytomJob, read_job
from pytom_tm.utils.utils import (
    matrix_to_relion_euler,
    read_star_table,
    relion_euler_to_matrix,
)


def real_coordinates() -> PointMatrixSet3D:
    coords = PytomCoordinates3D().pytom_tm_to_cets(STAR, TOMO_ID, job=read_job(JOB))
    assert coords is not None
    return coords


class CetsToPytomBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_to_pytom_tm"))
        self.out_star = self.test_dir / "out" / f"{TOMO_NAME}_particles.star"
        self.out_star.parent.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class MatrixToRelionEulerTest(TestCase):
    def test_inverse_of_relion_euler_to_matrix(self) -> None:
        """For all the orientations of the real particles"""
        for angles in columns(star_table(), ANGLE_COLUMNS):
            np.testing.assert_allclose(
                matrix_to_relion_euler(relion_euler_to_matrix(*angles)),
                angles,
                atol=1e-9,
            )

    def test_gimbal_lock_keeps_the_rotation(self) -> None:
        """With tilt 0 or 180 the angles are not unique, but the rotation is kept"""
        for angles in [(30.0, 0.0, 45.0), (30.0, 180.0, 45.0), (0.0, 0.0, 0.0)]:
            matrix = relion_euler_to_matrix(*angles)
            new_angles = matrix_to_relion_euler(matrix)
            self.assertEqual(new_angles[0], 0.0)
            np.testing.assert_allclose(
                relion_euler_to_matrix(*new_angles), matrix, atol=1e-12
            )


class CetsToPytomCoordinatesTest(CetsToPytomBaseTest):
    def test_real_star_roundtrip(self) -> None:
        """--relion5-compat star file -> CETS -> star file: the same table"""
        original = star_table()
        out = PytomCoordinates3D().cets_to_pytom_tm(
            real_coordinates(),
            self.out_star,
            TOMO_NAME,
            cut_off=float(original["rlnCutOff"][0]),
            search_std=float(original["rlnSearchStd"][0]),
            relion5_compat=True,
            job=read_job(JOB),
        )
        self.assertEqual(read_star_table(out, "particles"), original)

    def test_voxel_flavour_roundtrip(self) -> None:
        """Default star file -> CETS -> star file: the same table"""
        star = write_table(
            self.test_dir / f"{TOMO_NAME}_particles.star",
            to_voxel_flavour(star_table()),
        )
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(star, TOMO_ID)
        assert coords is not None
        original = read_star_table(star, "particles")
        out = converter.cets_to_pytom_tm(
            coords,
            self.out_star,
            TOMO_NAME,
            cut_off=float(original["rlnCutOff"][0]),
            search_std=float(original["rlnSearchStd"][0]),
        )
        self.assertEqual(read_star_table(out, "particles"), original)

    def test_voxel_flavour_from_relion5(self) -> None:
        """The particles of the --relion5-compat star file written in the default flavour:
        their voxels, angles and scores"""
        out = PytomCoordinates3D().cets_to_pytom_tm(
            real_coordinates(), self.out_star, TOMO_NAME
        )
        table = read_star_table(out, "particles")
        original = star_table()
        voxels = columns(original, CENTERED_COLUMNS) / PIXEL_SIZE + np.array(GRID) / 2
        np.testing.assert_allclose(columns(table, VOXEL_COLUMNS), voxels, atol=1e-6)
        for label in ANGLE_COLUMNS + ["rlnLCCmax"]:
            self.assertEqual(table[label], original[label])
        self.assertEqual(table["rlnDetectorPixelSize"], ["21.600000"] * N_PARTICLES)
        self.assertEqual(table["rlnMicrographName"], [TOMO_NAME] * N_PARTICLES)

    def test_centered_positions_with_tomo_shape(self) -> None:
        """relion5_compat without job: (voxel - dims / 2) * voxel size, with the given
        tomogram size and the voxel size of the annotation"""
        converter = PytomCoordinates3D()
        coords = real_coordinates()
        with self.assertRaises(ValueError):  # the size of the tomogram is unknown
            converter.cets_to_pytom_tm(
                coords, self.out_star, TOMO_NAME, relion5_compat=True
            )
        out = converter.cets_to_pytom_tm(
            coords, self.out_star, TOMO_NAME, relion5_compat=True, tomo_shape=GRID
        )
        table = read_star_table(out, "particles")
        original = star_table()
        for label in CENTERED_COLUMNS + ["rlnTomoTiltSeriesPixelSize", "rlnTomoName"]:
            self.assertEqual(table[label], original[label])

    def test_centered_positions_of_relion_job(self) -> None:
        """Jobs of RELION5 tomograms: written with their binning and tilt-series pixel
        size, and read back with the job, the same voxels"""
        job = read_job(
            write_job_copy(
                self.test_dir / JOB.name,
                {
                    "ts_metadata": {
                        "binning": 16.0,
                        "tilt_series_pixel_size": 1.35,
                        "__class__": "RelionTiltSeriesMetaData",
                    }
                },
            )
        )
        converter = PytomCoordinates3D()
        coords = real_coordinates()
        out = converter.cets_to_pytom_tm(
            coords, self.out_star, TOMO_NAME, relion5_compat=True, job=job
        )
        table = read_star_table(out, "particles")
        center = np.array(GRID) * 16.0 / 2 - 1
        np.testing.assert_allclose(
            columns(table, CENTERED_COLUMNS),
            (np.array(coords.origin3D) * 16.0 - center) * 1.35,
            atol=1e-5,
        )
        back = converter.pytom_tm_to_cets(out, TOMO_ID, job=job)
        assert back is not None
        np.testing.assert_allclose(back.origin3D, coords.origin3D, atol=1e-5)

    def test_pixel_size(self) -> None:
        """Given, else the one of the job, else the one of the annotation"""
        converter = PytomCoordinates3D()
        coords = real_coordinates()
        job = read_job(write_job_copy(self.test_dir / JOB.name, {"voxel_size": 5.4}))
        cases: List[Tuple[Optional[float], Optional[PytomJob], str]] = [
            (None, None, "21.600000"),
            (10.8, None, "10.800000"),
            (None, job, "5.400000"),
            (10.8, job, "10.800000"),
        ]
        for pixel_size, job_or_none, expected in cases:
            with self.subTest(pixel_size=pixel_size, job=job_or_none is not None):
                out = converter.cets_to_pytom_tm(
                    coords,
                    self.out_star,
                    TOMO_NAME,
                    pixel_size=pixel_size,
                    job=job_or_none,
                )
                table = read_star_table(out, "particles")
                self.assertEqual(set(table["rlnDetectorPixelSize"]), {expected})
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(coords, self.out_star, TOMO_NAME, pixel_size=0)

    def test_optional_columns(self) -> None:
        """Cut-off, search std and voxel size are only written when known; angles and
        scores are 0 without orientations and scores"""
        coords = real_coordinates().model_copy(
            update={"coordinate_transformations": [], "matrix3D": None, "scores": None}
        )
        table = read_star_table(
            PytomCoordinates3D().cets_to_pytom_tm(coords, self.out_star, TOMO_NAME),
            "particles",
        )
        self.assertEqual(
            list(table),
            VOXEL_COLUMNS + ANGLE_COLUMNS + ["rlnLCCmax", "rlnMicrographName"],
        )
        self.assertEqual(set(table["rlnAngleTilt"]), {"0.000000"})
        self.assertEqual(set(table["rlnLCCmax"]), {"0.000000"})

    def test_invalid_input(self) -> None:
        converter = PytomCoordinates3D()
        coords = real_coordinates()
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(coords, self.test_dir / "out.em", TOMO_NAME)
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(
                coords.model_copy(update={"scores": [0.1, 0.2]}),
                self.out_star,
                TOMO_NAME,
            )
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(
                coords.model_copy(update={"origin3D": []}), self.out_star, TOMO_NAME
            )


class CetsToPytomScoresTest(CetsToPytomBaseTest):
    def test_linked_with_the_pytom_name(self) -> None:
        converter = PytomScoreTomogram()
        scores = converter.pytom_tm_to_cets(SCORES_MAP, TOMO_ID)
        out = converter.cets_to_pytom_tm(scores, self.test_dir / "job", "tomo_a")
        self.assertEqual(out.name, "tomo_a_scores.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), SCORES_MAP.resolve())
        # Again: already there
        self.assertEqual(
            converter.cets_to_pytom_tm(scores, self.test_dir / "job", "tomo_a"), out
        )
        # Copied on request
        out = converter.cets_to_pytom_tm(
            scores, self.test_dir / "copy", "tomo_a", copy=True
        )
        self.assertFalse(out.is_symlink())
        self.assertEqual(out.read_bytes(), SCORES_MAP.read_bytes())

    def test_copied_if_the_sampling_rate_differs(self) -> None:
        converter = PytomScoreTomogram()
        scores = converter.pytom_tm_to_cets(SCORES_MAP, TOMO_ID, pixel_size=10.8)
        out = converter.cets_to_pytom_tm(scores, self.test_dir / "job", TOMO_NAME)
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 10.8, places=5)
        with mrcfile.open(SCORES_MAP, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), PIXEL_SIZE, places=5)
        # The whole file is copied, not only its header
        self.assertEqual(out.stat().st_size, SCORES_MAP.stat().st_size)
        self.assertEqual(
            mrcfile.read(out).tobytes(), mrcfile.read(SCORES_MAP).tobytes()
        )

    def test_existing_output(self) -> None:
        converter = PytomScoreTomogram()
        scores = converter.pytom_tm_to_cets(SCORES_MAP, TOMO_ID)
        (self.test_dir / f"{TOMO_NAME}_scores.mrc").write_text("other")
        with self.assertRaises(FileExistsError):
            converter.cets_to_pytom_tm(scores, self.test_dir, TOMO_NAME)

    def test_not_a_scores_map(self) -> None:
        converter = PytomScoreTomogram()
        scores = converter.pytom_tm_to_cets(SCORES_MAP, TOMO_ID)
        probabilities = scores.model_copy(
            update={"value_type": ProbabilityMapValueType.probability.value}
        )
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(probabilities, self.test_dir / "job", TOMO_NAME)
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(
                scores.model_copy(update={"path": None}), self.test_dir, TOMO_NAME
            )


class CetsYamlToPytomTest(CetsToPytomBaseTest):
    """From the CETS yaml files of ``data/outputs``, as written by ``pytom_tm_to_cets``."""

    @staticmethod
    def coords_yaml() -> PointMatrixSet3D:
        return load_cets_object(COORDS_YAML, "coordinates", PointMatrixSet3D)

    @staticmethod
    def scores_yaml() -> ProbabilityMap3D:
        return load_cets_object(SCORES_YAML, "scores", ProbabilityMap3D)

    def test_yaml_files_are_up_to_date(self) -> None:
        """They are what the converters give for the pytom-match-pick files (the scores
        map with its path relative to ``data``)"""
        scores = PytomScoreTomogram().pytom_tm_to_cets(
            SCORES_MAP, TOMO_ID, threshold=CUT_OFF
        )
        self.assertEqual(
            self.scores_yaml(),
            scores.model_copy(update={"path": "./outputs/TS_03_scores.mrc"}),
        )
        coords = PytomCoordinates3D().pytom_tm_to_cets(
            STAR, TOMO_ID, job=read_job(JOB), score_annotation_id=scores.id
        )
        self.assertEqual(self.coords_yaml(), coords)

    def test_coordinates_to_relion5_star(self) -> None:
        """With the job: the star file pytom-match-pick wrote"""
        out = PytomCoordinates3D().cets_to_pytom_tm(
            self.coords_yaml(),
            self.out_star,
            TOMO_NAME,
            cut_off=CUT_OFF,
            search_std=SEARCH_STD,
            relion5_compat=True,
            job=read_job(JOB),
        )
        self.assertEqual(read_star_table(out, "particles"), star_table())

    def test_coordinates_to_voxel_star(self) -> None:
        """Without the job: the star file pytom-match-pick writes without --relion5-compat"""
        out = PytomCoordinates3D().cets_to_pytom_tm(
            self.coords_yaml(),
            self.out_star,
            TOMO_NAME,
            cut_off=CUT_OFF,
            search_std=SEARCH_STD,
        )
        self.assertEqual(
            read_star_table(out, "particles"), to_voxel_flavour(star_table())
        )

    def test_scores_map(self) -> None:
        """Its relative path is read from ``data``: linked to the real map"""
        converter = PytomScoreTomogram()
        scores = self.scores_yaml()
        with contextlib.chdir(DATA_DIR):
            out = converter.cets_to_pytom_tm(scores, self.test_dir / "job", TOMO_NAME)
        self.assertEqual(out.name, f"{TOMO_NAME}_scores.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), SCORES_MAP.resolve())
        with contextlib.chdir(DATA_DIR):
            out = converter.cets_to_pytom_tm(
                scores, self.test_dir / "copy", TOMO_NAME, copy=True
            )
        self.assertFalse(out.is_symlink())
        self.assertEqual(out.read_bytes(), SCORES_MAP.read_bytes())
        # From another directory the map is not found
        with contextlib.chdir(self.test_dir), self.assertRaises(FileNotFoundError):
            converter.cets_to_pytom_tm(scores, self.test_dir / "other", TOMO_NAME)
