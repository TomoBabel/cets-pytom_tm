# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import ProbabilityMapValueType
from pytom_tm.converters.coordinates3d import PytomCoordinates3D
from pytom_tm.converters.score_tomogram import PytomScoreTomogram
from pytom_tm.tests import (
    GRID,
    PIXEL_SIZE,
    REAL_JOB,
    REAL_STAR,
    REAL_TOMO_NAME,
    TOMO_ID,
    TOMO_NAME,
    particle,
    write_job,
    write_mrc,
    write_star,
)
from pytom_tm.utils.job import read_job
from pytom_tm.utils.utils import (
    matrix_to_relion_euler,
    read_star_table,
    relion_euler_to_matrix,
)


class CetsToPytomBaseTest(TestCase):
    test_dir = Path()

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="cets_to_pytom_tm"))
        self.star = self.test_dir / f"{TOMO_NAME}_particles.star"
        self.out_star = self.test_dir / "out" / f"{TOMO_NAME}_particles.star"
        self.out_star.parent.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class MatrixToRelionEulerTest(TestCase):
    def test_inverse_of_relion_euler_to_matrix(self) -> None:
        for angles in [(67.5, 90.0, -60.0), (45.0, 109.471221, -180.0), (-10, 1, 20)]:
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
    def test_voxel_positions_roundtrip(self) -> None:
        """star -> CETS -> star gives back the original table (default flavour)"""
        write_star(
            self.star,
            [
                particle((1, 2, 3), angles=(67.5, 90.0, -60.0), score=0.3),
                particle((39, 29, 19), angles=(45.0, 70.528779, 60.0), score=0.2),
            ],
        )
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        out = converter.cets_to_pytom_tm(
            coords, self.out_star, TOMO_NAME, cut_off=0.1, search_std=0.02
        )
        original = read_star_table(self.star, "particles")
        written = read_star_table(out, "particles")
        self.assertEqual(list(written), list(original))
        for label in original:
            if label == "rlnMicrographName":
                self.assertEqual(written[label], original[label])
            else:
                np.testing.assert_allclose(
                    [float(v) for v in written[label]],
                    [float(v) for v in original[label]],
                    atol=1e-6,
                )

    def test_real_star_roundtrip(self) -> None:
        """Real --relion5-compat star file: written back with the same text"""
        converter = PytomCoordinates3D()
        job = read_job(REAL_JOB)
        coords = converter.pytom_tm_to_cets(REAL_STAR, "TS_03", job=job)
        assert coords is not None
        original = read_star_table(REAL_STAR, "particles")
        out = converter.cets_to_pytom_tm(
            coords,
            self.out_star,
            REAL_TOMO_NAME,
            cut_off=float(original["rlnCutOff"][0]),
            search_std=float(original["rlnSearchStd"][0]),
            relion5_compat=True,
            job=job,
        )
        self.assertEqual(read_star_table(out, "particles"), original)

    def test_centered_positions(self) -> None:
        """relion5_compat: (voxel - dims / 2) * voxel size, with the given tomogram size"""
        voxels = np.array([[1.0, 2.0, 3.0], [39.0, 29.0, 19.0]])
        write_star(self.star, [particle(p) for p in voxels])
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        with self.assertRaises(ValueError):  # the size of the tomogram is unknown
            converter.cets_to_pytom_tm(
                coords, self.out_star, TOMO_NAME, relion5_compat=True
            )
        out = converter.cets_to_pytom_tm(
            coords, self.out_star, TOMO_NAME, relion5_compat=True, tomo_shape=GRID
        )
        table = read_star_table(out, "particles")
        centered = np.array(
            [table[f"rlnCenteredCoordinate{axis}Angst"] for axis in "XYZ"], dtype=float
        ).T
        np.testing.assert_allclose(
            centered, (voxels - np.array(GRID) / 2) * PIXEL_SIZE, atol=1e-6
        )
        self.assertEqual(table["rlnTomoName"], [TOMO_NAME] * 2)
        self.assertEqual(table["rlnTomoTiltSeriesPixelSize"], ["5.400000"] * 2)

    def test_centered_positions_of_relion_job(self) -> None:
        binning, ts_pixel_size = 4.0, 1.35
        voxels = np.array([[1.0, 2.0, 3.0], [39.0, 29.0, 19.0]])
        write_star(self.star, [particle(p) for p in voxels])
        job = read_job(
            write_job(
                self.test_dir / f"{TOMO_NAME}_job.json",
                relion_binning=binning,
                relion_ts_pixel_size=ts_pixel_size,
            )
        )
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        out = converter.cets_to_pytom_tm(
            coords, self.out_star, TOMO_NAME, relion5_compat=True, job=job
        )
        # Read back with the job: the same voxels
        back = converter.pytom_tm_to_cets(out, TOMO_ID, job=job)
        assert back is not None
        np.testing.assert_allclose(back.origin3D, voxels, atol=1e-5)

    def test_optional_columns(self) -> None:
        """Cut-off, search std and voxel size are only written when known"""
        write_star(self.star, [particle((1, 2, 3))])
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        coords = coords.model_copy(
            update={"coordinate_transformations": [], "matrix3D": None, "scores": None}
        )
        table = read_star_table(
            converter.cets_to_pytom_tm(coords, self.out_star, TOMO_NAME), "particles"
        )
        self.assertEqual(
            list(table),
            [
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnLCCmax",
                "rlnMicrographName",
            ],
        )
        self.assertEqual(table["rlnAngleTilt"], ["0.000000"])
        self.assertEqual(table["rlnLCCmax"], ["0.000000"])

    def test_invalid_input(self) -> None:
        write_star(self.star, [particle((1, 2, 3))])
        converter = PytomCoordinates3D()
        coords = converter.pytom_tm_to_cets(self.star, TOMO_ID)
        assert coords is not None
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(coords, self.test_dir / "out.em", TOMO_NAME)
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(
                coords.model_copy(update={"scores": [0.1, 0.2]}),
                self.out_star,
                TOMO_NAME,
            )


class CetsToPytomScoresTest(CetsToPytomBaseTest):
    def test_linked_with_the_pytom_name(self) -> None:
        converter = PytomScoreTomogram()
        scores_map = write_mrc(self.test_dir / "scores.mrc")
        scores = converter.pytom_tm_to_cets(scores_map, TOMO_ID)
        out = converter.cets_to_pytom_tm(scores, self.test_dir / "job", TOMO_NAME)
        self.assertEqual(out.name, f"{TOMO_NAME}_scores.mrc")
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), scores_map.resolve())

    def test_copied_if_the_sampling_rate_differs(self) -> None:
        converter = PytomScoreTomogram()
        scores_map = write_mrc(self.test_dir / "scores.mrc", voxel_size=1.0)
        scores = converter.pytom_tm_to_cets(scores_map, TOMO_ID, pixel_size=PIXEL_SIZE)
        out = converter.cets_to_pytom_tm(scores, self.test_dir / "job", TOMO_NAME)
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), PIXEL_SIZE, places=5)

    def test_not_a_scores_map(self) -> None:
        converter = PytomScoreTomogram()
        scores = converter.pytom_tm_to_cets(
            write_mrc(self.test_dir / "scores.mrc"), TOMO_ID
        )
        probabilities = scores.model_copy(
            update={"value_type": ProbabilityMapValueType.probability.value}
        )
        with self.assertRaises(ValueError):
            converter.cets_to_pytom_tm(probabilities, self.test_dir / "job", TOMO_NAME)
