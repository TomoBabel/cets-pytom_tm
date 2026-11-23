# -*- coding: utf-8 -*-
"""Tests of the ``cets-pytom_tm`` command line, on the real template matching of TS_03 (see
``pytom_tm.tests``), in both directions."""

import contextlib
import hashlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import TestCase

import mrcfile
import numpy as np

from cets_data_model.models.models import PointMatrixSet3D, ProbabilityMap3D
from cets_data_model.utils.yaml_io import load_cets_object
from pytom_tm.cli import main
from pytom_tm.tests import (
    ANGLE_COLUMNS,
    CENTERED_COLUMNS,
    COORDS_YAML,
    CUT_OFF,
    DATA_DIR,
    FIRST_PARTICLES,
    GRID,
    JOB,
    MASK,
    N_PARTICLES,
    PIXEL_SIZE,
    SCORES_MAP,
    SCORES_YAML,
    SEARCH_STD,
    STAR,
    TOMO_ID,
    TOMO_NAME,
    TOMOGRAM,
    VOXEL_COLUMNS,
    columns,
    star_table,
    to_voxel_flavour,
    write_job_copy,
    write_table,
)
from pytom_tm.utils.utils import read_star_table, relion_euler_to_matrix

SCORES_ID = f"pytom_tm_scores_{TOMO_ID}"


def run(args: List[str]) -> Tuple[int, str, str]:
    """Runs cets-pytom_tm with the given arguments; returns the exit code and what it
    printed to stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def tree_digest(directory: Path) -> Dict[str, str]:
    """sha256 of every file under a directory, to check that it is not modified."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def first_voxels() -> np.ndarray:
    return np.array([voxel for voxel, _, _, _ in FIRST_PARTICLES], dtype=float)


class CetsPytomCliTest(TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_pytom_tm_cli"))
        self.yaml = self.dir / "TS_03_cets.yaml"
        self.out_star = self.dir / "out" / "sub" / f"{TOMO_NAME}_particles.star"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def assert_run(self, args: List[str], code: int = 0) -> str:
        """Runs cets-pytom_tm, checks its exit code and returns what it printed to
        stdout and stderr."""
        result, out, err = run(args)
        self.assertEqual(result, code, msg=f"stdout:\n{out}\nstderr:\n{err}")
        return out + err

    def _coords_args(self, *extra: str, star: Path = STAR) -> List[str]:
        return [
            "coordinates",
            "to-cets",
            "--star",
            str(star),
            "--tomo-id",
            TOMO_ID,
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _coords_to_cets(self, *extra: str, star: Path = STAR) -> PointMatrixSet3D:
        self.assert_run(self._coords_args(*extra, star=star))
        return load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)

    def _coords_from_cets(self, *extra: str, code: int = 0) -> Dict[str, List[str]]:
        self.assert_run(
            [
                "coordinates",
                "from-cets",
                "-i",
                str(self.yaml),
                "-o",
                str(self.out_star),
                *extra,
            ],
            code,
        )
        return read_star_table(self.out_star, "particles") if code == 0 else {}

    def _scores_args(self, *extra: str) -> List[str]:
        return [
            "scores",
            "to-cets",
            "--scores-map",
            str(SCORES_MAP),
            "--tomo-id",
            TOMO_ID,
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _scores_to_cets(self, *extra: str) -> ProbabilityMap3D:
        self.assert_run(self._scores_args(*extra))
        return load_cets_object(self.yaml, "scores", ProbabilityMap3D)

    # ------------------------------------------------------------------ coordinates
    def test_coordinates_to_cets(self) -> None:
        """--relion5-compat star file: the job file next to it gives the size and voxel
        size of the tomogram"""
        output = self.assert_run(self._coords_args())
        self.assertIn(f"Using the job file found next to the star file: {JOB}", output)
        coords = load_cets_object(self.yaml, "coordinates", PointMatrixSet3D)
        assert coords.scores is not None and coords.matrix3D is not None
        self.assertEqual(coords.id, f"pytom_tm_coords_{TOMO_ID}")
        self.assertEqual(coords.source_tomogram_id, TOMO_ID)
        self.assertEqual(len(coords.origin3D), N_PARTICLES)
        np.testing.assert_allclose(coords.origin3D[:3], first_voxels(), atol=1e-6)
        for i, (_, angles, score, _) in enumerate(FIRST_PARTICLES):
            np.testing.assert_allclose(
                coords.matrix3D[i], relion_euler_to_matrix(*angles), atol=1e-12
            )
            self.assertEqual(coords.scores[i], score)
        self.assertEqual(coords.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3)
        self.assertIsNone(coords.derived_from_annotation_id)
        self.assertIsNone(coords.annotated_object)

    def test_coordinates_options(self) -> None:
        coords = self._coords_to_cets(
            "--annotated-object",
            "ribosome",
            "--score-annotation-id",
            SCORES_ID,
            "--annotation-id",
            "my_coords",
            "--pixel-size",
            "10.8",
            "--reference-map",
            str(SCORES_MAP),
        )
        self.assertEqual(coords.id, "my_coords")
        self.assertEqual(coords.derived_from_annotation_id, SCORES_ID)
        assert coords.annotated_object is not None
        self.assertEqual(coords.annotated_object.name, "ribosome")
        self.assertEqual(coords.coordinate_transformations[0].scale, [10.8] * 3)
        # The positions are still the voxels: they were converted with the job
        np.testing.assert_allclose(coords.origin3D[:3], first_voxels(), atol=1e-6)

    def test_coordinates_job_options(self) -> None:
        """--no-job: the size of the tomogram is unknown, unless a reference map gives it;
        --job: another job file"""
        self.assert_run(self._coords_args("--no-job"), 1)
        coords = self._coords_to_cets("--no-job", "--reference-map", str(TOMOGRAM))
        np.testing.assert_allclose(coords.origin3D[:3], first_voxels(), atol=1e-6)
        # A job of a tomogram of another size: other voxels, outside its grid
        small_job = write_job_copy(
            self.dir / "small.json", {"tomo_shape": [100, 100, 40]}
        )
        self.assert_run(self._coords_args("--job", str(small_job)), 1)
        job = write_job_copy(self.dir / "other.json")
        coords = self._coords_to_cets("--job", str(job))
        np.testing.assert_allclose(coords.origin3D[:3], first_voxels(), atol=1e-6)
        # A reference map that is not on the grid of the job
        self.assert_run(self._coords_args("--reference-map", str(MASK)), 1)

    def test_coordinates_voxel_flavour(self) -> None:
        """Default star file (no job needed)"""
        star = write_table(
            self.dir / f"{TOMO_NAME}_particles.star", to_voxel_flavour(star_table())
        )
        coords = self._coords_to_cets(star=star)
        np.testing.assert_allclose(coords.origin3D[:3], first_voxels(), atol=1e-6)
        self.assertEqual(coords.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3)

    def test_coordinates_of_several_tomograms(self) -> None:
        table = star_table()
        merged = {label: values + values[:5] for label, values in table.items()}
        merged["rlnTomoName"][N_PARTICLES:] = ["TS_04"] * 5
        star = write_table(self.dir / "merged_particles.star", merged)
        job = str(JOB)
        self.assert_run(self._coords_args("--job", job, star=star), 1)
        coords = self._coords_to_cets("--job", job, "--tomo-name", "TS_04", star=star)
        self.assertEqual(len(coords.origin3D), 5)
        coords = self._coords_to_cets("--job", job, "--tomo-name", TOMO_NAME, star=star)
        self.assertEqual(len(coords.origin3D), N_PARTICLES)
        # No particles of the tomogram
        self.assert_run(
            self._coords_args("--job", job, "--tomo-name", "TS_05", star=star), 1
        )

    def test_coordinates_round_trip(self) -> None:
        """star -> CETS -> star, with the job: the same table"""
        self._coords_to_cets()
        table = self._coords_from_cets(
            "--relion5-compat",
            "--job",
            str(JOB),
            "--cut-off",
            str(CUT_OFF),
            "--search-std",
            str(SEARCH_STD),
        )
        self.assertEqual(table, star_table())

    def test_coordinates_from_cets_voxel_flavour(self) -> None:
        self._coords_to_cets()
        table = self._coords_from_cets()
        original = star_table()
        voxels = columns(original, CENTERED_COLUMNS) / PIXEL_SIZE + np.array(GRID) / 2
        np.testing.assert_allclose(columns(table, VOXEL_COLUMNS), voxels, atol=1e-6)
        for label in ANGLE_COLUMNS + ["rlnLCCmax"]:
            self.assertEqual(table[label], original[label])
        self.assertEqual(set(table["rlnDetectorPixelSize"]), {"21.600000"})
        # Default name: the source tomogram id; CETS does not store the cut-off
        self.assertEqual(set(table["rlnMicrographName"]), {TOMO_ID})
        self.assertNotIn("rlnCutOff", table)
        table = self._coords_from_cets("--tomo-name", "tomo_a", "--pixel-size", "10.8")
        self.assertEqual(set(table["rlnMicrographName"]), {"tomo_a"})
        self.assertEqual(set(table["rlnDetectorPixelSize"]), {"10.800000"})

    def test_coordinates_from_cets_relion5_without_job(self) -> None:
        self._coords_to_cets()
        # Without the size of the tomogram
        self._coords_from_cets("--relion5-compat", code=1)
        table = self._coords_from_cets(
            "--relion5-compat", "--tomo-shape", *[str(n) for n in GRID]
        )
        original = star_table()
        for label in CENTERED_COLUMNS + ANGLE_COLUMNS + ["rlnLCCmax", "rlnTomoName"]:
            self.assertEqual(table[label], original[label])

    # ------------------------------------------------------------------ scores
    def test_scores_to_cets(self) -> None:
        scores = self._scores_to_cets("--threshold", str(CUT_OFF))
        self.assertEqual(scores.id, SCORES_ID)
        self.assertEqual(scores.threshold, CUT_OFF)
        self.assertEqual((scores.width, scores.height, scores.depth), GRID)
        self.assertEqual(scores.path, str(SCORES_MAP.resolve()))
        self.assertEqual(scores.coordinate_transformations[0].scale, [PIXEL_SIZE] * 3)
        scores = self._scores_to_cets(
            "--pixel-size", "10.8", "--annotation-id", "s", "--annotated-object", "x"
        )
        self.assertEqual(scores.id, "s")
        self.assertIsNone(scores.threshold)
        self.assertEqual(scores.coordinate_transformations[0].scale, [10.8] * 3)

    def test_scores_from_cets(self) -> None:
        self._scores_to_cets()
        out_dir = self.dir / "tm"
        from_cets = ["scores", "from-cets", "-i", str(self.yaml), "--output-dir"]
        self.assert_run([*from_cets, str(out_dir)])
        # Default name: the source tomogram id; linked
        out = out_dir / f"{TOMO_ID}_scores.mrc"
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), SCORES_MAP.resolve())
        self.assert_run([*from_cets, str(out_dir), "--tomo-name", "tomo_a", "--copy"])
        out = out_dir / "tomo_a_scores.mrc"
        self.assertFalse(out.is_symlink())
        self.assertEqual(out.read_bytes(), SCORES_MAP.read_bytes())

    def test_scores_from_cets_other_voxel_size(self) -> None:
        """Copied (whole) with the header updated"""
        self._scores_to_cets("--pixel-size", "10.8")
        self.assert_run(
            ["scores", "from-cets", "-i", str(self.yaml), "--output-dir", str(self.dir)]
        )
        out = self.dir / f"{TOMO_ID}_scores.mrc"
        self.assertFalse(out.is_symlink())
        with mrcfile.open(out, header_only=True) as mrc:
            self.assertAlmostEqual(float(mrc.voxel_size.x), 10.8, places=4)
        self.assertEqual(out.stat().st_size, SCORES_MAP.stat().st_size)
        np.testing.assert_array_equal(mrcfile.read(out), mrcfile.read(SCORES_MAP))

    # ------------------------------------------------------------------ errors
    # ------------------------------------------------------- from the CETS yaml files
    def test_coordinates_from_cets_yaml(self) -> None:
        """The coordinates yaml of ``data/outputs``: with the job, the star file
        pytom-match-pick wrote; without it, the one it writes without --relion5-compat"""
        from_cets = ["coordinates", "from-cets", "-i", str(COORDS_YAML)]
        values = ["--cut-off", str(CUT_OFF), "--search-std", str(SEARCH_STD)]
        self.assert_run(
            [
                *from_cets,
                "-o",
                str(self.out_star),
                "--relion5-compat",
                "--job",
                str(JOB),
            ]
            + values
        )
        self.assertEqual(read_star_table(self.out_star, "particles"), star_table())
        voxel_star = self.dir / "voxel" / f"{TOMO_NAME}_particles.star"
        self.assert_run([*from_cets, "-o", str(voxel_star), *values])
        self.assertEqual(
            read_star_table(voxel_star, "particles"), to_voxel_flavour(star_table())
        )

    def test_scores_from_cets_yaml(self) -> None:
        """The scores yaml of ``data/outputs``: its path is relative to ``data``"""
        from_cets = ["scores", "from-cets", "-i", str(SCORES_YAML), "--output-dir"]
        out = self.dir / "tm" / f"{TOMO_ID}_scores.mrc"
        with contextlib.chdir(DATA_DIR):
            self.assert_run([*from_cets, str(out.parent)])
        self.assertTrue(out.is_symlink())
        self.assertEqual(out.resolve(), SCORES_MAP.resolve())
        with contextlib.chdir(DATA_DIR):
            self.assert_run([*from_cets, str(self.dir), "--tomo-name", "a", "--copy"])
        self.assertEqual(
            (self.dir / "a_scores.mrc").read_bytes(), SCORES_MAP.read_bytes()
        )
        # From another directory the map is not found
        with contextlib.chdir(self.dir):
            output = self.assert_run([*from_cets, str(self.dir / "other")], 1)
        self.assertIn("TS_03_scores.mrc", output)
        self.assertFalse((self.dir / "other").exists())

    def test_errors_give_exit_code_1(self) -> None:
        missing = self._coords_args(star=self.dir / "missing_particles.star")
        not_star = self._coords_args(star=JOB)
        not_mrc = self._scores_args()
        not_mrc[not_mrc.index("--scores-map") + 1] = str(STAR)
        zero_pixel_size = self._scores_args("--pixel-size", "0")
        for args in (missing, not_star, not_mrc, zero_pixel_size):
            with self.subTest(args=args):
                output = self.assert_run(args, 1)
                self.assertEqual(len(output.strip().splitlines()), 1)
                self.assertFalse(self.yaml.exists())
        # Coordinates are not a scores map
        self._coords_to_cets()
        self.assert_run(
            [
                "scores",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.dir),
            ],
            1,
        )
        # The output exists and is another file
        self._scores_to_cets()
        (self.dir / f"{TOMO_ID}_scores.mrc").write_text("other")
        self.assert_run(
            [
                "scores",
                "from-cets",
                "-i",
                str(self.yaml),
                "--output-dir",
                str(self.dir),
            ],
            1,
        )

    def test_input_files_are_not_modified(self) -> None:
        before = tree_digest(DATA_DIR)
        self._coords_to_cets()
        self._coords_from_cets("--relion5-compat", "--job", str(JOB))
        self._scores_to_cets("--pixel-size", "10.8")
        self.assert_run(
            ["scores", "from-cets", "-i", str(self.yaml), "--output-dir", str(self.dir)]
        )
        self.assert_run(
            ["coordinates", "from-cets", "-i", str(COORDS_YAML), "-o"]
            + [str(self.dir / "yaml" / f"{TOMO_NAME}_particles.star")]
            + ["--relion5-compat", "--job", str(JOB)]
        )
        with contextlib.chdir(DATA_DIR):
            self.assert_run(
                ["scores", "from-cets", "-i", str(SCORES_YAML), "--output-dir"]
                + [str(self.dir / "yaml"), "--copy"]
            )
        self.assertEqual(tree_digest(DATA_DIR), before)

    def test_invalid_arguments(self) -> None:
        """Usage errors exit with code 2 (argparse) before running any conversion."""
        invalid: List[List[str]] = [
            [],
            ["coordinates"],
            ["tomogram", "to-cets"],
            # --tomo-id is required
            ["coordinates", "to-cets", "--star", str(STAR), "-o", str(self.yaml)],
            self._coords_args("--pixel-size", "one"),
            ["coordinates", "from-cets", "-i", str(self.yaml)],  # -o
            [
                "coordinates",
                "from-cets",
                "-i",
                "x.yaml",
                "-o",
                "x.star",
                "--tomo-shape",
                "1",
            ],
            ["scores", "from-cets", "-i", str(self.yaml)],  # --output-dir
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as ctx:
                    run(args)
                self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(self.yaml.exists())

    def test_help(self) -> None:
        for args in (
            ["--help"],
            ["coordinates", "to-cets", "--help"],
            ["scores", "from-cets", "--help"],
        ):
            with self.subTest(args=args):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit) as ctx:
                        main(args)
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("cets-pytom_tm", out.getvalue())
