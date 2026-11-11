import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from cets_data_model.models.models import (
    AnnotatedObject,
    AnnotationType,
    PointMatrixSet3D,
)
from cets_data_model.utils.image_utils import get_mrc_info
from pytom_tm.constants import (
    ANGLE_PSI,
    ANGLE_ROT,
    ANGLE_TILT,
    CENTERED_COORD_X,
    CENTERED_COORD_Y,
    CENTERED_COORD_Z,
    COORD_X,
    COORD_Y,
    COORD_Z,
    CUT_OFF,
    DETECTOR_PIXEL_SIZE,
    LCC_MAX,
    MICROGRAPH_NAME,
    PARTICLES_BLOCK,
    STAR_EXT,
    TILT_SERIES_PIXEL_SIZE,
    TOMO_NAME,
)
from pytom_tm.utils.job import PytomJob
from pytom_tm.utils.utils import (
    gen_array_to_physical,
    gen_coordinate_systems,
    read_star_table,
    relion_euler_to_matrix,
    validate_file,
)

StarTable = Dict[str, List[str]]


class PytomCoordinates3D:
    def pytom_tm_to_cets(
        self,
        star_file: os.PathLike,
        tomo_id: str,
        tomo_name: Optional[str] = None,
        pixel_size: Optional[float] = None,
        job: Optional[PytomJob] = None,
        reference_map: Optional[os.PathLike] = None,
        annotated_object: Optional[AnnotatedObject] = None,
        score_annotation_id: Optional[str] = None,
        annotation_id: Optional[str] = None,
    ) -> Optional[PointMatrixSet3D]:
        """Converts the particles picked by pytom-match-pick template matching (a star file
        written by pytom_extract_candidates.py, <tomo_id>_particles.star) for one tomogram into
        CETS metadata.

        The picked particles are represented as a ``PointMatrixSet3D`` annotation (stored under
        ``Region.annotations``) that links back to the tomogram through ``source_tomogram_id``:

        * ``origin3D``: the particle positions in the tomogram array (voxel) frame, 0-based.
          Both flavours of the star file are supported: the default one stores them directly
          (rlnCoordinateX/Y/Z), and the RELION5 one (--relion5-compat) stores them in Å relative
          to the centre of the tomogram (rlnCenteredCoordinateX/Y/ZAngst), which are converted
          back to voxels undoing the operation of pytom-match-pick. This needs the size of the
          tomogram, taken from ``job`` or, without it, from ``reference_map``.
        * ``matrix3D``: the rotation matrix of each particle, the one that rotates the template
          into the particle, built from its RELION Euler angles (see
          ``relion_euler_to_matrix``).
        * ``scores``: the template matching score of each particle (rlnLCCmax).

        The particles keep the order of the star file (descending score).

        :param star_file: path to the particles star file.
        :type star_file: os.PathLike.

        :param tomo_id: identifier of the CETS tomogram the particles were picked in. It is used
        as ``source_tomogram_id`` and to name the coordinate systems, so it must match the id of
        the CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        :type tomo_id: str.

        :param tomo_name: name of the tomogram (rlnMicrographName / rlnTomoName column) of the
        particles to be converted, as a star file may contain the particles of several
        tomograms (e.g. merged with pytom_merge_stars.py). When omitted, the star file must
        contain the particles of a single tomogram.
        :type tomo_name: str, optional. Defaults to None.

        :param pixel_size: sampling rate (Å/px) of the tomogram. If given, it is used;
        otherwise it is taken from ``job``, from the star file (rlnDetectorPixelSize /
        rlnTomoTiltSeriesPixelSize, where pytom-match-pick writes the voxel size of the
        tomogram) or from the header of ``reference_map``, in this order.
        :type pixel_size: float, optional. Defaults to None.

        :param job: template matching job that produced the star file (see
        ``pytom_tm.utils.job.find_job``). It provides the size and voxel size of the tomogram
        and, for the jobs created from RELION5 tomograms, the binning and the tilt-series
        pixel size the centred positions refer to: without it, the centred positions of such
        jobs cannot be converted correctly.
        :type job: PytomJob, optional. Defaults to None.

        :param reference_map: an mrc file on the voxel grid the particles were picked in (the
        tomogram or the scores map). Used, when there is no ``job``, to get the size of the
        tomogram, and to check that all the particles are inside the grid.
        :type reference_map: os.PathLike, optional. Defaults to None.

        :param annotated_object: what the picked particles are (e.g. the template structure).
        Template matching does not record it.
        :type annotated_object: AnnotatedObject, optional. Defaults to None.

        :param score_annotation_id: id of the annotation of the scores map the particles were
        extracted from (see ``PytomScoreTomogram``), stored as ``derived_from_annotation_id``.
        :type score_annotation_id: str, optional. Defaults to None.

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations. Defaults to ``get_annotation_id(tomo_id)``.
        :type annotation_id: str, optional. Defaults to None.

        :returns: the coordinates annotation, or None if there are no particles of the tomogram.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        star_path = validate_file(star_file, expected_ext=STAR_EXT)
        table = self._select_tomogram(self._read_particles(star_path), tomo_name)
        n_particles = len(table[LCC_MAX])
        if n_particles == 0:
            return None

        dims: Optional[Tuple[int, int, int]] = None
        ref_pixel_size: Optional[float] = None
        if job is not None:
            dims = job.tomo_shape
        if reference_map is not None:
            mrc_info = get_mrc_info(reference_map)
            ref_dims = (mrc_info.size_x, mrc_info.size_y, mrc_info.size_z)
            if dims is not None and dims != ref_dims:
                raise ValueError(
                    f"The size of {reference_map} {ref_dims} is not the one of the "
                    f"tomogram of the job {dims}."
                )
            dims = ref_dims
            ref_pixel_size = mrc_info.apix_x or None
        star_pixel_size = self._get_star_pixel_size(table)

        origin_3d = self._get_positions(table, dims, star_pixel_size, job, star_path)
        if dims is not None:
            self._check_inside_grid(origin_3d, dims, star_path)
        matrix_3d = [
            relion_euler_to_matrix(float(rot), float(tilt), float(psi)).tolist()
            for rot, tilt, psi in zip(
                table[ANGLE_ROT], table[ANGLE_TILT], table[ANGLE_PSI]
            )
        ]

        # Picked coordinates are positions in the tomogram's array (voxel) frame. When the
        # tomogram sampling rate is known, the physical (Å) system and the canonical
        # array_to_physical scale are also attached, making the coordinates self-describing.
        if pixel_size is None:
            pixel_size = (
                (job.voxel_size if job is not None else None)
                or star_pixel_size
                or ref_pixel_size
            )
        array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
        if pixel_size is not None:
            coordinate_systems = [array_cs, physical_cs]
            coordinate_transformations = [
                gen_array_to_physical(
                    pixel_size, array_cs.name, physical_cs.name, ndim=3
                )
            ]
        else:
            coordinate_systems = [array_cs]
            coordinate_transformations = []

        return PointMatrixSet3D(
            id=annotation_id or self.get_annotation_id(tomo_id),
            name=f"pytom-match-pick picked coordinates for {tomo_id}",
            annotation_type=AnnotationType.point_matrix_set_3D,
            source_tomogram_id=tomo_id,
            annotated_object=annotated_object,
            derived_from_annotation_id=score_annotation_id,
            origin3D=origin_3d.tolist(),
            matrix3D=matrix_3d,
            scores=[float(score) for score in table[LCC_MAX]],
            coordinate_systems=coordinate_systems,
            coordinate_transformations=coordinate_transformations,
        )

    @staticmethod
    def get_annotation_id(tomo_id: str) -> str:
        """Default id of the coordinates annotation."""
        return f"pytom_tm_coords_{tomo_id}"

    def get_tomo_names(self, star_file: os.PathLike) -> List[str]:
        """Names of the tomograms (rlnMicrographName / rlnTomoName column) present in a
        particles star file, in order of appearance."""
        table = self._read_particles(validate_file(star_file, expected_ext=STAR_EXT))
        return list(dict.fromkeys(table[self._tomo_name_column(table)]))

    def get_cut_off(
        self, star_file: os.PathLike, tomo_name: Optional[str] = None
    ) -> Optional[float]:
        """Score cut-off used to extract the particles of a tomogram (rlnCutOff column), or
        None if there are no particles of the tomogram."""
        table = self._select_tomogram(
            self._read_particles(validate_file(star_file, expected_ext=STAR_EXT)),
            tomo_name,
        )
        cut_offs = table.get(CUT_OFF, [])
        return float(cut_offs[0]) if cut_offs else None

    @staticmethod
    def _read_particles(star_path: os.PathLike) -> StarTable:
        table = read_star_table(star_path, PARTICLES_BLOCK)
        has_positions = all(c in table for c in (COORD_X, COORD_Y, COORD_Z)) or all(
            c in table for c in (CENTERED_COORD_X, CENTERED_COORD_Y, CENTERED_COORD_Z)
        )
        required = (ANGLE_ROT, ANGLE_TILT, ANGLE_PSI, LCC_MAX)
        if not has_positions or any(column not in table for column in required):
            raise ValueError(
                f"{star_path} is not a pytom-match-pick particles star file: found the "
                f"columns {list(table)}"
            )
        return table

    @staticmethod
    def _tomo_name_column(table: StarTable) -> str:
        return TOMO_NAME if TOMO_NAME in table else MICROGRAPH_NAME

    def _select_tomogram(self, table: StarTable, tomo_name: Optional[str]) -> StarTable:
        names = table.get(self._tomo_name_column(table))
        if names is None:
            if tomo_name is not None:
                raise ValueError(
                    "The star file does not contain the name of the tomogram of the "
                    "particles, so they cannot be selected by tomo_name."
                )
            return table
        if tomo_name is None:
            unique_names = list(dict.fromkeys(names))
            if len(unique_names) > 1:
                raise ValueError(
                    f"The star file contains the particles of several tomograms "
                    f"{unique_names}; choose one with tomo_name."
                )
            return table
        selected = [i for i, name in enumerate(names) if name == tomo_name]
        return {
            column: [values[i] for i in selected] for column, values in table.items()
        }

    @staticmethod
    def _get_star_pixel_size(table: StarTable) -> Optional[float]:
        """Voxel size of the tomogram written in the star file, if any."""
        for column in (TILT_SERIES_PIXEL_SIZE, DETECTOR_PIXEL_SIZE):
            if table.get(column):
                return float(table[column][0]) or None
        return None

    @staticmethod
    def _get_positions(
        table: StarTable,
        dims: Optional[Tuple[int, int, int]],
        star_pixel_size: Optional[float],
        job: Optional[PytomJob],
        star_path: os.PathLike,
    ) -> np.ndarray:
        """Particle positions (N, 3) in the 0-based array (voxel) frame of the tomogram."""
        if COORD_X in table:
            # Default flavour: indices of the voxels of the scores map
            return np.array(
                [table[COORD_X], table[COORD_Y], table[COORD_Z]], dtype=float
            ).T
        # RELION5 flavour: Å relative to the centre of the tomogram
        centered = np.array(
            [
                table[CENTERED_COORD_X],
                table[CENTERED_COORD_Y],
                table[CENTERED_COORD_Z],
            ],
            dtype=float,
        ).T
        voxel_size = job.voxel_size if job is not None else star_pixel_size
        if dims is None or not voxel_size:
            raise ValueError(
                f"The positions of {star_path} are relative to the centre of the tomogram "
                f"and in Å (--relion5-compat): the size and voxel size of the tomogram are "
                f"needed to convert them. Provide the job or a reference map."
            )
        dims_array = np.array(dims, dtype=float)
        if job is not None and job.relion_binning and job.relion_ts_pixel_size:
            # Jobs of RELION5 tomograms: pytom-match-pick writes
            # (voxel * binning - (dims * binning / 2 - 1)) * tilt-series pixel size
            binning = job.relion_binning
            center = dims_array * binning / 2 - 1
            return (centered / job.relion_ts_pixel_size + center) / binning
        # Other jobs: pytom-match-pick writes (voxel - dims / 2) * voxel size
        return centered / voxel_size + dims_array / 2

    @staticmethod
    def _check_inside_grid(
        origin_3d: np.ndarray, dims: Tuple[int, int, int], star_path: os.PathLike
    ) -> None:
        # Tolerance for the rounding of the positions written in Å
        tolerance = 1e-3
        outside = np.any(
            (origin_3d < -tolerance) | (origin_3d > np.array(dims) - 1 + tolerance),
            axis=1,
        )
        if np.any(outside):
            raise ValueError(
                f"{int(outside.sum())} particles of {star_path} fall outside the "
                f"{dims[0]} x {dims[1]} x {dims[2]} voxel grid of the tomogram; it is not "
                f"the one they were picked in."
            )
