import os
from pathlib import Path
from typing import Optional

from cets_data_model.models.models import (
    AnnotatedObject,
    AnnotationType,
    ProbabilityMap3D,
    ProbabilityMapValueType,
)
from cets_data_model.utils.image_utils import get_mrc_info
from pytom_tm.constants import MRC_EXT, SCORES_SUFFIX
from pytom_tm.utils.utils import (
    export_mrc,
    gen_array_to_physical,
    gen_coordinate_systems,
    get_voxel_size,
    validate_file,
)


class PytomScoreTomogram:
    def pytom_tm_to_cets(
        self,
        scores_map: os.PathLike,
        tomo_id: str,
        pixel_size: Optional[float] = None,
        threshold: Optional[float] = None,
        annotated_object: Optional[AnnotatedObject] = None,
        annotation_id: Optional[str] = None,
    ) -> ProbabilityMap3D:
        """Converts a pytom-match-pick scores map (<tomo_id>_scores.mrc, written by
        pytom_match_template.py) into CETS metadata.

        The scores map holds, for each voxel of the tomogram, the best template matching score
        over all the sampled orientations, so it is represented as a ``ProbabilityMap3D``
        annotation (``value_type: score``, stored under ``Region.annotations``) on the voxel
        grid of the tomogram, linked to it through ``source_tomogram_id``. The particles
        extracted from it refer to it through ``derived_from_annotation_id``.

        :param scores_map: path to the scores map.
        :type scores_map: os.PathLike.

        :param tomo_id: identifier of the CETS tomogram that was template-matched. It is used as
        ``source_tomogram_id`` and to name the coordinate systems, so it must match the id of the
        CETS tomogram ({tomo_id}_array / {tomo_id}_physical refer to the same frame).
        :type tomo_id: str.

        :param pixel_size: sampling rate (Å/px). If given, it is used; otherwise it is read from
        the mrc header (pytom-match-pick writes the voxel size of the tomogram there).
        :type pixel_size: float, optional. Defaults to None.

        :param threshold: score above which the voxels were considered particle candidates in
        the particle extraction (rlnCutOff column of the particles star file), if known.
        :type threshold: float, optional. Defaults to None.

        :param annotated_object: what the template represents. Template matching does not
        record it.
        :type annotated_object: AnnotatedObject, optional. Defaults to None.

        :param annotation_id: id of the annotation. It must be unique within its
        Region.annotations. Defaults to ``get_annotation_id(tomo_id)``.
        :type annotation_id: str, optional. Defaults to None.
        """
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError(f"The pixel size must be positive. Got: {pixel_size}")
        scores_path = validate_file(scores_map, expected_ext=MRC_EXT)
        mrc_info = get_mrc_info(scores_path)
        voxel_size = pixel_size or mrc_info.apix_x
        if not voxel_size:
            raise ValueError(
                f"No pixel size was provided and {scores_path} does not contain it in its "
                f"header."
            )
        array_cs, physical_cs = gen_coordinate_systems(tomo_id, ndim=3)
        return ProbabilityMap3D(
            id=annotation_id or self.get_annotation_id(tomo_id),
            name=f"pytom-match-pick template matching scores for {tomo_id}",
            annotation_type=AnnotationType.probability_map_3D,
            source_tomogram_id=tomo_id,
            annotated_object=annotated_object,
            value_type=ProbabilityMapValueType.score,
            threshold=threshold,
            path=str(scores_path),
            width=mrc_info.size_x,
            height=mrc_info.size_y,
            depth=mrc_info.size_z,
            coordinate_systems=[array_cs, physical_cs],
            coordinate_transformations=[
                gen_array_to_physical(
                    voxel_size, array_cs.name, physical_cs.name, ndim=3
                )
            ],
        )

    def cets_to_pytom_tm(
        self,
        scores: ProbabilityMap3D,
        output_directory: os.PathLike,
        tomo_name: str,
        copy: bool = False,
    ) -> Path:
        """Makes the scores map of a CETS ``ProbabilityMap3D`` annotation available with the
        name pytom-match-pick gives it (<tomo_name>_scores.mrc), in the output directory of a
        job.

        The map itself is the mrc file the annotation points to, so nothing is converted: it
        is linked, or copied if requested or if the sampling rate of its header is not the
        one of the annotation (the header of the copy is then updated).

        CETS does not store the rest of the files of the job that
        pytom_extract_candidates.py needs: the angles map (<tomo_name>_angles.mrc) and the
        job file (<tomo_name>_job.json).

        :param scores: the scores map annotation.
        :type scores: ProbabilityMap3D.

        :param output_directory: directory in which the scores map will be written.
        :type output_directory: os.PathLike.

        :param tomo_name: name of the tomogram, as pytom-match-pick names it: the name of the
        tomogram file without extension.
        :type tomo_name: str.

        :param copy: copy the map instead of linking it.
        :type copy: bool, optional. Defaults to False.

        :returns: the path of the scores map.
        """
        if scores.value_type not in (None, ProbabilityMapValueType.score.value):
            raise ValueError(
                f"The annotation [{scores.id}] holds values of type "
                f"'{scores.value_type}', not template matching scores."
            )
        if not scores.path:
            raise ValueError(f"The annotation [{scores.id}] has no file.")
        return export_mrc(
            scores.path,
            Path(output_directory) / f"{tomo_name}{SCORES_SUFFIX}",
            voxel_size=get_voxel_size(scores),
            copy=copy,
        )

    @staticmethod
    def get_annotation_id(tomo_id: str) -> str:
        """Default id of the scores map annotation."""
        return f"pytom_tm_scores_{tomo_id}"
