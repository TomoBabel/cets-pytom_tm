# cets-pytom_tm
Contains the converters from pytom-match-pick (`pytom_tm`) to CETS:

- `PytomCoordinates3D`: particles star file (`<tomo>_particles.star`, written by
  `pytom_extract_candidates.py`, with or without `--relion5-compat`) to `PointMatrixSet3D`
  (positions, orientations and template matching scores).
- `PytomScoreTomogram`: scores map (`<tomo>_scores.mrc`, written by
  `pytom_match_template.py`) to `ProbabilityMap3D`.

The job file (`<tomo>_job.json`) and the scores map of a star file are found by name next to
it (`pytom_tm.utils.job.find_job`), which links the coordinates to the scores map they were
extracted from. See `pytom_tm/scripts/example.py`.
