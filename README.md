# cets-pytom_tm

Converters between [pytom-match-pick](https://github.com/SBC-Utrecht/pytom-match-pick)
(`pytom_tm`) template matching outputs and CETS, the cryo-electron tomography metadata standard
defined in [cets-data-models](https://github.com/TomoBabel/cets-data-models).

| Data | pytom-match-pick files | CETS objects | Directions |
|---|---|---|---|
| Picked particles | `<tomo>_particles.star` (written by `pytom_extract_candidates.py`, with or without `--relion5-compat`) | `PointMatrixSet3D` (positions, orientations, scores) | pytom → CETS, CETS → pytom |
| Scores map | `<tomo>_scores.mrc` (written by `pytom_match_template.py`) | `ProbabilityMap3D` (`value_type: score`) | pytom → CETS, CETS → pytom |

The job file (`<tomo>_job.json`) next to a star file is found by name
(`pytom_tm.utils.job.find_job`): it gives the size and voxel size of the tomogram and links the
particles to the scores map they were extracted from.

## Installation

Python 3.11 or newer. The CETS data model is not declared as a dependency, so install it first:

```bash
pip install -e /path/to/cets-data-models
pip install -e .          # or: pip install -e .[dev] for the development tools
```

This installs the `pytom_tm` package and the `cets-pytom_tm` command. pytom-match-pick itself is
not needed.

## Command line

```text
cets-pytom_tm {coordinates,scores} {to-cets,from-cets} [options]
```

```bash
# Particles: pytom -> CETS -> pytom
cets-pytom_tm coordinates to-cets --star TS_03_particles.star --tomo-id TS_03 \
    --annotated-object ribosome --score-annotation-id pytom_tm_scores_TS_03 -o coords.yaml
cets-pytom_tm coordinates from-cets -i coords.yaml -o out/TS_03_particles.star --cut-off 0.2

# Scores map: pytom -> CETS -> pytom
cets-pytom_tm scores to-cets --scores-map TS_03_scores.mrc --tomo-id TS_03 -o scores.yaml
cets-pytom_tm scores from-cets -i scores.yaml --output-dir out
```

`to-cets` writes a CETS yaml file with the entry `coordinates` or `scores`; `from-cets` reads
it. Every sub-command has `--help`. Errors print a one-line message and exit with code 1; put
`--debug` before the data type to also print the traceback.

| `coordinates to-cets` option | Meaning |
|---|---|
| `--star` (required) | Particles star file. |
| `--tomo-id` (required) | Id of the CETS tomogram (`source_tomogram_id`). It also names the coordinate systems (`<tomo id>_array`, `<tomo id>_physical`), so it must be the id of the CETS tomogram. |
| `--tomo-name` | Tomogram name (`rlnMicrographName` / `rlnTomoName`) of the particles to convert; needed if the star file has the particles of several tomograms (e.g. merged with `pytom_merge_stars.py`). |
| `--job` | Job file. Default: the one next to the star file, if found, unless `--no-job`. Needed for `--relion5-compat` star files. |
| `--reference-map` | mrc file on the voxel grid the particles were picked in: gives the tomogram size without a job and checks that the particles are inside the grid. |
| `--pixel-size` | Sampling rate of the tomogram (Å). Default: the job, the star file or the reference map, in this order. |
| `--annotated-object` | What the template represents (e.g. `ribosome`); template matching does not record it. |
| `--score-annotation-id` | Id of the scores map annotation the particles were extracted from (`derived_from_annotation_id`). |
| `--annotation-id` | Id of the annotation. Default: `pytom_tm_coords_<tomo id>`. |

| `coordinates from-cets` option | Meaning |
|---|---|
| `--tomo-name` | Name written in the star file. Default: the one of `--job`, else the source tomogram id. |
| `--pixel-size` | Sampling rate (Å). Default: the one of `--job`, else the one of the annotation. |
| `--cut-off`, `--search-std` | Values of `rlnCutOff` and `rlnSearchStd`, which CETS does not store. |
| `--relion5-compat` | Write the positions in Å from the centre of the tomogram, as `pytom_extract_candidates.py --relion5-compat`. Needs `--job`, or `--tomo-shape X Y Z` and a pixel size. |

| `scores` options | Meaning |
|---|---|
| `to-cets --threshold` | Score above which voxels were particle candidates (the `rlnCutOff` of the particles). |
| `to-cets --pixel-size` | Sampling rate (Å). Default: the one of the map header; an error is raised if neither is known. |
| `from-cets --tomo-name` | Name of the map, `<tomo name>_scores.mrc`. Default: the source tomogram id. |
| `from-cets --copy` | Copy the map instead of linking it (it is always copied if its header sampling rate is not the CETS one). |

## Python API

```python
from pytom_tm.converters.coordinates3d import PytomCoordinates3D
from pytom_tm.converters.score_tomogram import PytomScoreTomogram
from pytom_tm.utils.job import find_job

star = "TS_03_particles.star"
job = find_job(star)
scores = PytomScoreTomogram().pytom_tm_to_cets("TS_03_scores.mrc", "TS_03")
coords = PytomCoordinates3D().pytom_tm_to_cets(star, "TS_03", job=job, score_annotation_id=scores.id)
PytomCoordinates3D().cets_to_pytom_tm(coords, "out/TS_03_particles.star", tomo_name="TS_03")
```

## Conventions

- **Positions** (`origin3D`) are 0-based voxel positions in the tomogram, as pytom-match-pick
  writes them by default (`rlnCoordinateX/Y/Z`). The `--relion5-compat` star files store them
  in Å from the centre of the tomogram (`rlnCenteredCoordinateX/Y/ZAngst`); they are converted
  back undoing the operation of pytom-match-pick, which needs the tomogram size and, for jobs
  created from RELION 5 tomograms, the binning and tilt-series pixel size of the job.
- **Orientations.** `matrix3D` is the rotation that brings the template into the particle,
  built from the RELION ZYZ angles: `R = Rz(-psi) @ Ry(-tilt) @ Rz(-rot)`. It is the inverse
  of the matrix Scipion stores, and the same rotation cets-scipion gives for the particle.
- **Scores** are `rlnLCCmax`. Particles keep the order of the star file (descending score).

## Limitations

- `rlnSearchStd` is not kept, and `rlnCutOff` only as the threshold of the scores map.
- The angles map (`<tomo>_angles.mrc`) and the job parameters (template, mask, angular search,
  symmetry, ...) are not converted, so a scores map written back is not enough to extract
  particles again.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest pytom_tm/tests/tests_*.py  # the test files must be given: they are named tests_*.py
ruff check . && ruff format .
mypy pytom_tm
```

### Tests

| File | What it tests |
|---|---|
| `pytom_tm/tests/tests_cets_pytom_tm.py` | pytom-match-pick → CETS: the consistency of the test data, the star file reader, the Euler angles, the job file (found by name, relative paths, RELION5 jobs), the coordinates in both star flavours and the scores map. |
| `pytom_tm/tests/tests_cets_to_pytom_tm.py` | CETS → pytom-match-pick: star files (round trips in both flavours, centred positions, pixel size, optional columns) and scores maps (links and copies), also from the CETS yaml files of the test data. |
| `pytom_tm/tests/tests_cli.py` | Every `cets-pytom_tm` sub-command: the outputs are compared with the values of the input files (for `from-cets`, the CETS yaml files of the test data give back the files pytom-match-pick wrote), plus the round trips, the options, the errors and the usage errors. |

```bash
pytest pytom_tm/tests/tests_cli.py                    # one file
pytest pytom_tm/tests/tests_cli.py -k scores          # some tests
```

All the tests use the data of a real pytom-match-pick template matching of the tomogram
TS_03 (`pytom_tm/tests/data`, `DATA_DIR` in `pytom_tm/tests/__init__.py`):

| File | Content |
|---|---|
| `inputs/TS_03.mrc` | The tomogram (240 x 232 x 70 voxels, 21.6 Å/voxel). |
| `inputs/Reference.mrc`, `inputs/Mask.mrc` | The template and its mask (24 x 24 x 24 voxels). |
| `inputs/TS_03.tlt`, `inputs/TS_03_dose.txt`, `inputs/TS_03.defocus` | Tilt angles, accumulated dose and defocus (IMOD format) of the 36 tilts of the tilt series. |
| `outputs/TS_03_job.json` | The job file: tomogram size and voxel size, and the tilt-series metadata of the inputs. Its paths are absolute paths of the machine where it ran, so the tests do not rely on them. |
| `outputs/TS_03_scores.mrc` | Scores map: best score of each voxel. |
| `outputs/TS_03_angles.mrc` | Index of the best of the 108 rotations of each voxel. |
| `outputs/TS_03_particles.star` | The 401 particles extracted with `--relion5-compat` (cut-off 0.212666): positions in Å from the centre of the tomogram, angles and scores. |
| `outputs/pytom_tm_coords_TS_03_pytom_tm_to_cets.yaml` | CETS `PointMatrixSet3D` of the particles, as `PytomCoordinates3D().pytom_tm_to_cets` writes it for the star and job files: the input of the `coordinates from-cets` tests. |
| `outputs/pytom_tm_scores_TS_03_pytom_tm_to_cets.yaml` | CETS `ProbabilityMap3D` of the scores map (threshold: the cut-off): the input of the `scores from-cets` tests. Its `path` (`./outputs/TS_03_scores.mrc`) is relative to `pytom_tm/tests/data`, so the tests read it from that directory. |

The other files the tests need (the star file without `--relion5-compat`, a job of a RELION5
tomogram, a star file with the particles of several tomograms, ...) are derived from these in
a temporary directory by the helpers of `pytom_tm/tests/__init__.py`. The files in
`pytom_tm/tests/data` are never modified by the tests. A test checks that the yaml files are
still what the converters give for the pytom-match-pick files; if the converters change, write
them again (`pytom_tm_to_cets` with an `output_directory`) and make the path of the scores map
relative to `pytom_tm/tests/data` again.
