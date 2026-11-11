STAR_EXT = ".star"
MRC_EXT = ".mrc"
JSON_EXT = ".json"

# Names of the files written by pytom-match-pick for a tomogram (<tomo_id> is the name of the
# tomogram file without extension), all in the output directory of the job:
#   <tomo_id>_job.json        parameters of the template matching job (pytom_match_template.py)
#   <tomo_id>_scores.mrc      scores map: best score of each voxel over all the rotations
#   <tomo_id>_angles.mrc      index, in the list of rotations, of the best rotation per voxel
#   <tomo_id>_particles.star  extracted particles (pytom_extract_candidates.py)
JOB_SUFFIX = "_job.json"
SCORES_SUFFIX = "_scores.mrc"
ANGLES_SUFFIX = "_angles.mrc"
PARTICLES_SUFFIX = "_particles.star"

# Data block of the particles star file
PARTICLES_BLOCK = "particles"

# Columns of the particles star file. By default the positions are 0-based voxel indices of
# the tomogram (COORD_*), with its name and voxel size in MICROGRAPH_NAME / DETECTOR_PIXEL_SIZE.
COORD_X = "rlnCoordinateX"
COORD_Y = "rlnCoordinateY"
COORD_Z = "rlnCoordinateZ"
MICROGRAPH_NAME = "rlnMicrographName"
DETECTOR_PIXEL_SIZE = "rlnDetectorPixelSize"
# With --relion5-compat the positions are in Å relative to the centre of the tomogram
# (CENTERED_COORD_*), with its name and voxel size in TOMO_NAME / TILT_SERIES_PIXEL_SIZE.
CENTERED_COORD_X = "rlnCenteredCoordinateXAngst"
CENTERED_COORD_Y = "rlnCenteredCoordinateYAngst"
CENTERED_COORD_Z = "rlnCenteredCoordinateZAngst"
TOMO_NAME = "rlnTomoName"
TILT_SERIES_PIXEL_SIZE = "rlnTomoTiltSeriesPixelSize"
# Common columns: RELION Euler angles (degrees), best score of the particle (LCCmax), score
# cut-off used in the extraction and standard deviation of the template matching search.
ANGLE_ROT = "rlnAngleRot"
ANGLE_TILT = "rlnAngleTilt"
ANGLE_PSI = "rlnAnglePsi"
LCC_MAX = "rlnLCCmax"
CUT_OFF = "rlnCutOff"
SEARCH_STD = "rlnSearchStd"

# pytom-match-pick removes this prefix of the tomogram name with --relion5-compat
RELION5_TOMO_PREFIX = "rec_"

# Fields of the job file
JOB_TOMO_ID = "tomo_id"
JOB_TOMOGRAM = "tomogram"
JOB_TEMPLATE = "template"
JOB_OUTPUT_DIR = "output_dir"
JOB_VOXEL_SIZE = "voxel_size"
JOB_TOMO_SHAPE = "tomo_shape"
JOB_TS_METADATA = "ts_metadata"
JOB_CLASS = "__class__"
JOB_BINNING = "binning"
JOB_TS_PIXEL_SIZE = "tilt_series_pixel_size"
# Tilt-series metadata class of the jobs created from RELION5 tomograms: the centred positions
# written with --relion5-compat refer to the unbinned tilt-series pixel size.
RELION_TS_METADATA_CLASS = "RelionTiltSeriesMetaData"
